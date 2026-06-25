from __future__ import annotations
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional
import torch

class EMA:
    """
    CPU-offloaded EMA of trainable params.

    - shadow weights live on CPU (fp32)
    - per-param pinned CPU staging buffers receive nonblocking GPU->CPU copies
    - CUDA snapshots use a side stream so the next forward/backward can overlap
    - CPU shadow math runs in a single background worker when offloaded from CUDA
    - `update_every` lets you thin updates to reduce overhead
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        *,
        dtype: torch.dtype = torch.float32,
        profile: str = "constant",
        gamma: Optional[float] = None,
        srel: Optional[float] = None,
        update_every: int = 1,          # do an EMA update every N steps
        offload_to_cpu: bool = True,    # keep EMA on CPU
        pin_memory: bool = True,        # use pinned buffers for async D2H
        compensate_update_every: bool = True,
    ) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.dtype = dtype
        self.offload = bool(offload_to_cpu)
        self.pin_memory = bool(pin_memory)
        self.update_every = max(1, int(update_every))
        # Kept as a public/checkpoint field for compatibility. Thinned EMA
        # updates are always decay-compensated by the actual elapsed step count:
        # an update every N model steps uses base_decay ** N.
        self.compensate_update_every = True
        self.shadow: Dict[str, torch.Tensor] = {}
        self.staging: Dict[str, torch.Tensor] = {}         # pinned CPU buffers
        self.pending_event: Dict[str, Optional[torch.cuda.Event]] = {}
        self.pending_decay: Dict[str, float] = {}          # kept for checkpoint/backcompat introspection
        self._copy_device: Optional[torch.device] = None
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self._copy_done: Optional[torch.cuda.Event] = None
        self._shadow_executor: Optional[ThreadPoolExecutor] = None
        self._shadow_future: Optional[Future[None]] = None
        self.num_updates: int = 0
        self._step_counter: int = 0
        self._last_update_step_counter: int = 0
        self._backup: Optional[Dict[str, torch.Tensor]] = None  # for store/restore

        # schedule
        self.profile = (profile or "constant").lower()
        if self.profile not in ("constant", "power"):
            raise ValueError(f"Unsupported EMA profile: {self.profile}")
        self.gamma = float(gamma) if gamma is not None else (self._gamma_from_srel(float(srel)) if srel is not None else None)

        self._build_from_model(model)

    # ---- helpers for power-EMA ----
    @staticmethod
    def _srel_from_gamma(gamma: float) -> float:
        import math
        g = float(gamma)
        return math.sqrt(g + 1.0) / ((g + 2.0) * math.sqrt(g + 3.0))

    @classmethod
    def _gamma_from_srel(cls, srel: float) -> float:
        import math
        s = max(1e-9, float(srel))
        lo, hi = 1e-6, 1e6
        for _ in range(64):
            mid = math.sqrt(lo * hi)
            val = cls._srel_from_gamma(mid)
            if val > s:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _current_decay(self, elapsed_steps: int = 1) -> float:
        if self.profile == "constant":
            return float(self.decay)
        g = float(self.gamma if self.gamma is not None else 6.94)  # ~10% s_rel
        t = max(1.0, float(self._step_counter))
        prev_t = max(0.0, t - float(max(1, int(elapsed_steps))))
        return float((prev_t / t) ** (g + 1.0))

    def _effective_decay(self, decay: Optional[float], elapsed_steps: int) -> float:
        d = float(self._current_decay(elapsed_steps) if decay is None else decay)
        if self.profile == "constant":
            d = d ** float(max(1, int(elapsed_steps)))
        return d

    def _build_from_model(self, model: torch.nn.Module) -> None:
        # Track only trainable params
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # shadow on CPU, fp32
            self.shadow[name] = p.detach().to("cpu", dtype=self.dtype).clone()
            # pinned staging buffer for async copies
            if self.offload and p.is_cuda and self.pin_memory:
                self.staging[name] = torch.empty_like(self.shadow[name], device="cpu", pin_memory=True)
            else:
                self.staging[name] = torch.empty_like(self.shadow[name], device="cpu")
            self.pending_event[name] = None
            self.pending_decay[name] = self.decay
            if self.offload and p.is_cuda and self._copy_device is None:
                self._copy_device = p.device
        if self._copy_device is not None:
            with torch.cuda.device(self._copy_device):
                self._copy_stream = torch.cuda.Stream(device=self._copy_device)
            self._shadow_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="koochak-ema",
            )

    # ---- public API ----
    @torch.no_grad()
    def update(self, model: torch.nn.Module, decay: Optional[float] = None) -> None:
        self._step_counter += 1
        if self.update_every > 1 and (self._step_counter % self.update_every) != 0:
            return  # thinning

        elapsed_steps = max(1, int(self._step_counter - self._last_update_step_counter))
        d = self._effective_decay(decay, elapsed_steps)
        self.num_updates += 1

        # Ensure the single staging buffer is no longer owned by the previous
        # background shadow update before reusing it for this snapshot.
        self._wait_for_shadow_update()

        if self._copy_stream is not None and self._copy_device is not None:
            self._launch_async_cuda_snapshot(model, d)
            self._last_update_step_counter = self._step_counter
            return

        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.shadow:
                continue
            # model on CPU or offload disabled: do EMA immediately on CPU
            dst = self.shadow[name]
            src_cpu = p.detach().to("cpu", dtype=self.dtype)
            dst.mul_(d).add_(src_cpu, alpha=(1.0 - d))
        self._last_update_step_counter = self._step_counter

    def wait_before_param_mutation(self) -> None:
        """Wait until any async CUDA snapshot no longer reads live parameters."""
        event = self._copy_done
        if event is None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                "EMA has a pending CUDA snapshot but CUDA is not available on this process"
            )
        event.synchronize()
        self._copy_done = None

    def state_dict(self, clone: bool = False) -> Dict[str, object]:
        # ensure latest staged copies are applied so we save fresh EMA
        self.flush()
        return {
            "decay": self.decay,
            "profile": self.profile,
            "gamma": (float(self.gamma) if self.gamma is not None else None),
            "num_updates": self.num_updates,
            "step_counter": self._step_counter,
            "last_update_step_counter": self._last_update_step_counter,
            "update_every": self.update_every,
            "compensate_update_every": self.compensate_update_every,
            # avoid doubling memory; clone only if requested
            "shadow": {k: (v.clone() if clone else v) for k, v in self.shadow.items()},
            "dtype": str(self.dtype),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        decay = state.get("decay", self.decay)
        if isinstance(decay, (int, float)):
            self.decay = float(decay)
        prof = state.get("profile", self.profile)
        if isinstance(prof, str):
            self.profile = prof.lower()
        g = state.get("gamma", None)
        if isinstance(g, (int, float)):
            self.gamma = float(g)
        num_updates = state.get("num_updates", 0)
        self.num_updates = int(num_updates) if isinstance(num_updates, (int, float)) else 0
        step_counter = state.get("step_counter", self.num_updates)
        self._step_counter = int(step_counter) if isinstance(step_counter, (int, float)) else self.num_updates
        update_every = state.get("update_every", self.update_every)
        if isinstance(update_every, (int, float)):
            self.update_every = max(1, int(update_every))
        last_update = state.get("last_update_step_counter", None)
        if isinstance(last_update, (int, float)):
            self._last_update_step_counter = max(0, int(last_update))
        else:
            self._last_update_step_counter = self._derive_last_update_step_counter()
        self.compensate_update_every = True

        shadow = state.get("shadow", {})
        if isinstance(shadow, dict):
            for k, v in shadow.items():
                if k in self.shadow and isinstance(v, torch.Tensor):
                    self.shadow[k].data.copy_(v.to(dtype=self.dtype, device="cpu"))

    # ---- eval helpers ----
    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        # snapshot current model params to restore later (on their device)
        self._backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        # ensure EMA is current
        self.flush()
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in self.shadow:
                continue
            p.data.copy_(self.shadow[n].to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if self._backup is None:
            return
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in self._backup:
                continue
            p.data.copy_(self._backup[n])
        self._backup = None

    # ---- internals ----
    @torch.no_grad()
    def _apply_ready_staged(self) -> None:
        # apply any staged values whose copies have finished. This remains for
        # callers/tests that inspect the old per-parameter pending_event fields;
        # the CUDA fast path uses one all-parameter event plus a background worker.
        if self._copy_done is not None:
            if not self._copy_done.query():
                return
            self._copy_done = None
        for name, ev in self.pending_event.items():
            if ev is None:
                continue
            if not ev.query():
                continue
            dst = self.shadow[name]
            d_used = self.pending_decay.get(name, self.decay)
            src_cpu = self.staging[name]  # already CPU fp32
            dst.mul_(d_used).add_(src_cpu, alpha=(1.0 - d_used))
            self.pending_event[name] = None

    @torch.no_grad()
    def _apply_staged(self, decay: float) -> None:
        for name, dst in self.shadow.items():
            dst.mul_(decay).add_(self.staging[name], alpha=(1.0 - decay))

    @torch.no_grad()
    def _apply_staged_after_event(self, event: torch.cuda.Event, decay: float) -> None:
        event.synchronize()
        self._apply_staged(decay)

    def _wait_for_shadow_update(self) -> None:
        future = self._shadow_future
        if future is None:
            return
        future.result()
        self._shadow_future = None

    @torch.no_grad()
    def _launch_async_cuda_snapshot(self, model: torch.nn.Module, decay: float) -> None:
        assert self._copy_device is not None
        assert self._copy_stream is not None
        assert self._shadow_executor is not None

        with torch.cuda.device(self._copy_device):
            ready = torch.cuda.Event(blocking=False)
            done = torch.cuda.Event(blocking=False)
            ready.record(torch.cuda.current_stream(device=self._copy_device))
            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_event(ready)
                for name, p in model.named_parameters():
                    if not p.requires_grad or name not in self.shadow:
                        continue
                    src = p.detach()
                    if src.dtype is not self.dtype:
                        src = src.to(dtype=self.dtype)
                    self.staging[name].copy_(src, non_blocking=True)
                    self.pending_decay[name] = decay
                done.record(self._copy_stream)
        self._copy_done = done
        self._shadow_future = self._shadow_executor.submit(
            self._apply_staged_after_event,
            done,
            decay,
        )

    @torch.no_grad()
    def flush(self) -> None:
        # Wait for in-flight snapshots and background shadow math so callers see
        # a current EMA. Avoid a global CUDA synchronize; the snapshot event is
        # sufficient and preserves unrelated stream work.
        self.wait_before_param_mutation()
        self._wait_for_shadow_update()
        for name in self.pending_event:
            self.pending_event[name] = None

    def _derive_last_update_step_counter(self) -> int:
        step = max(0, int(self._step_counter))
        every = max(1, int(self.update_every))
        if every == 1:
            return step
        return step - (step % every)
