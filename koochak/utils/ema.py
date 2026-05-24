from __future__ import annotations
from typing import Dict, Optional
import torch

class EMA:
    """
    CPU-offloaded EMA of trainable params.

    - shadow weights live on CPU (fp32)
    - per-param pinned CPU staging buffers receive nonblocking GPU->CPU copies
    - EMA update uses previous step's staged values (one-step lag)
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
    ) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.dtype = dtype
        self.offload = bool(offload_to_cpu)
        self.pin_memory = bool(pin_memory)
        self.update_every = int(update_every)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.staging: Dict[str, torch.Tensor] = {}         # pinned CPU buffers
        self.pending_event: Dict[str, Optional[torch.cuda.Event]] = {}
        self.pending_decay: Dict[str, float] = {}          # decay used when launching copy
        self.num_updates: int = 0
        self._step_counter: int = 0
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

    def _current_decay(self) -> float:
        if self.profile == "constant":
            return float(self.decay)
        g = float(self.gamma if self.gamma is not None else 6.94)  # ~10% s_rel
        t = max(1.0, float(self.num_updates + 1))
        return float((1.0 - 1.0 / t) ** (g + 1.0))

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

    # ---- public API ----
    @torch.no_grad()
    def update(self, model: torch.nn.Module, decay: Optional[float] = None) -> None:
        self._step_counter += 1
        if self.update_every > 1 and (self._step_counter % self.update_every) != 0:
            return  # thinning

        d = float(self._current_decay() if decay is None else decay)
        self.num_updates += 1

        # 1) consume any finished copies from prior step and apply EMA math on CPU
        self._apply_ready_staged()

        # 2) launch new nonblocking copies of current params into pinned CPU buffers
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.shadow:
                continue
            if p.is_cuda and self.offload:
                src_gpu = p.detach()
                if src_gpu.dtype is not self.dtype:
                    src_gpu = src_gpu.to(dtype=self.dtype)  # cast on GPU
                # enqueue D2H into pinned buffer
                self.staging[name].copy_(src_gpu, non_blocking=True)
                # record event so we can test readiness next call
                ev = self.pending_event[name] or torch.cuda.Event(blocking=False)
                ev.record(torch.cuda.current_stream())
                self.pending_event[name] = ev
                self.pending_decay[name] = d
            else:
                # model on CPU or offload disabled: do EMA immediately on CPU
                dst = self.shadow[name]
                src_cpu = p.detach().to("cpu", dtype=self.dtype)
                dst.mul_(d).add_(src_cpu, alpha=(1.0 - d))

    def state_dict(self, clone: bool = False) -> Dict[str, object]:
        # ensure latest staged copies are applied so we save fresh EMA
        self.flush()
        return {
            "decay": self.decay,
            "profile": self.profile,
            "gamma": (float(self.gamma) if self.gamma is not None else None),
            "num_updates": self.num_updates,
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
        # apply any staged values whose copies have finished
        for name, ev in self.pending_event.items():
            if ev is None:
                continue
            if not ev.query():
                continue  # copy still in flight; leave it for next call
            dst = self.shadow[name]
            d_used = self.pending_decay.get(name, self.decay)
            src_cpu = self.staging[name]  # already CPU fp32
            dst.mul_(d_used).add_(src_cpu, alpha=(1.0 - d_used))
            self.pending_event[name] = None

    @torch.no_grad()
    def flush(self) -> None:
        # Wait for any in-flight copies and apply them. Events can only exist when CUDA
        # is available — but if a state-dict was loaded from a CUDA host onto a CPU-only
        # one, refuse to silently drop the pending copy.
        if not any(ev is not None for ev in self.pending_event.values()):
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                "EMA has pending CUDA copies but CUDA is not available on this process"
            )
        torch.cuda.synchronize()
        for name, ev in self.pending_event.items():
            if ev is None:
                continue
            dst = self.shadow[name]
            d_used = self.pending_decay.get(name, self.decay)
            dst.mul_(d_used).add_(self.staging[name], alpha=(1.0 - d_used))
            self.pending_event[name] = None
