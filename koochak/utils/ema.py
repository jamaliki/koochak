from __future__ import annotations
from typing import Dict, Optional
import threading
import torch

class SnapshotLease:
    """Read-only ownership of one completed CPU snapshot bank."""

    def __init__(self, owner: "EMA", bank_index: int, tensors: Dict[str, torch.Tensor]) -> None:
        self._owner = owner
        self._bank_index = bank_index
        self.tensors = tensors
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._owner._release_snapshot(self._bank_index)

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
        self._banks: list[Dict[str, torch.Tensor]] = []
        self._bank_states: list[str] = []
        self._bank_events: list[Dict[str, Optional[torch.cuda.Event]]] = []
        self._bank_decays: list[Dict[str, float]] = []
        self._ready_banks: list[int] = []
        self._bank_lock = threading.Lock()
        self.staging: Dict[str, torch.Tensor] = {}         # compatibility alias for bank 0
        self.pending_event: Dict[str, Optional[torch.cuda.Event]] = {}  # compatibility alias
        self.pending_decay: Dict[str, float] = {}          # compatibility alias
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
        self._banks = [{}, {}]
        self._bank_states = ["free", "free"]
        self._bank_events = []
        self._bank_decays = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # shadow on CPU, fp32
            self.shadow[name] = p.detach().to("cpu", dtype=self.dtype).clone()
            for bank in self._banks:
                if self.offload and p.is_cuda and self.pin_memory:
                    bank[name] = torch.empty_like(self.shadow[name], device="cpu", pin_memory=True)
                else:
                    bank[name] = torch.empty_like(self.shadow[name], device="cpu")
        self._bank_events = [{name: None for name in bank} for bank in self._banks]
        self._bank_decays = [{name: self.decay for name in bank} for bank in self._banks]
        self.staging = self._banks[0]
        self.pending_event = self._bank_events[0]
        self.pending_decay = self._bank_decays[0]

    # ---- public API ----
    @torch.no_grad()
    def update(self, model: torch.nn.Module, decay: Optional[float] = None) -> None:
        self._step_counter += 1
        if self.update_every > 1 and (self._step_counter % self.update_every) != 0:
            return  # thinning

        d = float(self._current_decay() if decay is None else decay)
        self.num_updates += 1

        # Release ready banks that were not leased by the observer, then apply
        # completed D2H copies before claiming the next bank.
        self._release_unoffered_ready()
        self._apply_ready_staged()
        bank_index = self._claim_free_bank()
        if self.offload and any(p.is_cuda for p in model.parameters()):
            if bank_index is None:
                return
            bank = self._banks[bank_index]
            try:
                for name, p in model.named_parameters():
                    if not p.requires_grad or name not in self.shadow:
                        continue
                    src_gpu = p.detach()
                    if src_gpu.dtype is not self.dtype:
                        src_gpu = src_gpu.to(dtype=self.dtype)  # cast on GPU
                    bank[name].copy_(src_gpu, non_blocking=True)
                    ev = self._bank_events[bank_index][name] or torch.cuda.Event(blocking=False)
                    ev.record(torch.cuda.current_stream())
                    self._bank_events[bank_index][name] = ev
                    self._bank_decays[bank_index][name] = d
            except BaseException:
                self._mark_bank_free(bank_index)
                raise
        else:
            if bank_index is not None:
                bank = self._banks[bank_index]
                for name, p in model.named_parameters():
                    if not p.requires_grad or name not in self.shadow:
                        continue
                    src_cpu = p.detach().to("cpu", dtype=self.dtype)
                    bank[name].copy_(src_cpu)
                    self.shadow[name].mul_(d).add_(bank[name], alpha=(1.0 - d))
                self._mark_bank_ready(bank_index)
            else:
                for name, p in model.named_parameters():
                    if not p.requires_grad or name not in self.shadow:
                        continue
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
        try: self.decay = float(state.get("decay", self.decay))  # type: ignore[arg-type]
        except Exception: pass
        prof = state.get("profile", self.profile)
        if isinstance(prof, str): self.profile = prof.lower()
        g = state.get("gamma", None)
        if g is not None:
            try: self.gamma = float(g)
            except Exception: pass
        try: self.num_updates = int(state.get("num_updates", 0))  # type: ignore[arg-type]
        except Exception: self.num_updates = 0

        shadow = state.get("shadow", {})  # type: ignore[assignment]
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
    def offer_snapshot(self) -> Optional[SnapshotLease]:
        """Claim one completed bank for a background observer, without waiting."""
        with self._bank_lock:
            if not self._ready_banks:
                return None
            bank_index = self._ready_banks.pop(0)
            if self._bank_states[bank_index] != "ready":
                return None
            self._bank_states[bank_index] = "leased"
            return SnapshotLease(self, bank_index, self._banks[bank_index])

    def _claim_free_bank(self) -> Optional[int]:
        with self._bank_lock:
            for bank_index, state in enumerate(self._bank_states):
                if state == "free":
                    self._bank_states[bank_index] = "pending"
                    self.staging = self._banks[bank_index]
                    self.pending_event = self._bank_events[bank_index]
                    self.pending_decay = self._bank_decays[bank_index]
                    return bank_index
        return None

    def _mark_bank_ready(self, bank_index: int) -> None:
        with self._bank_lock:
            self._bank_states[bank_index] = "ready"
            if bank_index not in self._ready_banks:
                self._ready_banks.append(bank_index)

    def _mark_bank_free(self, bank_index: int) -> None:
        with self._bank_lock:
            self._bank_states[bank_index] = "free"
            if bank_index in self._ready_banks:
                self._ready_banks.remove(bank_index)
            for name in self._bank_events[bank_index]:
                self._bank_events[bank_index][name] = None

    def _release_unoffered_ready(self) -> None:
        with self._bank_lock:
            ready = list(self._ready_banks)
            self._ready_banks.clear()
        for bank_index in ready:
            self._mark_bank_free(bank_index)

    def _release_snapshot(self, bank_index: int) -> None:
        self._mark_bank_free(bank_index)

    @torch.no_grad()
    def _apply_ready_staged(self) -> None:
        for bank_index, state in enumerate(self._bank_states):
            if state != "pending":
                continue
            events = self._bank_events[bank_index]
            if any(event is None or not event.query() for event in events.values()):
                continue
            self._apply_bank(bank_index)
            self._mark_bank_ready(bank_index)

    @torch.no_grad()
    def _apply_bank(self, bank_index: int) -> None:
        bank = self._banks[bank_index]
        for name, source in bank.items():
            destination = self.shadow[name]
            decay = self._bank_decays[bank_index].get(name, self.decay)
            destination.mul_(decay).add_(source, alpha=(1.0 - decay))
            self._bank_events[bank_index][name] = None

    @torch.no_grad()
    def flush(self) -> None:
        pending = [index for index, state in enumerate(self._bank_states) if state == "pending"]
        if not pending:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for bank_index in pending:
            self._apply_bank(bank_index)
            self._mark_bank_ready(bank_index)
