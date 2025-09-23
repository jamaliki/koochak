from __future__ import annotations

from typing import Dict, Optional

import torch


class EMA:
    """Exponential Moving Average (EMA) of model parameters.

    - Tracks trainable parameters (requires_grad=True) by name.
    - Stores EMA weights in float32 for numerical stability.
    - Updates are in-place on the EMA buffers: ema = decay * ema + (1 - decay) * param.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        *,
        profile: str = "constant",
        gamma: Optional[float] = None,
        srel: Optional[float] = None,
    ) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.dtype = dtype
        self.device = device
        self.shadow: Dict[str, torch.Tensor] = {}
        self.num_updates: int = 0
        # Schedule selection
        self.profile = (profile or "constant").lower()
        if self.profile not in ("constant", "power"):
            raise ValueError(f"Unsupported EMA profile: {self.profile}")
        self.gamma = float(gamma) if gamma is not None else (self._gamma_from_srel(float(srel)) if srel is not None else None)
        self._build_from_model(model)

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
        # Power EMA: beta_t = (1 - 1/t)^(gamma+1)
        g = float(self.gamma if self.gamma is not None else 6.94)  # ~10% s_rel
        t = max(1.0, float(self.num_updates + 1))
        return float((1.0 - 1.0 / t) ** (g + 1.0))

    def _build_from_model(self, model: torch.nn.Module) -> None:
        dev = self.device
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            t = p.detach().data
            if dev is None:
                dev = t.device
            self.shadow[name] = t.to(device=dev, dtype=self.dtype).clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        d = self._current_decay()
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.shadow:
                continue
            src = p.detach().data.to(dtype=self.dtype)
            dst = self.shadow[name]
            # ema = d * ema + (1 - d) * param
            dst.mul_(d).add_(src, alpha=(1.0 - d))

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "profile": self.profile,
            "gamma": (float(self.gamma) if self.gamma is not None else None),
            "num_updates": self.num_updates,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
            "dtype": str(self.dtype),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        try:
            self.decay = float(state.get("decay", self.decay))  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            prof = state.get("profile", self.profile)
            if isinstance(prof, str):
                self.profile = prof.lower()
        except Exception:
            pass
        try:
            g = state.get("gamma", None)
            if g is not None:
                self.gamma = float(g)
        except Exception:
            pass
        try:
            self.num_updates = int(state.get("num_updates", 0))  # type: ignore[arg-type]
        except Exception:
            self.num_updates = 0
        shadow = state.get("shadow", {})  # type: ignore[assignment]
        if isinstance(shadow, dict):
            # Only load matching keys
            for k, v in shadow.items():
                if k in self.shadow and isinstance(v, torch.Tensor):
                    self.shadow[k].data.copy_(v.to(dtype=self.dtype, device=self.shadow[k].device))
