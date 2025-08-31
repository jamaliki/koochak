from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.cuda.amp import autocast, GradScaler

__all__ = ["autocast_context", "Scaler", "is_fp16"]


def is_fp16(mode: str | None) -> bool:
    return (mode or "fp32").lower() == "fp16"


def autocast_context(mode: str | None, device: torch.device):
    mode = (mode or "fp32").lower()
    if mode == "fp16":
        return autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), dtype=torch.float16)
    if mode == "bf16":
        return autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


class _NoOpScaler:
    def scale(self, loss):  # type: ignore[override]
        return loss

    def step(self, optimizer):  # type: ignore[override]
        optimizer.step()

    def update(self):  # type: ignore[override]
        pass

    def unscale_(self, optimizer):  # type: ignore[override]
        pass

    def state_dict(self):  # for compatibility
        return {}


def Scaler(mode: str | None):
    """Grad scaler for fp16, no-op otherwise."""
    if is_fp16(mode) and torch.cuda.is_available():
        return GradScaler(enabled=True)
    return _NoOpScaler()

