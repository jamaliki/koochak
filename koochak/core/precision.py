from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.amp import autocast, GradScaler

__all__ = ["autocast_context", "Scaler", "is_fp16", "prepare_compile_backend"]

_TF32_PATCH_ATTR = "__koochak_tf32_state_key_patch__"


def is_fp16(mode: str | None) -> bool:
    return (mode or "fp32").lower() == "fp16"


def autocast_context(mode: str | None, device: torch.device):
    mode = (mode or "fp32").lower()
    if mode == "fp16":
        return autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), dtype=torch.float16)
    if mode == "bf16":
        return autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _cuda_matmul_allow_tf32_via_fp32_precision() -> bool:
    if not hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        return bool(torch._C._get_cublas_allow_tf32())
    precision = str(torch.backends.cuda.matmul.fp32_precision).strip().lower()
    return precision != "ieee"


def prepare_compile_backend() -> None:
    if not hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        return
    try:
        from torch._dynamo import graph_region_tracker
    except Exception:
        return

    current = getattr(graph_region_tracker, "get_global_state_key", None)
    if current is None or getattr(current, _TF32_PATCH_ATTR, False):
        return

    def _patched_get_global_state_key():
        return (
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
            torch.get_num_threads(),
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
            torch.get_default_dtype(),
            torch.are_deterministic_algorithms_enabled(),
            _cuda_matmul_allow_tf32_via_fp32_precision(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch._C._autograd._saved_tensors_hooks_is_enabled(),  # type: ignore[attr-defined]
        )

    setattr(_patched_get_global_state_key, _TF32_PATCH_ATTR, True)
    graph_region_tracker.get_global_state_key = _patched_get_global_state_key


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
