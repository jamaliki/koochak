"""Clip dense gradients without squaring large values in their original scale."""

from __future__ import annotations

import math
from typing import Iterable, NamedTuple

import torch


class GradientClipStats(NamedTuple):
    norm: float
    coefficient: float
    used_scaled_norm: bool


def _global_norm(norms: Iterable[torch.Tensor]) -> float:
    # Only these per-tensor scalars use FP64; gradient tensors keep their dtype.
    values = torch.stack(tuple(norms))
    if values.device.type == "mps":
        values = values.cpu()
    return float(values.double().norm())


def _scaled_norm(grad: torch.Tensor, root: float) -> torch.Tensor:
    scaled = grad / root
    scaled.div_(root)
    return scaled.norm()


def _scaled_clip(grads: list[torch.Tensor], maximum: float) -> GradientClipStats:
    largest = float(torch.stack(torch._foreach_norm(grads, math.inf)).max())
    if not math.isfinite(largest):
        raise FloatingPointError("Gradient entries contain NaN or infinity; optimizer step stopped.")
    # Two normal-range factors avoid a subnormal reciprocal near the FP32 limit.
    root = math.sqrt(largest)
    scaled_norm = _global_norm(_scaled_norm(grad, root) for grad in grads)
    norm = largest * scaled_norm
    coefficient = min(1.0, (maximum / largest) / (scaled_norm + 1e-6 / largest))
    if coefficient < 1.0:
        multiplier = math.sqrt(coefficient)
        torch._foreach_mul_(grads, multiplier)
        torch._foreach_mul_(grads, multiplier)
    return GradientClipStats(norm, coefficient, True)


@torch.no_grad()
def clip_grad_norm_(parameters: Iterable[torch.Tensor], max_norm: float) -> GradientClipStats:
    """Clip the joint L2 norm and reject nonfinite entries before mutating them.

    The usual path uses foreach reductions and one scalar device-to-host sync.
    An overflowing norm triggers scaled reductions with at most one gradient's
    worth of temporary storage. No full-sized FP64 copies are needed.
    """
    maximum = float(max_norm)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("max_norm must be finite and nonnegative.")
    grads = [param.grad.detach() for param in parameters
             if param.grad is not None and param.grad.numel()]
    if not grads:
        return GradientClipStats(0.0, 1.0, False)
    if any(grad.layout != torch.strided or not grad.is_floating_point() for grad in grads):
        raise TypeError("Gradient clipping requires dense floating-point gradients.")
    if any(grad.device != grads[0].device for grad in grads):
        raise ValueError("Gradient clipping requires all gradients on the same device.")
    norm = _global_norm(torch._foreach_norm(grads, 2.0))
    coefficient = min(1.0, maximum / (norm + 1e-6))
    # A direct FP32 multiplier can flush subnormal coefficients to zero on CUDA.
    tiny_coefficient = 0.0 < coefficient < torch.finfo(torch.float32).tiny
    if not math.isfinite(norm) or tiny_coefficient:
        return _scaled_clip(grads, maximum)
    torch._foreach_mul_(grads, coefficient)
    return GradientClipStats(norm, coefficient, False)
