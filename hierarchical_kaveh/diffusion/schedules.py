"""Pallatom's scaled log-normal EDM noise schedule."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sample_training_sigma(
    generator: torch.Generator,
    *,
    p_mean: float = -1.2,
    p_std: float = 1.5,
    sigma_data: float = 16.0,
) -> Tensor:
    """Sample ``sigma_data * exp(N(p_mean, p_std**2))``."""

    if p_std <= 0 or sigma_data <= 0:
        raise ValueError("p_std and sigma_data must be positive")
    return float(sigma_data) * (
        float(p_mean) + float(p_std) * torch.randn((), generator=generator)
    ).exp()


def sigma_from_probability(
    probability: Tensor,
    *,
    p_mean: float = -1.2,
    p_std: float = 1.5,
    sigma_data: float = 16.0,
) -> Tensor:
    """Map normalized time through Pallatom's inverse-normal schedule."""

    if p_std <= 0 or sigma_data <= 0:
        raise ValueError("p_std and sigma_data must be positive")
    probability = probability.to(torch.float64).clamp(1.0e-12, 1.0 - 1.0e-12)
    normal = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    return float(sigma_data) * (float(p_mean) + float(p_std) * normal).exp()


__all__ = ["sample_training_sigma", "sigma_from_probability"]
