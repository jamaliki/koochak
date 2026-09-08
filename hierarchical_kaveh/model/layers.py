"""Small neural-network primitives used by the single supported model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def init_linear(layer: nn.Linear, kind: str = "normal") -> nn.Linear:
    """Apply the model's intentionally small set of initialization schemes."""

    if kind == "normal":
        nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
    elif kind == "dit":
        nn.init.xavier_uniform_(layer.weight)
    elif kind in {"zero", "final"}:
        nn.init.zeros_(layer.weight)
    elif kind == "gate":
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, -1.0)
    else:
        raise ValueError(f"unknown initialization: {kind}")
    if layer.bias is not None and kind not in {"gate"}:
        nn.init.zeros_(layer.bias)
    return layer


class RMSNorm(nn.Module):
    """RMS normalization with float32 statistics and input-dtype output."""

    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class NonAffineRMSNorm(nn.Module):
    """Post-residual RMS normalization without a learned affine transform."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype)


class UnitRMSNorm(NonAffineRMSNorm):
    """Parameter-free RMS normalization used at fixed-gain pair updates."""


def _masked_rms(x: Tensor, mask: Tensor | None) -> Tensor:
    """Return one float32 RMS, excluding padded stream positions."""

    values = x.float().square().mean(-1)
    if mask is None:
        return values.mean().sqrt()
    weights = mask.to(device=x.device, dtype=values.dtype)
    return ((values * weights).sum() / weights.sum().clamp_min(1.0)).sqrt()


def apply_residual_stage(
    x: Tensor,
    update: Tensor,
    *,
    scale: Tensor,
    post_norm: NonAffineRMSNorm | None,
    mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Add one branch, optionally sandwich-normalize the post-add stream.

    The returned diagnostics are ``[raw_update_rms, post_add_rms,
    post_norm_rms, active]``.  The raw update is measured before residual
    scaling.  Both stream measurements are taken after masking, and the
    normalization is deliberately applied to the residual stream after the
    addition rather than to the branch update itself.
    """

    x = x + scale.to(dtype=update.dtype) * update
    if mask is not None:
        x = x * mask[..., None].to(dtype=x.dtype)
    with torch.no_grad():
        raw_rms = _masked_rms(update, mask)
        post_add_rms = _masked_rms(x, mask)
    if post_norm is not None:
        x = post_norm(x)
        if mask is not None:
            x = x * mask[..., None].to(dtype=x.dtype)
    with torch.no_grad():
        post_norm_rms = _masked_rms(x, mask)
    diagnostics = torch.stack(
        (raw_rms, post_add_rms, post_norm_rms, raw_rms.new_ones(()))
    )
    return x, diagnostics


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm whose scale receives a zero-initialized condition delta."""

    def __init__(self, width: int, condition_dim: int):
        super().__init__()
        self.eps = 1e-6
        self.modulation = init_linear(nn.Linear(condition_dim, width, bias=False), "zero")

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        scale = 1.0 + self.modulation(condition)
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * scale.float()).to(x.dtype)


class DiTAdaLNZero(nn.Module):
    """Canonical DiT adaLN-Zero modulation for attention and MLP branches.

    This follows the official DiT block exactly: two affine-free LayerNorms and
    one ``SiLU -> Linear(6 * width)`` projection yielding shift, scale, and
    residual gate for attention followed by the same three values for the MLP.
    The projection is zero-initialized so both residual branches start closed.
    """

    def __init__(
        self,
        width: int,
        condition_dim: int,
        *,
        bounded: bool = False,
        modulation_limit: float = 0.5,
    ):
        super().__init__()
        if modulation_limit <= 0:
            raise ValueError("modulation_limit must be positive")
        self.bounded = bool(bounded)
        self.modulation_limit = float(modulation_limit)
        self.condition_norm = (
            nn.LayerNorm(condition_dim, elementwise_affine=False)
            if self.bounded else nn.Identity()
        )
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        modulation_width = 4 if self.bounded else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            init_linear(nn.Linear(condition_dim, modulation_width * width, bias=True), "zero"),
        )

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale.to(x.dtype)) + shift.to(x.dtype)

    def modulation_parameters(
        self, condition: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        values = self.adaLN_modulation(self.condition_norm(condition)).chunk(
            4 if self.bounded else 6, dim=-1
        )
        if not self.bounded:
            return values
        limit = self.modulation_limit
        return tuple(limit * value.tanh() for value in values)

    def attention_input(
        self,
        x: Tensor,
        shift: Tensor,
        scale: Tensor,
    ) -> Tensor:
        return self.modulate(self.norm1(x), shift, scale)

    def mlp_input(
        self,
        x: Tensor,
        shift: Tensor,
        scale: Tensor,
    ) -> Tensor:
        return self.modulate(self.norm2(x), shift, scale)


class GEGLU(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, initialization: str | None = None
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, 2 * hidden_dim, bias=False)
        if initialization is not None:
            init_linear(self.projection, initialization)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.projection(x).chunk(2, dim=-1)
        return value * F.gelu(gate)


class FeedForward(nn.Module):
    """Zero-initialized conditioned GEGLU residual."""

    def __init__(
        self,
        width: int,
        condition_dim: int | None,
        expansion: int,
        dropout: float,
        residual_scale: float = 1.0,
        output_initialization: str = "zero",
    ):
        super().__init__()
        self.norm = RMSNorm(width) if condition_dim is None else AdaptiveRMSNorm(width, condition_dim)
        self.up = GEGLU(
            width,
            width * expansion,
            "dit" if output_initialization == "dit" else None,
        )
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.down = init_linear(
            nn.Linear(width * expansion, width, bias=False), output_initialization
        )
        self.register_buffer("residual_scale", torch.tensor(residual_scale), persistent=False)

    def forward(self, x: Tensor, condition: Tensor | None = None) -> Tensor:
        update = self.update(x, condition)
        return x + self.residual_scale.to(update.dtype) * update

    def update(self, x: Tensor, condition: Tensor | None = None) -> Tensor:
        """Return the unscaled branch update used by the residual stage."""

        normalized = self.norm(x) if condition is None else self.norm(x, condition)
        return self.project(normalized)

    def project(self, normalized: Tensor) -> Tensor:
        """Project an already-normalized input, as required by adaLN-Zero."""

        return self.down(self.dropout(self.up(normalized)))


class BoundedFiLM(nn.Module):
    """Identity-initialized positive multiplicative conditioning."""

    def __init__(self, width: int, condition_dim: int, max_multiplier: float):
        super().__init__()
        self.projection = init_linear(nn.Linear(condition_dim, width, bias=False), "zero")
        self.log_limit = math.log(max_multiplier)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        multiplier = torch.exp(self.log_limit * torch.tanh(self.projection(condition).float()))
        return x * multiplier.to(x.dtype)


class TimeEmbedding(nn.Module):
    """EDM Fourier embedding used for both node conditioning and log variance."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, time: Tensor) -> Tensor:
        hidden = torch.cos(2.0 * math.pi * self.input(time).float())
        return self.output(self.norm(hidden).to(self.output.weight.dtype))


class RotaryEmbedding(nn.Module):
    """RoPE with explicit integer/float positions and base 1000."""

    def __init__(self, head_dim: int, base: float = 1000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        frequency = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("frequency", frequency, persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def forward(self, query: Tensor, key: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        # q/k are [B,T,H,D] or packed [T,H,D].
        phase = positions.float()[..., None] * self.frequency.float()
        cos = torch.repeat_interleave(phase.cos(), 2, dim=-1)
        sin = torch.repeat_interleave(phase.sin(), 2, dim=-1)
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)

        def rotate(x: Tensor) -> Tensor:
            return (x.float() * cos + self._rotate_half(x.float()) * sin).to(x.dtype)

        return rotate(query), rotate(key)


def signed_log_separation(delta: Tensor, max_distance: float = 1024.0) -> Tensor:
    """Four bounded channels for signed residue-index separation."""

    value = delta.float()
    magnitude = value.abs()
    log_magnitude = torch.log1p(magnitude) / math.log1p(max_distance)
    return torch.stack(
        (value.sign(), log_magnitude, value.sign() * log_magnitude, torch.rsqrt(magnitude + 1.0)),
        dim=-1,
    )


def checkpoint(module: nn.Module, *args: Tensor, enabled: bool) -> object:
    """Non-reentrant activation checkpointing with an inference fast path."""

    if not enabled or not torch.is_grad_enabled():
        return module(*args)
    return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=False)
