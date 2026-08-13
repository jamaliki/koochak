"""Fused output-gate plus residual kernels for Pairformer updates."""

from __future__ import annotations

import torch


def _load_triton():
    """Import Triton lazily and raise a focused error if unavailable."""
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("triton Pairformer gated-residual fusion requires triton") from exc
    return triton, tl


triton, tl = _load_triton()


@triton.jit
def _gated_residual_fwd_kernel(
    Z,
    X,
    Gate,
    PairMask,
    Dropout,
    Scale,
    Out,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    COLUMNWISE: tl.constexpr,
    APPLY_MASK: tl.constexpr,
    ASSUME_ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute ``z + dropout * scale * mask * x * sigmoid(gate)`` in one pass."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < TOTAL
    c = offsets % C
    pair_idx = offsets // C
    j = pair_idx % N
    tmp = pair_idx // N
    i = tmp % N
    b = tmp // N

    if ASSUME_ALL_ACTIVE:
        active = valid
    else:
        active = tl.load(PairMask + pair_idx, mask=valid, other=0) != 0
    drop_idx = b * N + tl.where(COLUMNWISE, j, i)
    drop = tl.load(Dropout + drop_idx, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(Scale).to(tl.float32)

    z = tl.load(Z + offsets, mask=valid, other=0.0).to(tl.float32)
    x = tl.load(X + offsets, mask=valid & active, other=0.0).to(tl.float32)
    gate_offsets = b * GS0 + i * GS1 + j * GS2 + c * GS3
    gate = tl.load(Gate + gate_offsets, mask=valid & active, other=0.0).to(tl.float32)
    update = x * tl.sigmoid(gate)
    out = z + tl.where(active, drop * scale * update, 0.0)
    if APPLY_MASK:
        out = tl.where(active, out, 0.0)
    tl.store(Out + offsets, out, mask=valid)


@triton.jit
def _gated_residual_bwd_kernel(
    GradOut,
    X,
    Gate,
    PairMask,
    Dropout,
    Scale,
    GradZ,
    GradX,
    GradGate,
    GradScale,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    COLUMNWISE: tl.constexpr,
    APPLY_MASK: tl.constexpr,
    NEED_GRAD_Z: tl.constexpr,
    NEED_GRAD_X: tl.constexpr,
    NEED_GRAD_GATE: tl.constexpr,
    NEED_GRAD_SCALE: tl.constexpr,
    ASSUME_ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backpropagate the fused output gate and residual add.

    The important trick is that residual backward no longer materializes
    ``dUpdate``.  The upstream gradient is converted directly into ``dX`` and
    ``dGate``, and the trainable residual-scale gradient is reduced from the same
    row tile.
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < TOTAL
    c = offsets % C
    pair_idx = offsets // C
    j = pair_idx % N
    tmp = pair_idx // N
    i = tmp % N
    b = tmp // N

    if ASSUME_ALL_ACTIVE:
        active = valid
    else:
        active = tl.load(PairMask + pair_idx, mask=valid, other=0) != 0
    grad = tl.load(GradOut + offsets, mask=valid, other=0.0).to(tl.float32)
    if APPLY_MASK:
        grad = tl.where(active, grad, 0.0)

    if NEED_GRAD_Z:
        tl.store(GradZ + offsets, grad, mask=valid)

    drop_idx = b * N + tl.where(COLUMNWISE, j, i)
    drop = tl.load(Dropout + drop_idx, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(Scale).to(tl.float32)
    residual_grad = tl.where(active, grad * drop * scale, 0.0)

    if NEED_GRAD_X or NEED_GRAD_GATE or NEED_GRAD_SCALE:
        gate_offsets = b * GS0 + i * GS1 + j * GS2 + c * GS3
        gate = tl.load(Gate + gate_offsets, mask=valid & active, other=0.0).to(tl.float32)
        sig = tl.sigmoid(gate)
        x = tl.load(X + offsets, mask=valid & active, other=0.0).to(tl.float32)
        if NEED_GRAD_X:
            tl.store(GradX + offsets, residual_grad * sig, mask=valid)
        if NEED_GRAD_GATE:
            tl.store(GradGate + offsets, residual_grad * x * sig * (1.0 - sig), mask=valid)
        if NEED_GRAD_SCALE:
            grad_scale = tl.sum(tl.where(valid & active, grad * drop * x * sig, 0.0), axis=0)
            tl.atomic_add(GradScale, grad_scale, sem="relaxed")


_BLOCK_SIZE = 1024


class _PairGatedResidualTriton(torch.autograd.Function):
    """Autograd bridge for fused output gate plus residual add."""

    @staticmethod
    def forward(
        ctx,
        z: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor,
        pair_mask: torch.Tensor,
        dropout_values: torch.Tensor,
        scale: torch.Tensor,
        columnwise: bool,
        apply_mask: bool,
        assume_all_active: bool,
    ) -> torch.Tensor:
        if z.ndim != 4 or x.shape != z.shape or gate.shape != z.shape:
            raise ValueError("fused gated residual expects z/x/gate shape [B, N, N, C]")
        if pair_mask.shape != z.shape[:3]:
            raise ValueError("fused gated residual expects pair_mask shape [B, N, N]")
        if dropout_values.shape != z.shape[:2]:
            raise ValueError("fused gated residual expects dropout_values shape [B, N]")
        if not (z.is_cuda and x.is_cuda and gate.is_cuda and pair_mask.is_cuda):
            raise RuntimeError("fused gated residual requires CUDA tensors")
        if not z.is_contiguous() or not x.is_contiguous() or not pair_mask.is_contiguous():
            raise ValueError("fused gated residual expects contiguous z, x, and pair_mask")
        if not dropout_values.is_contiguous():
            raise ValueError("fused gated residual expects contiguous dropout values")

        out = torch.empty_like(z)
        total = z.numel()
        grid = (triton.cdiv(total, _BLOCK_SIZE),)
        gate_strides = tuple(int(s) for s in gate.stride())
        _gated_residual_fwd_kernel[grid](
            z,
            x,
            gate,
            pair_mask,
            dropout_values,
            scale,
            out,
            gate_strides[0],
            gate_strides[1],
            gate_strides[2],
            gate_strides[3],
            total,
            z.shape[1],
            z.shape[3],
            bool(columnwise),
            bool(apply_mask),
            bool(assume_all_active),
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        ctx.save_for_backward(x, gate, pair_mask, dropout_values, scale)
        ctx.shape = tuple(z.shape)
        ctx.z_dtype = z.dtype
        ctx.scale_dtype = scale.dtype
        ctx.columnwise = bool(columnwise)
        ctx.apply_mask = bool(apply_mask)
        ctx.assume_all_active = bool(assume_all_active)
        ctx.gate_strides = gate_strides
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, gate, pair_mask, dropout_values, scale = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        needs = ctx.needs_input_grad

        grad_z = None
        # When metadata proves an exact full-prefix group, APPLY_MASK is the
        # identity.  Reuse grad_out for dZ instead of allocating/storing it.
        write_grad_z = bool(needs[0] and ctx.apply_mask and not ctx.assume_all_active)
        if needs[0]:
            # In the minimal-mask target path the residual identity derivative is
            # exactly grad_out.  Returning it directly avoids one dense pair-sized
            # allocation at the B172/B180 memory frontier.
            grad_z = (
                torch.empty(ctx.shape, device=grad_out.device, dtype=ctx.z_dtype)
                if write_grad_z
                else grad_out
            )
        grad_x = torch.empty_like(x) if needs[1] else None
        grad_gate = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype) if needs[2] else None
        grad_scale = torch.zeros((), device=grad_out.device, dtype=torch.float32) if needs[5] else None

        if write_grad_z or needs[1] or needs[2] or needs[5]:
            total = grad_out.numel()
            grid = (triton.cdiv(total, _BLOCK_SIZE),)
            _gated_residual_bwd_kernel[grid](
                grad_out,
                x,
                gate,
                pair_mask,
                dropout_values,
                scale,
                grad_z,
                grad_x,
                grad_gate,
                grad_scale,
                ctx.gate_strides[0],
                ctx.gate_strides[1],
                ctx.gate_strides[2],
                ctx.gate_strides[3],
                total,
                ctx.shape[1],
                ctx.shape[3],
                ctx.columnwise,
                ctx.apply_mask,
                write_grad_z,
                needs[1],
                needs[2],
                needs[5],
                ctx.assume_all_active,
                BLOCK_SIZE=_BLOCK_SIZE,
            )

        if grad_scale is not None:
            grad_scale = grad_scale.to(dtype=ctx.scale_dtype)
        return grad_z, grad_x, grad_gate, None, None, grad_scale, None, None, None


def pair_gated_residual_triton(
    z: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    pair_mask: torch.Tensor,
    dropout_values: torch.Tensor,
    scale: torch.Tensor,
    *,
    columnwise: bool,
    apply_mask: bool,
    assume_all_active: bool = False,
) -> torch.Tensor:
    """Apply fused output gating and residual addition.

    ``assume_all_active`` is reserved for callers that can prove the pair mask is
    entirely true; it avoids redundant mask traffic in exact-prefix groups.
    """
    return _PairGatedResidualTriton.apply(
        z.contiguous(),
        x.contiguous(),
        gate,
        pair_mask.contiguous(),
        dropout_values.contiguous(),
        scale,
        bool(columnwise),
        bool(apply_mask),
        bool(assume_all_active),
    )
