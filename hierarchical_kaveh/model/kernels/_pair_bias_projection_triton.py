"""Fused pair-bias LayerNorm + projection kernels.

The ordinary transformer builds pair attention bias as:

    b = Linear(LayerNorm(p))          # [B, N, N, H]
    b = b.permute(0, 3, 1, 2)        # [B, H, N, N]

For the promoted H100 training shapes this is dense pair-tensor traffic over
millions of rows with only 64 channels and 12 heads. This Triton path fuses the
last-dim LayerNorm and the small head projection, and writes directly to the
attention-bias layout.
"""

from __future__ import annotations

import os
from contextlib import nullcontext as profile_range

import torch



def _load_triton():
    """Import Triton lazily and raise a focused error if unavailable."""
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Triton pair-bias projection requires triton") from exc
    return triton, tl


triton, tl = _load_triton()


def _active_forward_enabled() -> bool:
    """Return whether residue_lengths should also compact the forward projection."""
    return os.environ.get("KAVEH_PAIR_BIAS_ACTIVE_FWD", "1") != "0"


def active_pair_lengths_from_token_mask(
    mask: torch.Tensor | None,
    *,
    batch: int,
    seqlen: int,
    num_registers: int,
    pair_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return compact residue lengths for the padding-aware projection contract."""
    if mask is None:
        return None
    if mask.shape != (batch, seqlen):
        raise ValueError(f"mask must have shape {(batch, seqlen)}, got {tuple(mask.shape)}")

    # Batches are prefix-padded.  Device-side summation keeps this helper cheap
    # enough to call in every layer; the FA3 wrapper has an opt-in prefix check
    # for debugging loader bugs without adding a production synchronization.
    token_lengths = mask.to(dtype=torch.bool, device=device).sum(dim=1, dtype=torch.int32)
    pair_lengths = token_lengths - int(num_registers)
    return torch.clamp(pair_lengths, min=0, max=int(pair_size)).to(dtype=torch.int32).contiguous()


@triton.jit
def _pair_bias_projection_fwd_kernel(
    X,
    LnWeight,
    LnBias,
    ProjWeight,
    Out,
    Mean,
    Rstd,
    ROWS: tl.constexpr,
    IN_N: tl.constexpr,
    OUT_N: tl.constexpr,
    NUM_REGISTERS: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fuse pair-bias LayerNorm, head projection, and register-offset store."""
    row_offsets = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    heads = tl.arange(0, BLOCK_H)

    row_mask = row_offsets < ROWS
    col_mask = cols < C
    head_mask = heads < H
    x_offsets = row_offsets[:, None] * C + cols[None, :]

    x = tl.load(X + x_offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    x = tl.where(col_mask[None, :], x, 0.0)
    mean = tl.sum(x, axis=1) / C
    centered = tl.where(row_mask[:, None] & col_mask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / C
    rstd = tl.rsqrt(var + EPS)

    ln_w = tl.load(LnWeight + cols, mask=col_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(LnBias + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = centered * rstd[:, None] * ln_w[None, :] + ln_b[None, :]
    y = tl.where(row_mask[:, None] & col_mask[None, :], y, 0.0)

    proj_w_t = tl.load(
        ProjWeight + heads[None, :] * C + cols[:, None],
        mask=col_mask[:, None] & head_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    out = tl.dot(y, proj_w_t, out_dtype=tl.float32)

    # Input rows are compact residue-residue pairs.  Attention logits include
    # register tokens at the front, so the store shifts residue indices by
    # NUM_REGISTERS and leaves all register interactions at zero bias.
    pair_index = row_offsets % (IN_N * IN_N)
    i = pair_index // IN_N
    j = pair_index - i * IN_N
    batch = row_offsets // (IN_N * IN_N)
    out_i = i + NUM_REGISTERS
    out_j = j + NUM_REGISTERS
    out_offsets = ((batch[:, None] * H + heads[None, :]) * OUT_N + out_i[:, None]) * OUT_N + out_j[:, None]
    tl.store(Out + out_offsets, out, mask=row_mask[:, None] & head_mask[None, :])
    tl.store(Mean + row_offsets, mean, mask=row_mask)
    tl.store(Rstd + row_offsets, rstd, mask=row_mask)


@triton.jit
def _pair_bias_projection_fwd_active_kernel(
    X,
    ResidueLengths,
    LnWeight,
    LnBias,
    ProjWeight,
    Out,
    Mean,
    Rstd,
    IN_N: tl.constexpr,
    OUT_N: tl.constexpr,
    NUM_REGISTERS: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Forward projection over only the active prefix square for one batch item."""
    block = tl.program_id(0)
    batch = tl.program_id(1)
    residue_len = tl.load(ResidueLengths + batch)
    active_rows = residue_len * residue_len
    block_start = block * BLOCK_R
    if block_start >= active_rows:
        return

    active_offsets = block_start + tl.arange(0, BLOCK_R)
    safe_len = tl.maximum(residue_len, 1)
    i = active_offsets // safe_len
    j = active_offsets - i * safe_len
    active_mask = active_offsets < active_rows
    cols = tl.arange(0, BLOCK_C)
    heads = tl.arange(0, BLOCK_H)
    col_mask = cols < C
    head_mask = heads < H

    full_rows = batch * IN_N * IN_N + i * IN_N + j
    x_offsets = full_rows[:, None] * C + cols[None, :]
    x = tl.load(X + x_offsets, mask=active_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    x = tl.where(col_mask[None, :], x, 0.0)
    mean = tl.sum(x, axis=1) / C
    centered = tl.where(active_mask[:, None] & col_mask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / C
    rstd = tl.rsqrt(var + EPS)

    ln_w = tl.load(LnWeight + cols, mask=col_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(LnBias + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = centered * rstd[:, None] * ln_w[None, :] + ln_b[None, :]
    y = tl.where(active_mask[:, None] & col_mask[None, :], y, 0.0)

    proj_w_t = tl.load(
        ProjWeight + heads[None, :] * C + cols[:, None],
        mask=col_mask[:, None] & head_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    out = tl.dot(y, proj_w_t, out_dtype=tl.float32)

    out_i = i + NUM_REGISTERS
    out_j = j + NUM_REGISTERS
    out_offsets = ((batch * H + heads[None, :]) * OUT_N + out_i[:, None]) * OUT_N + out_j[:, None]
    tl.store(Out + out_offsets, out, mask=active_mask[:, None] & head_mask[None, :])
    tl.store(Mean + full_rows, mean, mask=active_mask)
    tl.store(Rstd + full_rows, rstd, mask=active_mask)


@triton.jit
def _pair_bias_projection_bwd_kernel(
    GradOut,
    X,
    LnWeight,
    LnBias,
    ProjWeight,
    Mean,
    Rstd,
    GradX,
    GradLnWeightPartial,
    GradLnBiasPartial,
    GradProjWeightPartial,
    ROWS: tl.constexpr,
    IN_N: tl.constexpr,
    OUT_N: tl.constexpr,
    NUM_REGISTERS: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    NEED_GRAD_X: tl.constexpr,
    NEED_GRAD_LN_WEIGHT: tl.constexpr,
    NEED_GRAD_LN_BIAS: tl.constexpr,
    NEED_GRAD_PROJ_WEIGHT: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Backpropagate fused pair-bias LayerNorm and projection."""
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    heads = tl.arange(0, BLOCK_H)

    row_mask = row_offsets < ROWS
    col_mask = cols < C
    head_mask = heads < H
    x_offsets = row_offsets[:, None] * C + cols[None, :]

    x = tl.load(X + x_offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    mean = tl.load(Mean + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
    xhat = tl.where(row_mask[:, None] & col_mask[None, :], (x - mean[:, None]) * rstd[:, None], 0.0)

    ln_w = tl.load(LnWeight + cols, mask=col_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(LnBias + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = xhat * ln_w[None, :] + ln_b[None, :]
    y = tl.where(row_mask[:, None] & col_mask[None, :], y, 0.0)

    # Gather the gradient from the final [B, H, N_with_registers, N_with_registers]
    # bias layout back to the compact residue-pair row layout used by X.
    pair_index = row_offsets % (IN_N * IN_N)
    i = pair_index // IN_N
    j = pair_index - i * IN_N
    batch = row_offsets // (IN_N * IN_N)
    out_i = i + NUM_REGISTERS
    out_j = j + NUM_REGISTERS
    grad_offsets = ((batch[:, None] * H + heads[None, :]) * OUT_N + out_i[:, None]) * OUT_N + out_j[:, None]
    grad_out = tl.load(
        GradOut + grad_offsets,
        mask=row_mask[:, None] & head_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    proj_w = tl.load(
        ProjWeight + heads[:, None] * C + cols[None, :],
        mask=head_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    grad_y = tl.dot(grad_out, proj_w, out_dtype=tl.float32)
    grad_y = tl.where(row_mask[:, None] & col_mask[None, :], grad_y, 0.0)

    if NEED_GRAD_X:
        grad_weighted = grad_y * ln_w[None, :]
        grad_sum = tl.sum(grad_weighted, axis=1)
        grad_xhat_sum = tl.sum(grad_weighted * xhat, axis=1)
        dx = (grad_weighted - (grad_sum[:, None] + xhat * grad_xhat_sum[:, None]) / C) * rstd[:, None]
        tl.store(GradX + x_offsets, dx, mask=row_mask[:, None] & col_mask[None, :])

    if NEED_GRAD_LN_WEIGHT:
        grad_ln_w = tl.sum(tl.where(row_mask[:, None] & col_mask[None, :], grad_y * xhat, 0.0), axis=0)
        tl.store(GradLnWeightPartial + pid * C + cols, grad_ln_w, mask=col_mask)

    if NEED_GRAD_LN_BIAS:
        grad_ln_b = tl.sum(tl.where(row_mask[:, None] & col_mask[None, :], grad_y, 0.0), axis=0)
        tl.store(GradLnBiasPartial + pid * C + cols, grad_ln_b, mask=col_mask)

    if NEED_GRAD_PROJ_WEIGHT:
        grad_proj = tl.dot(tl.trans(grad_out), y, out_dtype=tl.float32)
        proj_offsets = pid * H * C + heads[:, None] * C + cols[None, :]
        tl.store(
            GradProjWeightPartial + proj_offsets,
            grad_proj,
            mask=head_mask[:, None] & col_mask[None, :],
        )


@triton.jit
def _pair_bias_projection_bwd_active_kernel(
    GradOut,
    ResidueLengths,
    X,
    LnWeight,
    LnBias,
    ProjWeight,
    Mean,
    Rstd,
    GradX,
    GradLnWeightPartial,
    GradLnBiasPartial,
    GradProjWeightPartial,
    IN_N: tl.constexpr,
    OUT_N: tl.constexpr,
    NUM_REGISTERS: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    NEED_GRAD_X: tl.constexpr,
    NEED_GRAD_LN_WEIGHT: tl.constexpr,
    NEED_GRAD_LN_BIAS: tl.constexpr,
    NEED_GRAD_PROJ_WEIGHT: tl.constexpr,
    BLOCKS_PER_BATCH: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Backpropagate only active prefix residue-pair rows.

    Active rows are the compact square [0:L, 0:L] inside each dense [N, N]
    residue-pair tensor.  The kernel writes every partial-gradient block,
    including explicit zeros for blocks past ``L * L``.  Inactive ``GradX``
    rows are handled by a separate sparse zeroing kernel.
    """
    block = tl.program_id(0)
    batch = tl.program_id(1)
    residue_len = tl.load(ResidueLengths + batch)
    active_rows = residue_len * residue_len
    block_start = block * BLOCK_R
    cols = tl.arange(0, BLOCK_C)
    heads = tl.arange(0, BLOCK_H)
    col_mask = cols < C
    head_mask = heads < H
    partial_pid = batch * BLOCKS_PER_BATCH + block
    if block_start >= active_rows:
        if NEED_GRAD_LN_WEIGHT:
            tl.store(
                GradLnWeightPartial + partial_pid * C + cols,
                tl.zeros((BLOCK_C,), dtype=tl.float32),
                mask=col_mask,
            )
        if NEED_GRAD_LN_BIAS:
            tl.store(
                GradLnBiasPartial + partial_pid * C + cols,
                tl.zeros((BLOCK_C,), dtype=tl.float32),
                mask=col_mask,
            )
        if NEED_GRAD_PROJ_WEIGHT:
            proj_offsets = partial_pid * H * C + heads[:, None] * C + cols[None, :]
            tl.store(
                GradProjWeightPartial + proj_offsets,
                tl.zeros((BLOCK_H, BLOCK_C), dtype=tl.float32),
                mask=head_mask[:, None] & col_mask[None, :],
            )
        return

    active_offsets = block_start + tl.arange(0, BLOCK_R)
    safe_len = tl.maximum(residue_len, 1)
    i = active_offsets // safe_len
    j = active_offsets - i * safe_len
    active_mask = active_offsets < active_rows

    full_rows = batch * IN_N * IN_N + i * IN_N + j
    x_offsets = full_rows[:, None] * C + cols[None, :]
    x = tl.load(X + x_offsets, mask=active_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    mean = tl.load(Mean + full_rows, mask=active_mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd + full_rows, mask=active_mask, other=0.0).to(tl.float32)
    xhat = tl.where(active_mask[:, None] & col_mask[None, :], (x - mean[:, None]) * rstd[:, None], 0.0)

    ln_w = tl.load(LnWeight + cols, mask=col_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(LnBias + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = xhat * ln_w[None, :] + ln_b[None, :]
    y = tl.where(active_mask[:, None] & col_mask[None, :], y, 0.0)

    out_i = i + NUM_REGISTERS
    out_j = j + NUM_REGISTERS
    grad_offsets = ((batch * H + heads[None, :]) * OUT_N + out_i[:, None]) * OUT_N + out_j[:, None]
    grad_out = tl.load(
        GradOut + grad_offsets,
        mask=active_mask[:, None] & head_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    proj_w = tl.load(
        ProjWeight + heads[:, None] * C + cols[None, :],
        mask=head_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    grad_y = tl.dot(grad_out, proj_w, out_dtype=tl.float32)
    grad_y = tl.where(active_mask[:, None] & col_mask[None, :], grad_y, 0.0)

    if NEED_GRAD_X:
        grad_weighted = grad_y * ln_w[None, :]
        grad_sum = tl.sum(grad_weighted, axis=1)
        grad_xhat_sum = tl.sum(grad_weighted * xhat, axis=1)
        dx = (grad_weighted - (grad_sum[:, None] + xhat * grad_xhat_sum[:, None]) / C) * rstd[:, None]
        tl.store(GradX + x_offsets, dx, mask=active_mask[:, None] & col_mask[None, :])

    if NEED_GRAD_LN_WEIGHT:
        grad_ln_w = tl.sum(tl.where(active_mask[:, None] & col_mask[None, :], grad_y * xhat, 0.0), axis=0)
        tl.store(GradLnWeightPartial + partial_pid * C + cols, grad_ln_w, mask=col_mask)

    if NEED_GRAD_LN_BIAS:
        grad_ln_b = tl.sum(tl.where(active_mask[:, None] & col_mask[None, :], grad_y, 0.0), axis=0)
        tl.store(GradLnBiasPartial + partial_pid * C + cols, grad_ln_b, mask=col_mask)

    if NEED_GRAD_PROJ_WEIGHT:
        grad_proj = tl.dot(tl.trans(grad_out), y, out_dtype=tl.float32)
        proj_offsets = partial_pid * H * C + heads[:, None] * C + cols[None, :]
        tl.store(
            GradProjWeightPartial + proj_offsets,
            grad_proj,
            mask=head_mask[:, None] & col_mask[None, :],
        )


@triton.jit
def _pair_bias_projection_zero_inactive_x_kernel(
    ResidueLengths,
    GradX,
    IN_N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Zero padded/inactive pair rows not visited by the compact active kernel."""
    block = tl.program_id(0)
    batch = tl.program_id(1)
    rows = block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    row_mask = rows < IN_N * IN_N
    col_mask = cols < C
    residue_len = tl.load(ResidueLengths + batch)
    i = rows // IN_N
    j = rows - i * IN_N
    inactive = row_mask & ((i >= residue_len) | (j >= residue_len))
    full_rows = batch * IN_N * IN_N + rows
    offsets = full_rows[:, None] * C + cols[None, :]
    tl.store(
        GradX + offsets,
        tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32),
        mask=inactive[:, None] & col_mask[None, :],
    )


def _block_c(width: int) -> int:
    """Return the padded channel block size required by tl.dot."""
    # tl.dot requires all matrix dimensions to be at least 16. The masks below
    # keep padded lanes inactive for smaller model/test widths.
    return max(16, 1 << (int(width) - 1).bit_length())


def _block_h(heads: int) -> int:
    """Return the padded head block size required by backward tl.dot."""
    # Backward computes grad_y = grad_out @ proj_w with K=BLOCK_H.
    return max(16, 1 << (int(heads) - 1).bit_length())


def _block_rows() -> int:
    """Return the row tile size for pair-bias projection kernels."""
    return 128


class _PairBiasProjectionTriton(torch.autograd.Function):
    """Autograd bridge for fused pair-bias projection Triton kernels."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        proj_weight: torch.Tensor,
        eps: float,
        output_size: int,
        num_registers: int,
        residue_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run fused LayerNorm-projection forward into attention-bias layout."""
        if x.ndim != 4:
            raise ValueError("fused pair-bias projection expects x shape [B, N, N, C]")
        if not x.is_cuda:
            raise RuntimeError("fused pair-bias projection requires CUDA tensors")
        if not x.is_contiguous():
            raise ValueError("fused pair-bias projection expects contiguous x")
        batch, in_n, in_n_j, width = (int(v) for v in x.shape)
        if in_n != in_n_j:
            raise ValueError("fused pair-bias projection expects square pair tensor")
        heads, proj_width = (int(v) for v in proj_weight.shape)
        if proj_width != width:
            raise ValueError("fused pair-bias projection weight shape mismatch")
        if ln_weight.shape != (width,) or ln_bias.shape != (width,):
            raise ValueError("fused pair-bias projection LayerNorm affine shape mismatch")
        output_size = int(output_size)
        num_registers = int(num_registers)
        if output_size < in_n + num_registers:
            raise ValueError("fused pair-bias projection output size is too small")
        if residue_lengths is not None:
            if residue_lengths.shape != (batch,):
                raise ValueError(
                    f"residue_lengths must have shape {(batch,)}, got {tuple(residue_lengths.shape)}"
                )
            if not residue_lengths.is_cuda or residue_lengths.device != x.device:
                raise RuntimeError("residue_lengths must be a CUDA tensor on the same device as x")
            if residue_lengths.dtype != torch.int32:
                raise ValueError("residue_lengths must have dtype int32")
            residue_lengths = residue_lengths.contiguous()

        rows = batch * in_n * in_n
        use_active_forward = residue_lengths is not None and _active_forward_enabled()
        if use_active_forward:
            out = x.new_empty((batch, heads, output_size, output_size))
            if num_registers > 0:
                out[:, :, :num_registers, :].zero_()
                out[:, :, :, :num_registers].zero_()
        else:
            out = x.new_zeros((batch, heads, output_size, output_size))
        mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
        block_r = _block_rows()
        block_c = _block_c(width)
        block_h = _block_h(heads)
        if use_active_forward:
            blocks_per_batch = triton.cdiv(in_n * in_n, block_r)
            _pair_bias_projection_fwd_active_kernel[(blocks_per_batch, batch)](
                x,
                residue_lengths,
                ln_weight,
                ln_bias,
                proj_weight,
                out,
                mean,
                rstd,
                in_n,
                output_size,
                num_registers,
                width,
                heads,
                float(eps),
                BLOCK_R=block_r,
                BLOCK_C=block_c,
                BLOCK_H=block_h,
                num_warps=4,
            )
        else:
            grid = (triton.cdiv(rows, block_r),)
            _pair_bias_projection_fwd_kernel[grid](
                x,
                ln_weight,
                ln_bias,
                proj_weight,
                out,
                mean,
                rstd,
                rows,
                in_n,
                output_size,
                num_registers,
                width,
                heads,
                float(eps),
                BLOCK_R=block_r,
                BLOCK_C=block_c,
                BLOCK_H=block_h,
                num_warps=4,
            )
        saved_lengths = (
            residue_lengths
            if residue_lengths is not None
            else torch.empty((0,), device=x.device, dtype=torch.int32)
        )
        ctx.save_for_backward(x, ln_weight, ln_bias, proj_weight, mean, rstd, saved_lengths)
        ctx.has_residue_lengths = residue_lengths is not None
        ctx.in_n = in_n
        ctx.output_size = output_size
        ctx.num_registers = num_registers
        ctx.width = width
        ctx.heads = heads
        ctx.block_r = block_r
        ctx.block_c = block_c
        ctx.block_h = block_h
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Run fused backward kernels and reduce parameter gradients."""
        x, ln_weight, ln_bias, proj_weight, mean, rstd, residue_lengths = ctx.saved_tensors
        needs = ctx.needs_input_grad
        rows = x.numel() // ctx.width
        blocks_per_batch = triton.cdiv(ctx.in_n * ctx.in_n, ctx.block_r)
        blocks = x.shape[0] * blocks_per_batch if ctx.has_residue_lengths else triton.cdiv(rows, ctx.block_r)
        with profile_range("pair_bias_projection.backward.grad_out_contiguous"):
            grad_out = grad_out.contiguous()

        grad_x = None
        if needs[0]:
            with profile_range("pair_bias_projection.backward.grad_x_alloc"):
                grad_x = torch.empty_like(x)
        grad_ln_weight = None
        grad_ln_bias = None
        grad_proj_weight = None

        def partial(shape, needed: bool) -> torch.Tensor | None:
            if not needed:
                return None
            return torch.empty(shape, device=x.device, dtype=torch.float32)

        grad_ln_weight_partial = partial((blocks, ctx.width), needs[1])
        grad_ln_bias_partial = partial((blocks, ctx.width), needs[2])
        grad_proj_weight_partial = partial((blocks, ctx.heads, ctx.width), needs[3])

        if any(needs[:4]):
            if ctx.has_residue_lengths:
                # Padding-heavy batches win by compacting each example to its
                # active L*L prefix instead of scanning the full N*N pair grid.
                # We still launch the same number of partial blocks per batch so
                # parameter-gradient reduction is a simple dense sum; inactive
                # partials are explicitly written as zero by the kernel.
                with profile_range("pair_bias_projection.backward.active_kernel"):
                    _pair_bias_projection_bwd_active_kernel[(blocks_per_batch, x.shape[0])](
                        grad_out,
                        residue_lengths,
                        x,
                        ln_weight,
                        ln_bias,
                        proj_weight,
                        mean,
                        rstd,
                        grad_x,
                        grad_ln_weight_partial,
                        grad_ln_bias_partial,
                        grad_proj_weight_partial,
                        ctx.in_n,
                        ctx.output_size,
                        ctx.num_registers,
                        ctx.width,
                        ctx.heads,
                        needs[0],
                        needs[1],
                        needs[2],
                        needs[3],
                        blocks_per_batch,
                        BLOCK_R=ctx.block_r,
                        BLOCK_C=ctx.block_c,
                        BLOCK_H=ctx.block_h,
                        num_warps=4,
                    )
                    if needs[0]:
                        # grad_x was allocated with empty_like for speed.  The
                        # active kernel writes only [0:L,0:L], so a tiny sparse
                        # cleanup restores dense autograd semantics without the
                        # cost of zero-initializing the whole [B,N,N,C] tensor.
                        with profile_range("pair_bias_projection.backward.zero_inactive_x"):
                            _pair_bias_projection_zero_inactive_x_kernel[(blocks_per_batch, x.shape[0])](
                                residue_lengths,
                                grad_x,
                                ctx.in_n,
                                ctx.width,
                                BLOCK_R=ctx.block_r,
                                BLOCK_C=ctx.block_c,
                                num_warps=4,
                            )
            else:
                with profile_range("pair_bias_projection.backward.kernel"):
                    _pair_bias_projection_bwd_kernel[(blocks,)](
                        grad_out,
                        x,
                        ln_weight,
                        ln_bias,
                        proj_weight,
                        mean,
                        rstd,
                        grad_x,
                        grad_ln_weight_partial,
                        grad_ln_bias_partial,
                        grad_proj_weight_partial,
                        rows,
                        ctx.in_n,
                        ctx.output_size,
                        ctx.num_registers,
                        ctx.width,
                        ctx.heads,
                        needs[0],
                        needs[1],
                        needs[2],
                        needs[3],
                        BLOCK_R=ctx.block_r,
                        BLOCK_C=ctx.block_c,
                        BLOCK_H=ctx.block_h,
                        num_warps=4,
                    )

        if True:
            if needs[1]:
                grad_ln_weight = grad_ln_weight_partial.sum(dim=0).to(dtype=ln_weight.dtype)
            if needs[2]:
                grad_ln_bias = grad_ln_bias_partial.sum(dim=0).to(dtype=ln_bias.dtype)
            if needs[3]:
                grad_proj_weight = grad_proj_weight_partial.sum(dim=0).to(dtype=proj_weight.dtype)

        return grad_x, grad_ln_weight, grad_ln_bias, grad_proj_weight, None, None, None, None


def pair_bias_projection_triton(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
    output_size: int,
    num_registers: int = 0,
    residue_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project pair features to attention bias with a fused Triton kernel.

    ``residue_lengths`` activates the padding-aware contract: each batch item
    projects only its active prefix square in forward, and backward reads only
    that active square from dBias.  Callers may therefore skip zero-filling
    padded dBias entries upstream when they pass matching active lengths.
    """
    return _PairBiasProjectionTriton.apply(
        x.contiguous(),
        ln_weight,
        ln_bias,
        proj_weight,
        float(eps),
        int(output_size),
        int(num_registers),
        residue_lengths,
    )
