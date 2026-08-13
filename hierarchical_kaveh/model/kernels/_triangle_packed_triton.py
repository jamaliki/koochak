"""Packed trainable input-gate path for Pairformer triangle multiplication.

The standard triangle-multiplication path gates a combined ``[B, N, N, 2H]``
projection, chunks it, then asks einsum/autograd to contract strided views.  This
module writes the two gated operands directly into the contraction layout
``[B, H, N, N]`` and supplies the matching backward, avoiding SplitBackward cat
traffic and giving the batched matmul a contiguous per-channel layout.
"""

from __future__ import annotations

import torch


def _load_triton():
    """Import Triton lazily and raise a focused error if unavailable."""
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("packed triangle multiplication input gate requires triton") from exc
    return triton, tl


triton, tl = _load_triton()


@triton.jit
def _pack_combined_gated_lr_fwd_kernel(
    Combined,
    PairMask,
    Left,
    Right,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Apply input gates from a combined ``[left, right, gate_l, gate_r]`` projection."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hids = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    hid_mask = hids < H

    nn: tl.constexpr = N * N
    b = rows // nn
    rem = rows - b * nn
    i = rem // N
    j = rem - i * N
    rows_i64 = rows.to(tl.int64)
    hids_i64 = hids.to(tl.int64)
    b_i64 = b.to(tl.int64)
    i_i64 = i.to(tl.int64)
    j_i64 = j.to(tl.int64)

    cols: tl.constexpr = 4 * H
    base = rows_i64[:, None] * cols + hids_i64[None, :]
    valid = row_mask[:, None] & hid_mask[None, :]
    active = tl.load(PairMask + rows, mask=row_mask, other=0) != 0
    load_mask = valid & active[:, None]

    proj_l = tl.load(Combined + base, mask=load_mask, other=0.0).to(tl.float32)
    proj_r = tl.load(Combined + base + H, mask=load_mask, other=0.0).to(tl.float32)
    gate_l = tl.load(Combined + base + 2 * H, mask=load_mask, other=0.0).to(tl.float32)
    gate_r = tl.load(Combined + base + 3 * H, mask=load_mask, other=0.0).to(tl.float32)
    left = proj_l * tl.sigmoid(gate_l)
    right = proj_r * tl.sigmoid(gate_r)

    packed_offsets = ((b_i64[:, None] * H + hids_i64[None, :]) * N + i_i64[:, None]) * N + j_i64[:, None]
    tl.store(Left + packed_offsets, left, mask=valid)
    tl.store(Right + packed_offsets, right, mask=valid)


@triton.jit
def _pack_combined_gated_lr_bwd_kernel(
    GradLeft,
    GradRight,
    Combined,
    PairMask,
    GradCombined,
    GLS0: tl.constexpr,
    GLS1: tl.constexpr,
    GLS2: tl.constexpr,
    GLS3: tl.constexpr,
    GRS0: tl.constexpr,
    GRS1: tl.constexpr,
    GRS2: tl.constexpr,
    GRS3: tl.constexpr,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    NEED_GRAD_COMBINED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Backpropagate packed operands into a combined projection tensor."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hids = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    hid_mask = hids < H

    nn: tl.constexpr = N * N
    b = rows // nn
    rem = rows - b * nn
    i = rem // N
    j = rem - i * N
    rows_i64 = rows.to(tl.int64)
    hids_i64 = hids.to(tl.int64)
    b_i64 = b.to(tl.int64)
    i_i64 = i.to(tl.int64)
    j_i64 = j.to(tl.int64)

    valid = row_mask[:, None] & hid_mask[None, :]
    active = tl.load(PairMask + rows, mask=row_mask, other=0) != 0
    load_mask = valid & active[:, None]
    left_grad_offsets = b_i64[:, None] * GLS0 + hids_i64[None, :] * GLS1 + i_i64[:, None] * GLS2 + j_i64[:, None] * GLS3
    right_grad_offsets = b_i64[:, None] * GRS0 + hids_i64[None, :] * GRS1 + i_i64[:, None] * GRS2 + j_i64[:, None] * GRS3
    grad_l = tl.load(GradLeft + left_grad_offsets, mask=load_mask, other=0.0).to(tl.float32)
    grad_r = tl.load(GradRight + right_grad_offsets, mask=load_mask, other=0.0).to(tl.float32)

    if NEED_GRAD_COMBINED:
        cols: tl.constexpr = 4 * H
        base = rows_i64[:, None] * cols + hids_i64[None, :]
        proj_l = tl.load(Combined + base, mask=load_mask, other=0.0).to(tl.float32)
        proj_r = tl.load(Combined + base + H, mask=load_mask, other=0.0).to(tl.float32)
        gate_l = tl.load(Combined + base + 2 * H, mask=load_mask, other=0.0).to(tl.float32)
        gate_r = tl.load(Combined + base + 3 * H, mask=load_mask, other=0.0).to(tl.float32)
        sig_l = tl.sigmoid(gate_l)
        sig_r = tl.sigmoid(gate_r)
        tl.store(GradCombined + base, grad_l * sig_l, mask=valid)
        tl.store(GradCombined + base + H, grad_r * sig_r, mask=valid)
        tl.store(GradCombined + base + 2 * H, grad_l * proj_l * sig_l * (1.0 - sig_l), mask=valid)
        tl.store(GradCombined + base + 3 * H, grad_r * proj_r * sig_r * (1.0 - sig_r), mask=valid)


@triton.jit
def _pack_combined_gated_lr_bwd_contiguous_kernel(
    GradLeft,
    GradRight,
    Combined,
    PairMask,
    GradCombined,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    NEED_GRAD_COMBINED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Backpropagate contiguous packed operands into a combined projection tensor."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hids = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    hid_mask = hids < H

    nn: tl.constexpr = N * N
    b = rows // nn
    rem = rows - b * nn
    i = rem // N
    j = rem - i * N
    rows_i64 = rows.to(tl.int64)
    hids_i64 = hids.to(tl.int64)
    b_i64 = b.to(tl.int64)
    i_i64 = i.to(tl.int64)
    j_i64 = j.to(tl.int64)

    valid = row_mask[:, None] & hid_mask[None, :]
    active = tl.load(PairMask + rows, mask=row_mask, other=0) != 0
    load_mask = valid & active[:, None]
    packed_offsets = ((b_i64[:, None] * H + hids_i64[None, :]) * N + i_i64[:, None]) * N + j_i64[:, None]
    grad_l = tl.load(GradLeft + packed_offsets, mask=load_mask, other=0.0).to(tl.float32)
    grad_r = tl.load(GradRight + packed_offsets, mask=load_mask, other=0.0).to(tl.float32)

    if NEED_GRAD_COMBINED:
        cols: tl.constexpr = 4 * H
        base = rows_i64[:, None] * cols + hids_i64[None, :]
        proj_l = tl.load(Combined + base, mask=load_mask, other=0.0).to(tl.float32)
        proj_r = tl.load(Combined + base + H, mask=load_mask, other=0.0).to(tl.float32)
        gate_l = tl.load(Combined + base + 2 * H, mask=load_mask, other=0.0).to(tl.float32)
        gate_r = tl.load(Combined + base + 3 * H, mask=load_mask, other=0.0).to(tl.float32)
        sig_l = tl.sigmoid(gate_l)
        sig_r = tl.sigmoid(gate_r)
        tl.store(GradCombined + base, grad_l * sig_l, mask=valid)
        tl.store(GradCombined + base + H, grad_r * sig_r, mask=valid)
        tl.store(GradCombined + base + 2 * H, grad_l * proj_l * sig_l * (1.0 - sig_l), mask=valid)
        tl.store(GradCombined + base + 3 * H, grad_r * proj_r * sig_r * (1.0 - sig_r), mask=valid)


@triton.jit
def _pack_combined_gated_lr_outgate_fwd_kernel(
    Combined,
    PairMask,
    Left,
    Right,
    OutGate,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    ASSUME_ALL_ACTIVE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Pack left/right operands and a contiguous output gate from one projection."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hids = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    hid_mask = hids < H

    nn: tl.constexpr = N * N
    b = rows // nn
    rem = rows - b * nn
    i = rem // N
    j = rem - i * N
    rows_i64 = rows.to(tl.int64)
    hids_i64 = hids.to(tl.int64)
    b_i64 = b.to(tl.int64)
    i_i64 = i.to(tl.int64)
    j_i64 = j.to(tl.int64)

    cols: tl.constexpr = 5 * H
    base = rows_i64[:, None] * cols + hids_i64[None, :]
    valid = row_mask[:, None] & hid_mask[None, :]
    if ASSUME_ALL_ACTIVE:
        active = row_mask
    else:
        active = tl.load(PairMask + rows, mask=row_mask, other=0) != 0
    load_mask = valid & active[:, None]

    proj_l = tl.load(Combined + base, mask=load_mask, other=0.0).to(tl.float32)
    proj_r = tl.load(Combined + base + H, mask=load_mask, other=0.0).to(tl.float32)
    gate_l = tl.load(Combined + base + 2 * H, mask=load_mask, other=0.0).to(tl.float32)
    gate_r = tl.load(Combined + base + 3 * H, mask=load_mask, other=0.0).to(tl.float32)
    out_gate = tl.load(Combined + base + 4 * H, mask=valid, other=0.0)
    left = proj_l * tl.sigmoid(gate_l)
    right = proj_r * tl.sigmoid(gate_r)

    packed_offsets = ((b_i64[:, None] * H + hids_i64[None, :]) * N + i_i64[:, None]) * N + j_i64[:, None]
    gate_offsets = rows_i64[:, None] * H + hids_i64[None, :]
    tl.store(Left + packed_offsets, left, mask=valid)
    tl.store(Right + packed_offsets, right, mask=valid)
    tl.store(OutGate + gate_offsets, out_gate, mask=valid)


@triton.jit
def _pack_combined_gated_lr_view_outgate_fwd_kernel(
    Combined,
    PairMask,
    Left,
    Right,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    ASSUME_ALL_ACTIVE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Pack left/right operands while leaving output-gate logits as a Combined view."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hids = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    hid_mask = hids < H

    nn: tl.constexpr = N * N
    b = rows // nn
    rem = rows - b * nn
    i = rem // N
    j = rem - i * N
    rows_i64 = rows.to(tl.int64)
    hids_i64 = hids.to(tl.int64)
    b_i64 = b.to(tl.int64)
    i_i64 = i.to(tl.int64)
    j_i64 = j.to(tl.int64)

    cols: tl.constexpr = 5 * H
    base = rows_i64[:, None] * cols + hids_i64[None, :]
    valid = row_mask[:, None] & hid_mask[None, :]
    if ASSUME_ALL_ACTIVE:
        active = row_mask
    else:
        active = tl.load(PairMask + rows, mask=row_mask, other=0) != 0
    load_mask = valid & active[:, None]

    proj_l = tl.load(Combined + base, mask=load_mask, other=0.0).to(tl.float32)
    proj_r = tl.load(Combined + base + H, mask=load_mask, other=0.0).to(tl.float32)
    gate_l = tl.load(Combined + base + 2 * H, mask=load_mask, other=0.0).to(tl.float32)
    gate_r = tl.load(Combined + base + 3 * H, mask=load_mask, other=0.0).to(tl.float32)
    left = proj_l * tl.sigmoid(gate_l)
    right = proj_r * tl.sigmoid(gate_r)

    packed_offsets = ((b_i64[:, None] * H + hids_i64[None, :]) * N + i_i64[:, None]) * N + j_i64[:, None]
    tl.store(Left + packed_offsets, left, mask=valid)
    tl.store(Right + packed_offsets, right, mask=valid)


@triton.jit
def _pack_combined_gated_lr_outgate_bwd_kernel(
    GradLeft,
    GradRight,
    GradOutGate,
    Combined,
    PairMask,
    GradCombined,
    GLS0: tl.constexpr,
    GLS1: tl.constexpr,
    GLS2: tl.constexpr,
    GLS3: tl.constexpr,
    GRS0: tl.constexpr,
    GRS1: tl.constexpr,
    GRS2: tl.constexpr,
    GRS3: tl.constexpr,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    NEED_GRAD_COMBINED: tl.constexpr,
    ASSUME_ALL_ACTIVE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Backpropagate packed operands and output gate into one projection tensor."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hids = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    hid_mask = hids < H

    nn: tl.constexpr = N * N
    b = rows // nn
    rem = rows - b * nn
    i = rem // N
    j = rem - i * N
    rows_i64 = rows.to(tl.int64)
    hids_i64 = hids.to(tl.int64)
    b_i64 = b.to(tl.int64)
    i_i64 = i.to(tl.int64)
    j_i64 = j.to(tl.int64)

    valid = row_mask[:, None] & hid_mask[None, :]
    if ASSUME_ALL_ACTIVE:
        active = row_mask
    else:
        active = tl.load(PairMask + rows, mask=row_mask, other=0) != 0
    load_mask = valid & active[:, None]
    left_grad_offsets = b_i64[:, None] * GLS0 + hids_i64[None, :] * GLS1 + i_i64[:, None] * GLS2 + j_i64[:, None] * GLS3
    right_grad_offsets = b_i64[:, None] * GRS0 + hids_i64[None, :] * GRS1 + i_i64[:, None] * GRS2 + j_i64[:, None] * GRS3
    grad_l = tl.load(GradLeft + left_grad_offsets, mask=load_mask, other=0.0).to(tl.float32)
    grad_r = tl.load(GradRight + right_grad_offsets, mask=load_mask, other=0.0).to(tl.float32)

    if NEED_GRAD_COMBINED:
        cols: tl.constexpr = 5 * H
        base = rows_i64[:, None] * cols + hids_i64[None, :]
        proj_l = tl.load(Combined + base, mask=load_mask, other=0.0).to(tl.float32)
        proj_r = tl.load(Combined + base + H, mask=load_mask, other=0.0).to(tl.float32)
        gate_l = tl.load(Combined + base + 2 * H, mask=load_mask, other=0.0).to(tl.float32)
        gate_r = tl.load(Combined + base + 3 * H, mask=load_mask, other=0.0).to(tl.float32)
        gate_offsets = rows_i64[:, None] * H + hids_i64[None, :]
        grad_out_gate = tl.load(GradOutGate + gate_offsets, mask=valid, other=0.0)
        sig_l = tl.sigmoid(gate_l)
        sig_r = tl.sigmoid(gate_r)
        tl.store(GradCombined + base, grad_l * sig_l, mask=valid)
        tl.store(GradCombined + base + H, grad_r * sig_r, mask=valid)
        tl.store(GradCombined + base + 2 * H, grad_l * proj_l * sig_l * (1.0 - sig_l), mask=valid)
        tl.store(GradCombined + base + 3 * H, grad_r * proj_r * sig_r * (1.0 - sig_r), mask=valid)
        tl.store(GradCombined + base + 4 * H, grad_out_gate, mask=valid)


@triton.jit
def _pack_combined_gated_lr_outgate_bwd_contiguous_kernel(
    GradLeft,
    GradRight,
    GradOutGate,
    Combined,
    PairMask,
    GradCombined,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    NEED_GRAD_COMBINED: tl.constexpr,
    ASSUME_ALL_ACTIVE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Backpropagate contiguous packed operands and output-gate gradients."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hids = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    hid_mask = hids < H

    nn: tl.constexpr = N * N
    b = rows // nn
    rem = rows - b * nn
    i = rem // N
    j = rem - i * N
    rows_i64 = rows.to(tl.int64)
    hids_i64 = hids.to(tl.int64)
    b_i64 = b.to(tl.int64)
    i_i64 = i.to(tl.int64)
    j_i64 = j.to(tl.int64)

    valid = row_mask[:, None] & hid_mask[None, :]
    if ASSUME_ALL_ACTIVE:
        active = row_mask
    else:
        active = tl.load(PairMask + rows, mask=row_mask, other=0) != 0
    load_mask = valid & active[:, None]
    packed_offsets = ((b_i64[:, None] * H + hids_i64[None, :]) * N + i_i64[:, None]) * N + j_i64[:, None]
    grad_l = tl.load(GradLeft + packed_offsets, mask=load_mask, other=0.0).to(tl.float32)
    grad_r = tl.load(GradRight + packed_offsets, mask=load_mask, other=0.0).to(tl.float32)

    if NEED_GRAD_COMBINED:
        cols: tl.constexpr = 5 * H
        base = rows_i64[:, None] * cols + hids_i64[None, :]
        proj_l = tl.load(Combined + base, mask=load_mask, other=0.0).to(tl.float32)
        proj_r = tl.load(Combined + base + H, mask=load_mask, other=0.0).to(tl.float32)
        gate_l = tl.load(Combined + base + 2 * H, mask=load_mask, other=0.0).to(tl.float32)
        gate_r = tl.load(Combined + base + 3 * H, mask=load_mask, other=0.0).to(tl.float32)
        gate_offsets = rows_i64[:, None] * H + hids_i64[None, :]
        grad_out_gate = tl.load(GradOutGate + gate_offsets, mask=valid, other=0.0)
        sig_l = tl.sigmoid(gate_l)
        sig_r = tl.sigmoid(gate_r)
        tl.store(GradCombined + base, grad_l * sig_l, mask=valid)
        tl.store(GradCombined + base + H, grad_r * sig_r, mask=valid)
        tl.store(GradCombined + base + 2 * H, grad_l * proj_l * sig_l * (1.0 - sig_l), mask=valid)
        tl.store(GradCombined + base + 3 * H, grad_r * proj_r * sig_r * (1.0 - sig_r), mask=valid)
        tl.store(GradCombined + base + 4 * H, grad_out_gate, mask=valid)


def _block_h(hidden: int) -> int:
    """Choose a power-of-two hidden tile for the pack kernels."""
    return min(128, 1 << (int(hidden) - 1).bit_length())


def _block_m(hidden: int) -> int:
    """Choose the row tile for pack kernels."""
    return 32


def _use_strided_grad_loads(n: int) -> bool:
    """Use strided pack-backward gradient loads only where they benchmark faster."""
    return int(n) >= 512


def _view_out_gate_enabled(n: int, hidden: int, batch: int) -> bool:
    """Return whether the output gate should alias the 5H combined projection."""
    # The N256 padded-training frontier crops PairFormer to roughly N160.  At
    # that shape, keeping the output gate as a strided view avoids a large
    # write/read of a separate contiguous gate tensor; the downstream gate kernel
    # already has a strided-gate path.  Smaller batch screens and dense N256 were
    # not the measured win, so keep the old materialized path there.
    if 128 <= int(n) <= 192 and int(hidden) == 64 and int(batch) >= 64:
        return True
    if int(n) < 512:
        return False
    return int(hidden) >= 128 or int(batch) > 1


class _TriangleMulPackCombinedGatedTriton(torch.autograd.Function):
    """Autograd bridge for a combined projection/gate tensor."""

    @staticmethod
    def forward(ctx, combined: torch.Tensor, pair_mask: torch.Tensor):
        """Pack combined ``[left, right, gate_l, gate_r]`` projections."""
        if combined.ndim != 4:
            raise ValueError("packed combined triangle input gate expects [B, N, N, 4H]")
        if pair_mask.shape != combined.shape[:3]:
            raise ValueError("packed combined triangle input gate expects pair_mask shape [B, N, N]")
        if combined.shape[1] != combined.shape[2]:
            raise ValueError("packed combined triangle input gate expects square pair dimensions")
        if combined.shape[-1] % 4 != 0:
            raise ValueError("packed combined triangle input gate expects projection width divisible by 4")
        if not combined.is_cuda or not pair_mask.is_cuda:
            raise RuntimeError("packed combined triangle input gate requires CUDA tensors")

        combined = combined.contiguous()
        pair_mask = pair_mask.to(device=combined.device, dtype=torch.bool).contiguous()
        batch, n, _, four_h = (int(v) for v in combined.shape)
        hidden = four_h // 4
        left = torch.empty((batch, hidden, n, n), device=combined.device, dtype=combined.dtype)
        right = torch.empty_like(left)
        rows = batch * n * n
        block_m = _block_m(hidden)
        block_h = _block_h(hidden)
        grid = (triton.cdiv(rows, block_m), triton.cdiv(hidden, block_h))
        _pack_combined_gated_lr_fwd_kernel[grid](
            combined.reshape(rows, four_h),
            pair_mask.view(rows),
            left,
            right,
            rows,
            n,
            hidden,
            BLOCK_M=block_m,
            BLOCK_H=block_h,
            num_warps=4 if block_h >= 64 else 1,
        )
        ctx.save_for_backward(combined, pair_mask)
        ctx.shape = (batch, n, hidden)
        ctx.block_m = block_m
        ctx.block_h = block_h
        return left, right

    @staticmethod
    def backward(ctx, grad_left: torch.Tensor, grad_right: torch.Tensor):
        """Unpack left/right gradients into the combined projection tensor."""
        combined, pair_mask = ctx.saved_tensors
        batch, n, hidden = ctx.shape
        rows = batch * n * n
        four_h = hidden * 4
        needs = ctx.needs_input_grad
        grad_combined = None
        if needs[0]:
            grad_combined = torch.empty_like(combined)
        if needs[0]:
            use_strided_grad = _use_strided_grad_loads(n)
            if use_strided_grad:
                grad_left_arg = grad_left
                grad_right_arg = grad_right
            else:
                grad_left_arg = grad_left.contiguous()
                grad_right_arg = grad_right.contiguous()
            grad_left_strides = tuple(int(s) for s in grad_left_arg.stride())
            grad_right_strides = tuple(int(s) for s in grad_right_arg.stride())
            grid = (triton.cdiv(rows, ctx.block_m), triton.cdiv(hidden, ctx.block_h))
            if use_strided_grad:
                _pack_combined_gated_lr_bwd_kernel[grid](
                    grad_left_arg,
                    grad_right_arg,
                    combined.reshape(rows, four_h),
                    pair_mask.view(rows),
                    grad_combined.reshape(rows, four_h),
                    grad_left_strides[0],
                    grad_left_strides[1],
                    grad_left_strides[2],
                    grad_left_strides[3],
                    grad_right_strides[0],
                    grad_right_strides[1],
                    grad_right_strides[2],
                    grad_right_strides[3],
                    rows,
                    n,
                    hidden,
                    True,
                    BLOCK_M=ctx.block_m,
                    BLOCK_H=ctx.block_h,
                    num_warps=4 if ctx.block_h >= 64 else 1,
                )
            else:
                _pack_combined_gated_lr_bwd_contiguous_kernel[grid](
                    grad_left_arg,
                    grad_right_arg,
                    combined.reshape(rows, four_h),
                    pair_mask.view(rows),
                    grad_combined.reshape(rows, four_h),
                    rows,
                    n,
                    hidden,
                    True,
                    BLOCK_M=ctx.block_m,
                    BLOCK_H=ctx.block_h,
                    num_warps=4 if ctx.block_h >= 64 else 1,
                )
        return grad_combined, None


def triangle_mul_packed_combined_gated_operands(
    combined: torch.Tensor,
    pair_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed operands from one combined projection/gate tensor."""
    return _TriangleMulPackCombinedGatedTriton.apply(combined, pair_mask)


class _TriangleMulPackCombinedGatedOutGateTriton(torch.autograd.Function):
    """Autograd bridge for combined input operands plus output-gate logits."""

    @staticmethod
    def forward(ctx, combined: torch.Tensor, pair_mask: torch.Tensor, assume_all_active: bool):
        """Pack combined ``[left, right, gate_l, gate_r, out_gate]`` projections."""
        if combined.ndim != 4:
            raise ValueError("packed combined triangle path expects [B, N, N, 5H]")
        if pair_mask.shape != combined.shape[:3]:
            raise ValueError("packed combined triangle path expects pair_mask shape [B, N, N]")
        if combined.shape[1] != combined.shape[2]:
            raise ValueError("packed combined triangle path expects square pair dimensions")
        if combined.shape[-1] % 5 != 0:
            raise ValueError("packed combined triangle path expects projection width divisible by 5")
        if not combined.is_cuda or not pair_mask.is_cuda:
            raise RuntimeError("packed combined triangle path requires CUDA tensors")

        combined = combined.contiguous()
        pair_mask = pair_mask.to(device=combined.device, dtype=torch.bool).contiguous()
        assume_all_active = bool(assume_all_active)
        batch, n, _, five_h = (int(v) for v in combined.shape)
        hidden = five_h // 5
        left = torch.empty((batch, hidden, n, n), device=combined.device, dtype=combined.dtype)
        right = torch.empty_like(left)
        rows = batch * n * n
        block_m = _block_m(hidden)
        block_h = _block_h(hidden)
        grid = (triton.cdiv(rows, block_m), triton.cdiv(hidden, block_h))
        if _view_out_gate_enabled(n, hidden, batch):
            _pack_combined_gated_lr_view_outgate_fwd_kernel[grid](
                combined.reshape(rows, five_h),
                pair_mask.view(rows),
                left,
                right,
                rows,
                n,
                hidden,
                assume_all_active,
                BLOCK_M=block_m,
                BLOCK_H=block_h,
                num_warps=4 if block_h >= 64 else 1,
            )
            out_gate = combined[..., 4 * hidden : 5 * hidden]
        else:
            out_gate = torch.empty((batch, n, n, hidden), device=combined.device, dtype=combined.dtype)
            _pack_combined_gated_lr_outgate_fwd_kernel[grid](
                combined.reshape(rows, five_h),
                pair_mask.view(rows),
                left,
                right,
                out_gate.reshape(rows, hidden),
                rows,
                n,
                hidden,
                assume_all_active,
                BLOCK_M=block_m,
                BLOCK_H=block_h,
                num_warps=4 if block_h >= 64 else 1,
            )
        ctx.save_for_backward(combined, pair_mask)
        ctx.shape = (batch, n, hidden)
        ctx.block_m = block_m
        ctx.block_h = block_h
        ctx.assume_all_active = assume_all_active
        return left, right, out_gate

    @staticmethod
    def backward(ctx, grad_left: torch.Tensor, grad_right: torch.Tensor, grad_out_gate: torch.Tensor):
        """Unpack left/right/output-gate gradients into the combined projection tensor."""
        combined, pair_mask = ctx.saved_tensors
        batch, n, hidden = ctx.shape
        rows = batch * n * n
        five_h = hidden * 5
        needs = ctx.needs_input_grad
        grad_combined = None
        if needs[0]:
            grad_combined = torch.empty_like(combined)
        if needs[0]:
            use_strided_grad = _use_strided_grad_loads(n)
            if use_strided_grad:
                grad_left_arg = grad_left
                grad_right_arg = grad_right
            else:
                grad_left_arg = grad_left.contiguous()
                grad_right_arg = grad_right.contiguous()
            grad_left_strides = tuple(int(s) for s in grad_left_arg.stride())
            grad_right_strides = tuple(int(s) for s in grad_right_arg.stride())
            grid = (triton.cdiv(rows, ctx.block_m), triton.cdiv(hidden, ctx.block_h))
            grad_out_gate_arg = grad_out_gate.contiguous()
            if use_strided_grad:
                _pack_combined_gated_lr_outgate_bwd_kernel[grid](
                    grad_left_arg,
                    grad_right_arg,
                    grad_out_gate_arg,
                    combined.reshape(rows, five_h),
                    pair_mask.view(rows),
                    grad_combined.reshape(rows, five_h),
                    grad_left_strides[0],
                    grad_left_strides[1],
                    grad_left_strides[2],
                    grad_left_strides[3],
                    grad_right_strides[0],
                    grad_right_strides[1],
                    grad_right_strides[2],
                    grad_right_strides[3],
                    rows,
                    n,
                    hidden,
                    True,
                    ctx.assume_all_active,
                    BLOCK_M=ctx.block_m,
                    BLOCK_H=ctx.block_h,
                    num_warps=4 if ctx.block_h >= 64 else 1,
                )
            else:
                _pack_combined_gated_lr_outgate_bwd_contiguous_kernel[grid](
                    grad_left_arg,
                    grad_right_arg,
                    grad_out_gate_arg,
                    combined.reshape(rows, five_h),
                    pair_mask.view(rows),
                    grad_combined.reshape(rows, five_h),
                    rows,
                    n,
                    hidden,
                    True,
                    ctx.assume_all_active,
                    BLOCK_M=ctx.block_m,
                    BLOCK_H=ctx.block_h,
                    num_warps=4 if ctx.block_h >= 64 else 1,
                )
        return grad_combined, None, None


def triangle_mul_packed_combined_gated_outgate(
    combined: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    assume_all_active: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed operands and output-gate logits from one combined projection.

    ``assume_all_active`` is only valid when caller-owned metadata proves every
    pair row is active.  It removes one mask load and branch from the hot packed
    pointwise kernels used by exact-prefix compact groups.
    """
    return _TriangleMulPackCombinedGatedOutGateTriton.apply(combined, pair_mask, bool(assume_all_active))
