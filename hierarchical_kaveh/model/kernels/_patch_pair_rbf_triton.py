"""Fused p=4 patch-slot RBF projection.

The eager hierarchical pair update builds a ``[B, M, M, 4, 4, R]`` RBF
tensor before applying a ``16 * R -> C`` projection. This operator performs the
same sixteen ordered slot-pair projections in registers and writes only the
``[B, M, M, C]`` result. Its production backward recomputes pair tiles for the
projection-weight gradient; differentiable-coordinate callers retain the exact
manual coordinate-gradient path.
"""

from __future__ import annotations

import os
from contextlib import nullcontext as profile_range

import torch
import torch.nn.functional as F



def _load_triton():
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Triton patch-pair RBF projection requires triton") from exc
    return triton, tl


triton, tl = _load_triton()


@triton.jit
def _patch_pair_rbf_projection_fwd_kernel(
    CA,
    SlotMask,
    Weight,
    Out,
    TOTAL_PAIRS: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    FEATURES: tl.constexpr,
    MIN_DIST: tl.constexpr,
    CENTER_STEP: tl.constexpr,
    INV_WIDTH: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    rbfs = tl.arange(0, BLOCK_R)

    row_mask = rows < TOTAL_PAIRS
    pair_index = rows % (M * M)
    batch = rows // (M * M)
    patch_i = pair_index // M
    patch_j = pair_index - patch_i * M
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)

    for slot_pair in range(16):
        slot_i: tl.constexpr = slot_pair // 4
        slot_j: tl.constexpr = slot_pair % 4
        valid_i = tl.load(
            SlotMask + (batch * M + patch_i) * 4 + slot_i,
            mask=row_mask,
            other=0,
        )
        valid_j = tl.load(
            SlotMask + (batch * M + patch_j) * 4 + slot_j,
            mask=row_mask,
            other=0,
        )
        pair_valid = row_mask & valid_i & valid_j

        ca_i = ((batch * M + patch_i) * 4 + slot_i) * 3
        ca_j = ((batch * M + patch_j) * 4 + slot_j) * 3
        dx = tl.load(CA + ca_i + 0, mask=row_mask, other=0.0).to(tl.float32) - tl.load(
            CA + ca_j + 0, mask=row_mask, other=0.0
        ).to(tl.float32)
        dy = tl.load(CA + ca_i + 1, mask=row_mask, other=0.0).to(tl.float32) - tl.load(
            CA + ca_j + 1, mask=row_mask, other=0.0
        ).to(tl.float32)
        dz = tl.load(CA + ca_i + 2, mask=row_mask, other=0.0).to(tl.float32) - tl.load(
            CA + ca_j + 2, mask=row_mask, other=0.0
        ).to(tl.float32)
        distance = tl.sqrt(dx * dx + dy * dy + dz * dz)
        centers = MIN_DIST + rbfs * CENTER_STEP
        z = (distance[:, None] - centers[None, :]) * INV_WIDTH
        rbf = tl.exp(-0.5 * z * z)
        rbf = tl.where(pair_valid[:, None] & (rbfs[None, :] < R), rbf, 0.0)

        features = slot_pair * R + rbfs
        weight = tl.load(
            Weight + channels[None, :] * FEATURES + features[:, None],
            mask=(channels[None, :] < C) & (rbfs[:, None] < R),
            other=0.0,
        ).to(tl.float32)
        if USE_BF16_DOT:
            acc += tl.dot(
                rbf.to(tl.bfloat16),
                weight.to(tl.bfloat16),
                out_dtype=tl.float32,
            )
        else:
            acc += tl.dot(rbf, weight, out_dtype=tl.float32, input_precision="ieee")

    tl.store(
        Out + rows[:, None] * C + channels[None, :],
        acc,
        mask=row_mask[:, None] & (channels[None, :] < C),
    )


@triton.jit
def _patch_pair_rbf_projection_bwd_weight_kernel(
    GradOut,
    CA,
    SlotMask,
    GradWeightPartial,
    TOTAL_PAIRS: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    MIN_DIST: tl.constexpr,
    CENTER_STEP: tl.constexpr,
    INV_WIDTH: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Recompute one ordered slot-pair RBF tile for projection-weight gradients."""

    tile = tl.program_id(0)
    slot_pair = tl.program_id(1)
    rows = tile * BLOCK_P + tl.arange(0, BLOCK_P)
    channels = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    rbfs = tl.arange(0, BLOCK_R)
    row_mask = rows < TOTAL_PAIRS
    pair_index = rows % (M * M)
    batch = rows // (M * M)
    patch_i = pair_index // M
    patch_j = pair_index - patch_i * M
    slot_i = slot_pair // 4
    slot_j = slot_pair - slot_i * 4

    valid_i = tl.load(
        SlotMask + (batch * M + patch_i) * 4 + slot_i,
        mask=row_mask,
        other=0,
    )
    valid_j = tl.load(
        SlotMask + (batch * M + patch_j) * 4 + slot_j,
        mask=row_mask,
        other=0,
    )
    pair_valid = row_mask & valid_i & valid_j
    ca_i = ((batch * M + patch_i) * 4 + slot_i) * 3
    ca_j = ((batch * M + patch_j) * 4 + slot_j) * 3
    dx = tl.load(CA + ca_i, mask=row_mask, other=0.0).to(tl.float32) - tl.load(
        CA + ca_j, mask=row_mask, other=0.0
    ).to(tl.float32)
    dy = tl.load(CA + ca_i + 1, mask=row_mask, other=0.0).to(tl.float32) - tl.load(
        CA + ca_j + 1, mask=row_mask, other=0.0
    ).to(tl.float32)
    dz = tl.load(CA + ca_i + 2, mask=row_mask, other=0.0).to(tl.float32) - tl.load(
        CA + ca_j + 2, mask=row_mask, other=0.0
    ).to(tl.float32)
    distance = tl.sqrt(dx * dx + dy * dy + dz * dz)
    centers = MIN_DIST + rbfs * CENTER_STEP
    z = (distance[:, None] - centers[None, :]) * INV_WIDTH
    rbf = tl.exp(-0.5 * z * z)
    rbf = tl.where(pair_valid[:, None] & (rbfs[None, :] < R), rbf, 0.0)
    grad_out = tl.load(
        GradOut + rows[:, None] * C + channels[None, :],
        mask=row_mask[:, None] & (channels[None, :] < C),
        other=0.0,
    ).to(tl.float32)
    if USE_BF16_DOT:
        partial = tl.dot(
            tl.trans(grad_out.to(tl.bfloat16)),
            rbf.to(tl.bfloat16),
            out_dtype=tl.float32,
        )
    else:
        partial = tl.dot(
            tl.trans(grad_out),
            rbf,
            out_dtype=tl.float32,
            input_precision="ieee",
        )
    offsets = (
        ((tile * 16 + slot_pair) * C + channels[:, None]) * R
        + rbfs[None, :]
    )
    tl.store(
        GradWeightPartial + offsets,
        partial,
        mask=(channels[:, None] < C) & (rbfs[None, :] < R),
    )


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _forward_block_pairs() -> int:
    return int(os.environ.get("KAVEH_PATCH_PAIR_RBF_BLOCK_PAIRS", "16"))


def _backward_weight_block_pairs() -> int:
    return int(os.environ.get("KAVEH_PATCH_PAIR_RBF_BWD_WEIGHT_BLOCK_PAIRS", "256"))


def _native_projection(
    ca_slots: torch.Tensor,
    slot_mask: torch.Tensor,
    weight: torch.Tensor,
    min_dist: float,
    max_dist: float,
) -> torch.Tensor:
    rbf_dim = int(weight.shape[1]) // 16
    delta = ca_slots[:, :, None, :, None, :] - ca_slots[:, None, :, None, :, :]
    distances = torch.linalg.vector_norm(delta.float(), dim=-1)
    centers = torch.linspace(
        float(min_dist),
        float(max_dist),
        rbf_dim,
        device=distances.device,
        dtype=distances.dtype,
    )
    width = max((float(max_dist) - float(min_dist)) / max(rbf_dim - 1, 1), 1e-6)
    rbf = torch.exp(-0.5 * ((distances[..., None] - centers) / width) ** 2)
    pair_mask = slot_mask[:, :, None, :, None] & slot_mask[:, None, :, None, :]
    rbf = rbf * pair_mask[..., None].to(rbf.dtype)
    return F.linear(rbf.flatten(start_dim=3).to(weight.dtype), weight, None)


def _manual_backward(
    grad_out: torch.Tensor,
    ca_slots: torch.Tensor,
    slot_mask: torch.Tensor,
    weight: torch.Tensor,
    min_dist: float,
    max_dist: float,
    *,
    need_ca: bool,
    need_weight: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Differentiate the ordered slot RBF projection without an autograd graph."""

    batch, patches = (int(v) for v in ca_slots.shape[:2])
    channels, features = (int(v) for v in weight.shape)
    rbf_dim = features // 16
    with profile_range("patch_pair_rbf_projection.backward.manual.rbf"):
        delta = ca_slots[:, :, None, :, None, :] - ca_slots[:, None, :, None, :, :]
        delta_float = delta.float()
        distances = torch.linalg.vector_norm(delta_float, dim=-1)
        centers = torch.linspace(
            float(min_dist),
            float(max_dist),
            rbf_dim,
            device=ca_slots.device,
            dtype=torch.float32,
        )
        width = max((float(max_dist) - float(min_dist)) / max(rbf_dim - 1, 1), 1e-6)
        centered = distances[..., None] - centers
        rbf = torch.exp(-0.5 * (centered / width) ** 2)
        pair_mask = slot_mask[:, :, None, :, None] & slot_mask[:, None, :, None, :]
        rbf = rbf * pair_mask[..., None].to(rbf.dtype)

    grad_out_2d = grad_out.reshape(batch * patches * patches, channels)
    rbf_2d = rbf.reshape(batch * patches * patches, features)
    dot_dtype = grad_out.dtype if grad_out.dtype in {torch.bfloat16, torch.float16} else torch.float32
    grad_ca = None
    if need_ca:
        with profile_range("patch_pair_rbf_projection.backward.manual.grad_rbf"):
            grad_rbf = torch.mm(
                grad_out_2d.to(dot_dtype),
                weight.to(dot_dtype),
                out_dtype=torch.float32,
            ).reshape(batch, patches, patches, 4, 4, rbf_dim)
        with profile_range("patch_pair_rbf_projection.backward.manual.grad_ca"):
            grad_distance = (grad_rbf * rbf * (-centered / (width * width))).sum(dim=-1)
            inv_distance = torch.where(distances > 0, distances.reciprocal(), 0.0)
            grad_delta = grad_distance[..., None] * inv_distance[..., None] * delta_float
            grad_ca = (
                grad_delta.sum(dim=(2, 4)) - grad_delta.sum(dim=(1, 3))
            ).to(ca_slots.dtype)

    grad_weight = None
    if need_weight:
        with profile_range("patch_pair_rbf_projection.backward.manual.grad_weight"):
            grad_weight = torch.mm(
                grad_out_2d.to(dot_dtype).mT,
                rbf_2d.to(dot_dtype),
                out_dtype=torch.float32,
            ).to(weight.dtype)
    return grad_ca, grad_weight


def _triton_weight_backward(
    grad_out: torch.Tensor,
    ca_slots: torch.Tensor,
    slot_mask: torch.Tensor,
    weight: torch.Tensor,
    min_dist: float,
    max_dist: float,
) -> torch.Tensor:
    """Recompute RBF tiles and reduce only the production projection-weight gradient."""

    grad_out = grad_out.contiguous()
    batch, patches = (int(v) for v in ca_slots.shape[:2])
    channels, features = (int(v) for v in weight.shape)
    rbf_dim = features // 16
    block_p_raw = _backward_weight_block_pairs()
    if block_p_raw <= 0:
        raise ValueError("patch-pair RBF backward pair block must be positive")
    block_p = _next_power_of_2(block_p_raw)
    block_c = min(64, _next_power_of_2(channels))
    block_r = max(16, _next_power_of_2(rbf_dim))
    total_pairs = batch * patches * patches
    tiles = triton.cdiv(total_pairs, block_p)
    partials = torch.empty(
        (tiles, 16, channels, rbf_dim),
        device=weight.device,
        dtype=torch.float32,
    )
    width = max((max_dist - min_dist) / max(rbf_dim - 1, 1), 1e-6)
    with profile_range("patch_pair_rbf_projection.backward.triton_weight.kernel"):
        _patch_pair_rbf_projection_bwd_weight_kernel[
            (tiles, 16, triton.cdiv(channels, block_c))
        ](
            grad_out,
            ca_slots,
            slot_mask,
            partials,
            total_pairs,
            patches,
            channels,
            rbf_dim,
            min_dist,
            0.0 if rbf_dim <= 1 else (max_dist - min_dist) / (rbf_dim - 1),
            1.0 / width,
            grad_out.dtype == torch.bfloat16,
            BLOCK_P=block_p,
            BLOCK_C=block_c,
            BLOCK_R=block_r,
            num_warps=4,
        )
    with profile_range("patch_pair_rbf_projection.backward.triton_weight.reduce"):
        return (
            partials.sum(dim=0)
            .permute(1, 0, 2)
            .reshape(channels, features)
            .to(weight.dtype)
        )


class _PatchPairRBFProjectionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ca_slots: torch.Tensor,
        slot_mask: torch.Tensor,
        weight: torch.Tensor,
        min_dist: float,
        max_dist: float,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if ca_slots.ndim != 4 or ca_slots.shape[-2:] != (4, 3):
            raise ValueError("patch-pair RBF projection expects CA slots [B, M, 4, 3]")
        if slot_mask.shape != ca_slots.shape[:3]:
            raise ValueError("patch-pair RBF projection slot mask shape mismatch")
        if weight.ndim != 2 or int(weight.shape[1]) % 16 != 0:
            raise ValueError("patch-pair RBF projection weight must be [C, 16 * R]")
        if not ca_slots.is_cuda or not slot_mask.is_cuda or not weight.is_cuda:
            raise RuntimeError("patch-pair RBF projection requires CUDA tensors")
        if not ca_slots.is_contiguous() or not slot_mask.is_contiguous() or not weight.is_contiguous():
            raise ValueError("patch-pair RBF projection expects contiguous tensors")

        batch, patches = (int(v) for v in ca_slots.shape[:2])
        channels, features = (int(v) for v in weight.shape)
        rbf_dim = features // 16
        if rbf_dim <= 0 or rbf_dim > 64 or channels <= 0 or channels > 256:
            raise ValueError("patch-pair RBF projection supports R<=64 and C<=256")
        width = max((float(max_dist) - float(min_dist)) / max(rbf_dim - 1, 1), 1e-6)
        block_p_raw = _forward_block_pairs()
        if block_p_raw <= 0:
            raise ValueError("patch-pair RBF block size must be positive")
        block_p = _next_power_of_2(block_p_raw)
        block_c = min(64, _next_power_of_2(channels))
        block_r = max(16, _next_power_of_2(rbf_dim))
        total_pairs = batch * patches * patches
        out = torch.empty(
            (batch, patches, patches, channels),
            device=ca_slots.device,
            dtype=out_dtype,
        )
        _patch_pair_rbf_projection_fwd_kernel[
            (triton.cdiv(total_pairs, block_p), triton.cdiv(channels, block_c))
        ](
            ca_slots,
            slot_mask,
            weight,
            out,
            total_pairs,
            patches,
            channels,
            rbf_dim,
            features,
            float(min_dist),
            0.0 if rbf_dim <= 1 else (float(max_dist) - float(min_dist)) / (rbf_dim - 1),
            1.0 / width,
            out_dtype == torch.bfloat16,
            BLOCK_P=block_p,
            BLOCK_C=block_c,
            BLOCK_R=block_r,
            num_warps=4,
        )
        ctx.save_for_backward(ca_slots, slot_mask, weight)
        ctx.min_dist = float(min_dist)
        ctx.max_dist = float(max_dist)
        ctx.out_dtype = out_dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        ca_slots, slot_mask, weight = ctx.saved_tensors
        need_ca, _, need_weight = ctx.needs_input_grad[:3]
        if not need_ca and not need_weight:
            return None, None, None, None, None, None

        backward_impl = os.environ.get("KAVEH_PATCH_PAIR_RBF_BACKWARD", "auto").lower()
        if backward_impl not in {"auto", "manual", "eager", "triton_weight"}:
            raise ValueError(
                "KAVEH_PATCH_PAIR_RBF_BACKWARD must be auto, manual, eager, or triton_weight"
            )
        if backward_impl == "auto":
            backward_impl = "manual" if need_ca else "triton_weight"
        if backward_impl == "triton_weight":
            if need_ca:
                raise RuntimeError("triton_weight patch-pair RBF backward does not support CA gradients")
            if not need_weight:
                return None, None, None, None, None, None
            grad_weight = _triton_weight_backward(
                grad_out,
                ca_slots,
                slot_mask,
                weight,
                ctx.min_dist,
                ctx.max_dist,
            )
            return None, None, grad_weight, None, None, None
        if backward_impl == "manual":
            grad_ca, grad_weight = _manual_backward(
                grad_out,
                ca_slots,
                slot_mask,
                weight,
                ctx.min_dist,
                ctx.max_dist,
                need_ca=need_ca,
                need_weight=need_weight,
            )
            return grad_ca, None, grad_weight, None, None, None

        with profile_range("patch_pair_rbf_projection.backward.eager_recompute"):
            with torch.enable_grad():
                ca_leaf = ca_slots.detach().requires_grad_(need_ca)
                weight_leaf = weight.detach().requires_grad_(need_weight)
                autocast_enabled = ctx.out_dtype in {torch.bfloat16, torch.float16}
                with torch.autocast("cuda", dtype=ctx.out_dtype, enabled=autocast_enabled):
                    out = _native_projection(
                        ca_leaf,
                        slot_mask,
                        weight_leaf,
                        ctx.min_dist,
                        ctx.max_dist,
                    )
                inputs = []
                if need_ca:
                    inputs.append(ca_leaf)
                if need_weight:
                    inputs.append(weight_leaf)
                grads = torch.autograd.grad(out, inputs, grad_out, create_graph=False)
        grad_index = 0
        grad_ca = None
        grad_weight = None
        if need_ca:
            grad_ca = grads[grad_index]
            grad_index += 1
        if need_weight:
            grad_weight = grads[grad_index]
        return grad_ca, None, grad_weight, None, None, None


def _autocast_output_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.device.type != "cuda":
        return tensor.dtype
    try:
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
    except TypeError:
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return tensor.dtype


def patch_pair_rbf_projection_triton(
    ca_slots: torch.Tensor,
    slot_mask: torch.Tensor,
    weight: torch.Tensor,
    min_dist: float,
    max_dist: float,
) -> torch.Tensor:
    """Project all sixteen ordered p=4 slot-pair RBFs without materializing them."""

    return _PatchPairRBFProjectionTriton.apply(
        ca_slots.contiguous(),
        slot_mask.to(torch.bool).contiguous(),
        weight.contiguous(),
        float(min_dist),
        float(max_dist),
        _autocast_output_dtype(ca_slots),
    )


__all__ = ["patch_pair_rbf_projection_triton"]
