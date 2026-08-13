"""Fused p=4 patch-to-residue distogram expansion and pull-reduction backward."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU runtime
    triton = None
    tl = None


Implementation = Literal["auto", "eager", "triton"]
_PATCH_SIZE = 4
_BINS = 64


def _validate(
    coarse_logits: Tensor,
    slot_bias: Tensor,
    residue_to_patch: Tensor,
    residue_slot: Tensor,
    patch_residue_idx: Tensor,
    residue_mask: Tensor,
) -> tuple[int, int, int, int]:
    if coarse_logits.ndim != 4 or coarse_logits.shape[1] != coarse_logits.shape[2]:
        raise ValueError("coarse_logits must have shape [B,M,M,C]")
    batch, patches, _patches, bins = (int(value) for value in coarse_logits.shape)
    if slot_bias.shape != (_PATCH_SIZE, _PATCH_SIZE, bins):
        raise ValueError(f"slot_bias must have shape {(4, 4, bins)}")
    if residue_to_patch.ndim != 2 or residue_to_patch.shape[0] != batch:
        raise ValueError("residue_to_patch must have shape [B,N]")
    length = int(residue_to_patch.shape[1])
    if residue_slot.shape != (batch, length) or residue_mask.shape != (batch, length):
        raise ValueError("residue_slot and residue_mask must have shape [B,N]")
    if patch_residue_idx.shape != (batch, patches, _PATCH_SIZE):
        raise ValueError("patch_residue_idx must have shape [B,M,4]")
    return batch, patches, length, bins


def _expand_eager(
    coarse_logits: Tensor,
    slot_bias: Tensor,
    residue_to_patch: Tensor,
    residue_slot: Tensor,
    residue_mask: Tensor,
) -> Tensor:
    batch = torch.arange(coarse_logits.shape[0], device=coarse_logits.device)[:, None, None]
    patch_idx = residue_to_patch.clamp_min(0)
    slot_idx = residue_slot.clamp_min(0)
    output = coarse_logits[batch, patch_idx[:, :, None], patch_idx[:, None, :]]
    output = output + slot_bias[slot_idx[:, :, None], slot_idx[:, None, :]].to(output.dtype)
    pair_mask = residue_mask.to(torch.bool)[:, :, None] & residue_mask.to(torch.bool)[:, None, :]
    return output * pair_mask[..., None].to(output.dtype)


if triton is not None:

    @triton.jit
    def _patch_distogram_forward_kernel(
        Coarse,
        SlotBias,
        ResidueToPatch,
        ResidueSlot,
        ResidueMask,
        Out,
        CS0: tl.constexpr,
        CS1: tl.constexpr,
        CS2: tl.constexpr,
        CS3: tl.constexpr,
        SS0: tl.constexpr,
        SS1: tl.constexpr,
        SS2: tl.constexpr,
        OS0: tl.constexpr,
        OS1: tl.constexpr,
        OS2: tl.constexpr,
        OS3: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        C: tl.constexpr,
        PATCH_SIZE: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        j_block = tl.program_id(1)
        batch = row // N
        i = row % N
        js = j_block * BLOCK_J + tl.arange(0, BLOCK_J)
        channels = tl.arange(0, BLOCK_C)
        valid_j = js < N
        row_valid = tl.load(ResidueMask + batch * N + i) != 0
        col_valid = tl.load(ResidueMask + batch * N + js, mask=valid_j, other=0) != 0
        active = row_valid & valid_j & col_valid
        patch_i = tl.load(ResidueToPatch + batch * N + i).to(tl.int32)
        slot_i = tl.load(ResidueSlot + batch * N + i).to(tl.int32)
        patch_j = tl.load(ResidueToPatch + batch * N + js, mask=valid_j, other=0).to(tl.int32)
        slot_j = tl.load(ResidueSlot + batch * N + js, mask=valid_j, other=0).to(tl.int32)
        patch_i = tl.maximum(0, tl.minimum(patch_i, M - 1))
        patch_j = tl.maximum(0, tl.minimum(patch_j, M - 1))
        slot_i = tl.maximum(0, tl.minimum(slot_i, PATCH_SIZE - 1))
        slot_j = tl.maximum(0, tl.minimum(slot_j, PATCH_SIZE - 1))
        coarse_offsets = (
            batch * CS0
            + patch_i * CS1
            + patch_j[:, None] * CS2
            + channels[None, :] * CS3
        )
        slot_offsets = slot_i * SS0 + slot_j[:, None] * SS1 + channels[None, :] * SS2
        values = tl.load(
            Coarse + coarse_offsets,
            mask=active[:, None] & (channels[None, :] < C),
            other=0.0,
        )
        bias = tl.load(
            SlotBias + slot_offsets,
            mask=active[:, None] & (channels[None, :] < C),
            other=0.0,
        )
        out_offsets = (
            batch * OS0 + i * OS1 + js[:, None] * OS2 + channels[None, :] * OS3
        )
        tl.store(
            Out + out_offsets,
            values + bias,
            mask=valid_j[:, None] & (channels[None, :] < C),
        )


    @triton.jit
    def _patch_distogram_dcoarse_kernel(
        GradOut,
        PatchResidueIdx,
        GradCoarse,
        GS0: tl.constexpr,
        GS1: tl.constexpr,
        GS2: tl.constexpr,
        GS3: tl.constexpr,
        PRS0: tl.constexpr,
        PRS1: tl.constexpr,
        PRS2: tl.constexpr,
        DCS0: tl.constexpr,
        DCS1: tl.constexpr,
        DCS2: tl.constexpr,
        DCS3: tl.constexpr,
        M: tl.constexpr,
        C: tl.constexpr,
        PATCH_SIZE: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        patch_pair = tl.program_id(0)
        q = patch_pair % M
        batch_p = patch_pair // M
        p = batch_p % M
        batch = batch_p // M
        slot_pairs = tl.arange(0, PATCH_SIZE * PATCH_SIZE)
        channels = tl.arange(0, BLOCK_C)
        slot_i = slot_pairs // PATCH_SIZE
        slot_j = slot_pairs % PATCH_SIZE
        residue_i = tl.load(
            PatchResidueIdx + batch * PRS0 + p * PRS1 + slot_i * PRS2
        ).to(tl.int32)
        residue_j = tl.load(
            PatchResidueIdx + batch * PRS0 + q * PRS1 + slot_j * PRS2
        ).to(tl.int32)
        active = (residue_i >= 0) & (residue_j >= 0)
        grad_offsets = (
            batch * GS0
            + tl.maximum(residue_i, 0)[:, None] * GS1
            + tl.maximum(residue_j, 0)[:, None] * GS2
            + channels[None, :] * GS3
        )
        grad = tl.load(
            GradOut + grad_offsets,
            mask=active[:, None] & (channels[None, :] < C),
            other=0.0,
        ).to(tl.float32)
        reduced = tl.sum(grad, axis=0)
        out_offsets = batch * DCS0 + p * DCS1 + q * DCS2 + channels * DCS3
        tl.store(GradCoarse + out_offsets, reduced, mask=channels < C)


    @triton.jit
    def _patch_distogram_dslot_partial_kernel(
        GradOut,
        PatchResidueIdx,
        Partial,
        GS0: tl.constexpr,
        GS1: tl.constexpr,
        GS2: tl.constexpr,
        GS3: tl.constexpr,
        PRS0: tl.constexpr,
        PRS1: tl.constexpr,
        PRS2: tl.constexpr,
        PS0: tl.constexpr,
        PS1: tl.constexpr,
        PS2: tl.constexpr,
        TOTAL_PATCH_PAIRS: tl.constexpr,
        M: tl.constexpr,
        C: tl.constexpr,
        PATCH_SIZE: tl.constexpr,
        CHUNK: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        chunk_id = tl.program_id(0)
        slot_pair = tl.program_id(1)
        pairs = chunk_id * CHUNK + tl.arange(0, CHUNK)
        channels = tl.arange(0, BLOCK_C)
        in_bounds = pairs < TOTAL_PATCH_PAIRS
        q = pairs % M
        batch_p = pairs // M
        p = batch_p % M
        batch = batch_p // M
        slot_i = slot_pair // PATCH_SIZE
        slot_j = slot_pair % PATCH_SIZE
        residue_i = tl.load(
            PatchResidueIdx + batch * PRS0 + p * PRS1 + slot_i * PRS2,
            mask=in_bounds,
            other=-1,
        ).to(tl.int32)
        residue_j = tl.load(
            PatchResidueIdx + batch * PRS0 + q * PRS1 + slot_j * PRS2,
            mask=in_bounds,
            other=-1,
        ).to(tl.int32)
        active = in_bounds & (residue_i >= 0) & (residue_j >= 0)
        grad_offsets = (
            batch[:, None] * GS0
            + tl.maximum(residue_i, 0)[:, None] * GS1
            + tl.maximum(residue_j, 0)[:, None] * GS2
            + channels[None, :] * GS3
        )
        grad = tl.load(
            GradOut + grad_offsets,
            mask=active[:, None] & (channels[None, :] < C),
            other=0.0,
        ).to(tl.float32)
        reduced = tl.sum(grad, axis=0)
        partial_offsets = chunk_id * PS0 + slot_pair * PS1 + channels * PS2
        tl.store(Partial + partial_offsets, reduced, mask=channels < C)


def patch_distogram_expand_triton_supported(
    coarse_logits: Tensor,
    slot_bias: Tensor,
) -> tuple[bool, str | None]:
    if triton is None:
        return False, "Triton is not installed"
    if not coarse_logits.is_cuda or not slot_bias.is_cuda:
        return False, "the Triton path requires CUDA tensors"
    if coarse_logits.dtype != torch.bfloat16:
        return False, "the Triton path currently requires BF16 coarse logits"
    if slot_bias.dtype != torch.float32:
        return False, "the Triton path currently requires FP32 slot bias"
    if int(coarse_logits.shape[-1]) != _BINS:
        return False, "the Triton path currently requires 64 distogram bins"
    if torch.cuda.get_device_capability(coarse_logits.device)[0] < 9:
        return False, "the tuned path requires SM90 or newer"
    return True, None


class _PatchDistogramExpandTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coarse_logits: Tensor,
        slot_bias: Tensor,
        residue_to_patch: Tensor,
        residue_slot: Tensor,
        patch_residue_idx: Tensor,
        residue_mask: Tensor,
    ) -> Tensor:
        batch, patches, length, bins = _validate(
            coarse_logits,
            slot_bias,
            residue_to_patch,
            residue_slot,
            patch_residue_idx,
            residue_mask,
        )
        residue_to_patch = residue_to_patch.to(device=coarse_logits.device, dtype=torch.int32).contiguous()
        residue_slot = residue_slot.to(device=coarse_logits.device, dtype=torch.int32).contiguous()
        patch_residue_idx = patch_residue_idx.to(device=coarse_logits.device, dtype=torch.int32).contiguous()
        residue_mask = residue_mask.to(device=coarse_logits.device, dtype=torch.bool).contiguous()
        output = torch.empty(
            (batch, length, length, bins),
            device=coarse_logits.device,
            dtype=coarse_logits.dtype,
        )
        _patch_distogram_forward_kernel[(batch * length, triton.cdiv(length, 8))](
            coarse_logits,
            slot_bias,
            residue_to_patch,
            residue_slot,
            residue_mask,
            output,
            *coarse_logits.stride(),
            *slot_bias.stride(),
            *output.stride(),
            N=length,
            M=patches,
            C=bins,
            PATCH_SIZE=_PATCH_SIZE,
            BLOCK_J=8,
            BLOCK_C=64,
            num_warps=4,
        )
        ctx.save_for_backward(patch_residue_idx)
        ctx.shape = (batch, patches, length, bins)
        ctx.slot_dtype = slot_bias.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (patch_residue_idx,) = ctx.saved_tensors
        batch, patches, _length, bins = ctx.shape
        grad_coarse = torch.empty(
            (batch, patches, patches, bins),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        _patch_distogram_dcoarse_kernel[(batch * patches * patches,)](
            grad_output,
            patch_residue_idx,
            grad_coarse,
            *grad_output.stride(),
            *patch_residue_idx.stride(),
            *grad_coarse.stride(),
            M=patches,
            C=bins,
            PATCH_SIZE=_PATCH_SIZE,
            BLOCK_C=64,
            num_warps=4,
        )

        total_pairs = batch * patches * patches
        chunk = 32
        chunks = triton.cdiv(total_pairs, chunk)
        partial = torch.empty(
            (chunks, _PATCH_SIZE * _PATCH_SIZE, bins),
            device=grad_output.device,
            dtype=torch.float32,
        )
        _patch_distogram_dslot_partial_kernel[(chunks, _PATCH_SIZE * _PATCH_SIZE)](
            grad_output,
            patch_residue_idx,
            partial,
            *grad_output.stride(),
            *patch_residue_idx.stride(),
            *partial.stride(),
            TOTAL_PATCH_PAIRS=total_pairs,
            M=patches,
            C=bins,
            PATCH_SIZE=_PATCH_SIZE,
            CHUNK=chunk,
            BLOCK_C=64,
            num_warps=4,
        )
        grad_slot = partial.sum(dim=0).reshape(_PATCH_SIZE, _PATCH_SIZE, bins).to(ctx.slot_dtype)
        return grad_coarse, grad_slot, None, None, None, None


def patch_distogram_expand(
    coarse_logits: Tensor,
    slot_bias: Tensor,
    residue_to_patch: Tensor,
    residue_slot: Tensor,
    patch_residue_idx: Tensor,
    residue_mask: Tensor,
    *,
    implementation: Implementation = "auto",
) -> Tensor:
    """Expand patch-pair logits without generic indexed-scatter backward."""

    _validate(
        coarse_logits,
        slot_bias,
        residue_to_patch,
        residue_slot,
        patch_residue_idx,
        residue_mask,
    )
    implementation = str(implementation).lower()
    if implementation not in {"auto", "eager", "triton"}:
        raise ValueError("implementation must be 'auto', 'eager', or 'triton'")
    if implementation == "eager":
        return _expand_eager(
            coarse_logits,
            slot_bias,
            residue_to_patch,
            residue_slot,
            residue_mask,
        )
    supported, reason = patch_distogram_expand_triton_supported(coarse_logits, slot_bias)
    if implementation == "triton" and not supported:
        raise RuntimeError(f"forced Triton patch distogram expansion is unavailable: {reason}")
    if supported:
        return _PatchDistogramExpandTriton.apply(
            coarse_logits,
            slot_bias,
            residue_to_patch,
            residue_slot,
            patch_residue_idx,
            residue_mask,
        )
    return _expand_eager(
        coarse_logits,
        slot_bias,
        residue_to_patch,
        residue_slot,
        residue_mask,
    )


__all__ = ["patch_distogram_expand", "patch_distogram_expand_triton_supported"]
