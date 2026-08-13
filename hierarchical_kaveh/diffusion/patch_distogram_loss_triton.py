"""Exact patch-aware distogram cross entropy without residue-logit expansion."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from hierarchical_kaveh.types import CompactDistogram


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU runtime
    triton = None
    tl = None


Implementation = Literal["auto", "eager", "triton"]
_PATCH_SIZE = 4
_BINS = 64


def patch_distogram_loss_triton_supported(output: CompactDistogram) -> tuple[bool, str | None]:
    """Return whether the tuned compact CE supports this output."""

    coarse = output.coarse_logits
    slot_bias = output.slot_bias
    if triton is None:
        return False, "Triton is not installed"
    if not coarse.is_cuda or not slot_bias.is_cuda:
        return False, "the Triton path requires CUDA tensors"
    if coarse.dtype != torch.bfloat16:
        return False, "the Triton path requires BF16 coarse logits"
    if slot_bias.dtype != torch.float32:
        return False, "the Triton path requires FP32 slot bias"
    if coarse.ndim != 4 or coarse.shape[1] != coarse.shape[2]:
        return False, "coarse logits must have shape [B,M,M,C]"
    if int(coarse.shape[-1]) != _BINS:
        return False, "the tuned path requires 64 bins"
    if tuple(slot_bias.shape) != (_PATCH_SIZE, _PATCH_SIZE, _BINS):
        return False, "slot bias must have shape [4,4,64]"
    if torch.cuda.get_device_capability(coarse.device)[0] < 9:
        return False, "the tuned path requires SM90 or newer"
    return True, None


def _validate(output: CompactDistogram, true_bins: Tensor, square_mask: Tensor) -> tuple[int, int, int]:
    coarse = output.coarse_logits
    if coarse.ndim != 4 or coarse.shape[1] != coarse.shape[2]:
        raise ValueError("coarse logits must have shape [B,M,M,C]")
    batch, patches, _patches, bins = (int(value) for value in coarse.shape)
    if tuple(output.slot_bias.shape) != (_PATCH_SIZE, _PATCH_SIZE, bins):
        raise ValueError(f"slot_bias must have shape {(4, 4, bins)}")
    if output.patch_residue_index.shape != (batch, patches, _PATCH_SIZE):
        raise ValueError("patch_residue_index must have shape [B,M,4]")
    if true_bins.ndim != 3 or true_bins.shape[0] != batch or true_bins.shape[1] != true_bins.shape[2]:
        raise ValueError("true_bins must have shape [B,N,N]")
    length = int(true_bins.shape[1])
    if square_mask.shape != (batch, length, length):
        raise ValueError("square_mask must match true_bins")
    if output.residue_to_patch.shape != (batch, length):
        raise ValueError("residue_to_patch must have shape [B,N]")
    if output.residue_slot.shape != (batch, length) or output.residue_mask.shape != (batch, length):
        raise ValueError("residue slot and mask metadata must have shape [B,N]")
    return batch, patches, bins


def _dense_logits(output: CompactDistogram) -> Tensor:
    """Materialize the exact dense logits for the portable oracle only."""

    coarse = output.coarse_logits
    batch = torch.arange(coarse.shape[0], device=coarse.device)[:, None, None]
    patch_idx = output.residue_to_patch.to(device=coarse.device, dtype=torch.long).clamp_min(0)
    slot_idx = output.residue_slot.to(device=coarse.device, dtype=torch.long).clamp_min(0)
    logits = coarse[batch, patch_idx[:, :, None], patch_idx[:, None, :]]
    logits = logits + output.slot_bias[slot_idx[:, :, None], slot_idx[:, None, :]].to(logits.dtype)
    pair_mask = output.residue_mask.to(device=coarse.device, dtype=torch.bool)
    logits = logits * (pair_mask[:, :, None] & pair_mask[:, None, :])[..., None].to(logits.dtype)
    if output.symmetrize:
        logits = (logits + logits.transpose(-3, -2))
    return logits


if triton is not None:

    @triton.jit
    def _patch_distogram_ce_forward_kernel(
        Coarse,
        SlotBias,
        PatchResidueIdx,
        TrueBins,
        SquareMask,
        PartialLoss,
        CS0: tl.constexpr,
        CS1: tl.constexpr,
        CS2: tl.constexpr,
        CS3: tl.constexpr,
        SS0: tl.constexpr,
        SS1: tl.constexpr,
        SS2: tl.constexpr,
        PS0: tl.constexpr,
        PS1: tl.constexpr,
        PS2: tl.constexpr,
        TS0: tl.constexpr,
        TS1: tl.constexpr,
        TS2: tl.constexpr,
        MS0: tl.constexpr,
        MS1: tl.constexpr,
        MS2: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        C: tl.constexpr,
        SYMMETRIZE: tl.constexpr,
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
        residue_i = tl.load(PatchResidueIdx + batch * PS0 + p * PS1 + slot_i * PS2).to(tl.int32)
        residue_j = tl.load(PatchResidueIdx + batch * PS0 + q * PS1 + slot_j * PS2).to(tl.int32)
        valid_residue = (residue_i >= 0) & (residue_j >= 0)
        safe_i = tl.maximum(residue_i, 0)
        safe_j = tl.maximum(residue_j, 0)
        active = valid_residue & (
            tl.load(
                SquareMask + batch * MS0 + safe_i * MS1 + safe_j * MS2,
                mask=valid_residue,
                other=0,
            )
            != 0
        )
        coarse_pq = tl.load(
            Coarse + batch * CS0 + p * CS1 + q * CS2 + channels[None, :] * CS3,
            mask=channels[None, :] < C,
            other=0.0,
        ).to(tl.float32)
        bias_ij = tl.load(
            SlotBias + slot_i[:, None] * SS0 + slot_j[:, None] * SS1 + channels[None, :] * SS2,
            mask=channels[None, :] < C,
            other=0.0,
        ).to(tl.float32)
        logits = coarse_pq + bias_ij
        if SYMMETRIZE:
            coarse_qp = tl.load(
                Coarse + batch * CS0 + q * CS1 + p * CS2 + channels[None, :] * CS3,
                mask=channels[None, :] < C,
                other=0.0,
            ).to(tl.float32)
            bias_ji = tl.load(
                SlotBias + slot_j[:, None] * SS0 + slot_i[:, None] * SS1 + channels[None, :] * SS2,
                mask=channels[None, :] < C,
                other=0.0,
            ).to(tl.float32)
            logits = logits + coarse_qp + bias_ji
        row_max = tl.max(logits, axis=1)
        log_denom = tl.log(tl.sum(tl.exp(logits - row_max[:, None]), axis=1)) + row_max
        target = tl.load(
            TrueBins + batch * TS0 + safe_i * TS1 + safe_j * TS2,
            mask=valid_residue,
            other=0,
        ).to(tl.int32)
        target_logit = tl.sum(tl.where(channels[None, :] == target[:, None], logits, 0.0), axis=1)
        errors = tl.where(active, log_denom - target_logit, 0.0)
        tl.store(PartialLoss + patch_pair, tl.sum(errors, axis=0))


    @triton.jit
    def _patch_distogram_ce_dcoarse_kernel(
        Coarse,
        SlotBias,
        PatchResidueIdx,
        TrueBins,
        SquareMask,
        Scale,
        GradCoarse,
        CS0: tl.constexpr,
        CS1: tl.constexpr,
        CS2: tl.constexpr,
        CS3: tl.constexpr,
        SS0: tl.constexpr,
        SS1: tl.constexpr,
        SS2: tl.constexpr,
        PS0: tl.constexpr,
        PS1: tl.constexpr,
        PS2: tl.constexpr,
        TS0: tl.constexpr,
        TS1: tl.constexpr,
        TS2: tl.constexpr,
        MS0: tl.constexpr,
        MS1: tl.constexpr,
        MS2: tl.constexpr,
        GS0: tl.constexpr,
        GS1: tl.constexpr,
        GS2: tl.constexpr,
        GS3: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        C: tl.constexpr,
        SYMMETRIZE: tl.constexpr,
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
        residue_i = tl.load(PatchResidueIdx + batch * PS0 + p * PS1 + slot_i * PS2).to(tl.int32)
        residue_j = tl.load(PatchResidueIdx + batch * PS0 + q * PS1 + slot_j * PS2).to(tl.int32)
        valid_residue = (residue_i >= 0) & (residue_j >= 0)
        safe_i = tl.maximum(residue_i, 0)
        safe_j = tl.maximum(residue_j, 0)
        direct = valid_residue & (
            tl.load(
                SquareMask + batch * MS0 + safe_i * MS1 + safe_j * MS2,
                mask=valid_residue,
                other=0,
            )
            != 0
        )
        weight = direct.to(tl.float32)
        coarse_pq = tl.load(
            Coarse + batch * CS0 + p * CS1 + q * CS2 + channels[None, :] * CS3,
            mask=channels[None, :] < C,
            other=0.0,
        ).to(tl.float32)
        bias_ij = tl.load(
            SlotBias + slot_i[:, None] * SS0 + slot_j[:, None] * SS1 + channels[None, :] * SS2,
            mask=channels[None, :] < C,
            other=0.0,
        ).to(tl.float32)
        logits = coarse_pq + bias_ij
        if SYMMETRIZE:
            reverse = valid_residue & (
                tl.load(
                    SquareMask + batch * MS0 + safe_j * MS1 + safe_i * MS2,
                    mask=valid_residue,
                    other=0,
                )
                != 0
            )
            weight = weight + reverse.to(tl.float32)
            coarse_qp = tl.load(
                Coarse + batch * CS0 + q * CS1 + p * CS2 + channels[None, :] * CS3,
                mask=channels[None, :] < C,
                other=0.0,
            ).to(tl.float32)
            bias_ji = tl.load(
                SlotBias + slot_j[:, None] * SS0 + slot_i[:, None] * SS1 + channels[None, :] * SS2,
                mask=channels[None, :] < C,
                other=0.0,
            ).to(tl.float32)
            logits = logits + coarse_qp + bias_ji
        row_max = tl.max(logits, axis=1)
        probs = tl.exp(logits - row_max[:, None])
        probs = probs / tl.sum(probs, axis=1)[:, None]
        target = tl.load(
            TrueBins + batch * TS0 + safe_i * TS1 + safe_j * TS2,
            mask=valid_residue,
            other=0,
        ).to(tl.int32)
        grad = probs - (channels[None, :] == target[:, None]).to(tl.float32)
        sample_scale = tl.load(Scale + batch).to(tl.float32)
        grad = grad * weight[:, None] * sample_scale
        reduced = tl.sum(grad, axis=0)
        tl.store(
            GradCoarse + batch * GS0 + p * GS1 + q * GS2 + channels * GS3,
            reduced,
            mask=channels < C,
        )


    @triton.jit
    def _patch_distogram_ce_dslot_partial_kernel(
        Coarse,
        SlotBias,
        PatchResidueIdx,
        TrueBins,
        SquareMask,
        Scale,
        Partial,
        CS0: tl.constexpr,
        CS1: tl.constexpr,
        CS2: tl.constexpr,
        CS3: tl.constexpr,
        SS0: tl.constexpr,
        SS1: tl.constexpr,
        SS2: tl.constexpr,
        PS0: tl.constexpr,
        PS1: tl.constexpr,
        PS2: tl.constexpr,
        TS0: tl.constexpr,
        TS1: tl.constexpr,
        TS2: tl.constexpr,
        MS0: tl.constexpr,
        MS1: tl.constexpr,
        MS2: tl.constexpr,
        DS0: tl.constexpr,
        DS1: tl.constexpr,
        DS2: tl.constexpr,
        TOTAL_PATCH_PAIRS: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        C: tl.constexpr,
        SYMMETRIZE: tl.constexpr,
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
            PatchResidueIdx + batch * PS0 + p * PS1 + slot_i * PS2,
            mask=in_bounds,
            other=-1,
        ).to(tl.int32)
        residue_j = tl.load(
            PatchResidueIdx + batch * PS0 + q * PS1 + slot_j * PS2,
            mask=in_bounds,
            other=-1,
        ).to(tl.int32)
        valid_residue = in_bounds & (residue_i >= 0) & (residue_j >= 0)
        safe_i = tl.maximum(residue_i, 0)
        safe_j = tl.maximum(residue_j, 0)
        direct = valid_residue & (
            tl.load(
                SquareMask + batch * MS0 + safe_i * MS1 + safe_j * MS2,
                mask=valid_residue,
                other=0,
            )
            != 0
        )
        weight = direct.to(tl.float32)
        coarse_pq = tl.load(
            Coarse + batch[:, None] * CS0 + p[:, None] * CS1 + q[:, None] * CS2 + channels[None, :] * CS3,
            mask=in_bounds[:, None] & (channels[None, :] < C),
            other=0.0,
        ).to(tl.float32)
        bias_ij = tl.load(
            SlotBias + slot_i * SS0 + slot_j * SS1 + channels[None, :] * SS2,
            mask=channels[None, :] < C,
            other=0.0,
        ).to(tl.float32)
        logits = coarse_pq + bias_ij
        if SYMMETRIZE:
            reverse = valid_residue & (
                tl.load(
                    SquareMask + batch * MS0 + safe_j * MS1 + safe_i * MS2,
                    mask=valid_residue,
                    other=0,
                )
                != 0
            )
            weight = weight + reverse.to(tl.float32)
            coarse_qp = tl.load(
                Coarse + batch[:, None] * CS0 + q[:, None] * CS1 + p[:, None] * CS2 + channels[None, :] * CS3,
                mask=in_bounds[:, None] & (channels[None, :] < C),
                other=0.0,
            ).to(tl.float32)
            bias_ji = tl.load(
                SlotBias + slot_j * SS0 + slot_i * SS1 + channels[None, :] * SS2,
                mask=channels[None, :] < C,
                other=0.0,
            ).to(tl.float32)
            logits = logits + coarse_qp + bias_ji
        row_max = tl.max(logits, axis=1)
        probs = tl.exp(logits - row_max[:, None])
        probs = probs / tl.sum(probs, axis=1)[:, None]
        target = tl.load(
            TrueBins + batch * TS0 + safe_i * TS1 + safe_j * TS2,
            mask=valid_residue,
            other=0,
        ).to(tl.int32)
        grad = probs - (channels[None, :] == target[:, None]).to(tl.float32)
        sample_scale = tl.load(Scale + batch, mask=in_bounds, other=0.0).to(tl.float32)
        grad = grad * weight[:, None] * sample_scale[:, None]
        reduced = tl.sum(grad, axis=0)
        tl.store(
            Partial + chunk_id * DS0 + slot_pair * DS1 + channels * DS2,
            reduced,
            mask=channels < C,
        )


class _PatchDistogramCrossEntropyTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coarse_logits: Tensor,
        slot_bias: Tensor,
        patch_residue_index: Tensor,
        true_bins: Tensor,
        square_mask: Tensor,
        pair_count: Tensor,
        eps: float,
        symmetrize: bool,
    ) -> Tensor:
        batch, patches, _patches, bins = (int(value) for value in coarse_logits.shape)
        patch_residue_index = patch_residue_index.to(
            device=coarse_logits.device, dtype=torch.int32
        ).contiguous()
        true_bins = true_bins.to(device=coarse_logits.device, dtype=torch.int32).contiguous()
        square_mask = square_mask.to(device=coarse_logits.device, dtype=torch.bool).contiguous()
        pair_count = pair_count.to(device=coarse_logits.device, dtype=torch.float32).contiguous()
        total_pairs = batch * patches * patches
        partial_loss = torch.empty(total_pairs, device=coarse_logits.device, dtype=torch.float32)
        _patch_distogram_ce_forward_kernel[(total_pairs,)](
            coarse_logits,
            slot_bias,
            patch_residue_index,
            true_bins,
            square_mask,
            partial_loss,
            *coarse_logits.stride(),
            *slot_bias.stride(),
            *patch_residue_index.stride(),
            *true_bins.stride(),
            *square_mask.stride(),
            M=patches,
            N=int(true_bins.shape[1]),
            C=bins,
            SYMMETRIZE=bool(symmetrize),
            PATCH_SIZE=_PATCH_SIZE,
            BLOCK_C=64,
            num_warps=4,
            num_stages=2,
        )
        loss_sum = partial_loss.reshape(batch, -1).sum(dim=1)
        per_sample = loss_sum / (float(eps) + pair_count)
        ctx.save_for_backward(
            coarse_logits,
            slot_bias,
            patch_residue_index,
            true_bins,
            square_mask,
            pair_count,
        )
        ctx.eps = float(eps)
        ctx.symmetrize = bool(symmetrize)
        return per_sample

    @staticmethod
    def backward(ctx, grad_per_sample: Tensor):
        coarse, slot_bias, patch_idx, true_bins, square_mask, pair_count = ctx.saved_tensors
        batch, patches, _patches, bins = (int(value) for value in coarse.shape)
        scale = (
            grad_per_sample.to(device=coarse.device, dtype=torch.float32)
            / (float(ctx.eps) + pair_count)
        ).contiguous()
        grad_coarse = torch.empty_like(coarse) if ctx.needs_input_grad[0] else None
        grad_slot = None
        common = {
            "M": patches,
            "N": int(true_bins.shape[1]),
            "C": bins,
            "SYMMETRIZE": bool(ctx.symmetrize),
            "PATCH_SIZE": _PATCH_SIZE,
            "BLOCK_C": 64,
            "num_warps": 4,
            "num_stages": 2,
        }
        total_pairs = batch * patches * patches
        if grad_coarse is not None:
            _patch_distogram_ce_dcoarse_kernel[(total_pairs,)](
                coarse,
                slot_bias,
                patch_idx,
                true_bins,
                square_mask,
                scale,
                grad_coarse,
                *coarse.stride(),
                *slot_bias.stride(),
                *patch_idx.stride(),
                *true_bins.stride(),
                *square_mask.stride(),
                *grad_coarse.stride(),
                **common,
            )
        if ctx.needs_input_grad[1]:
            chunk = 32
            chunks = triton.cdiv(total_pairs, chunk)
            partial = torch.empty(
                (chunks, _PATCH_SIZE * _PATCH_SIZE, bins),
                device=coarse.device,
                dtype=torch.float32,
            )
            _patch_distogram_ce_dslot_partial_kernel[(chunks, _PATCH_SIZE * _PATCH_SIZE)](
                coarse,
                slot_bias,
                patch_idx,
                true_bins,
                square_mask,
                scale,
                partial,
                *coarse.stride(),
                *slot_bias.stride(),
                *patch_idx.stride(),
                *true_bins.stride(),
                *square_mask.stride(),
                *partial.stride(),
                TOTAL_PATCH_PAIRS=total_pairs,
                CHUNK=chunk,
                **common,
            )
            grad_slot = partial.sum(dim=0).reshape_as(slot_bias).to(slot_bias.dtype)
        return grad_coarse, grad_slot, None, None, None, None, None, None


def patch_distogram_cross_entropy(
    output: CompactDistogram,
    true_bins: Tensor,
    square_mask: Tensor,
    pair_count: Tensor,
    *,
    eps: float,
    implementation: Implementation = "auto",
) -> Tensor:
    """Return exact per-sample CE from compact patch logits."""

    _validate(output, true_bins, square_mask)
    implementation = str(implementation).lower()
    if implementation not in {"auto", "eager", "triton"}:
        raise ValueError("implementation must be auto, eager, or triton")
    supported, reason = patch_distogram_loss_triton_supported(output)
    if implementation == "triton" and not supported:
        raise RuntimeError(f"forced Triton patch distogram loss is unavailable: {reason}")
    if implementation == "triton" or (implementation == "auto" and supported):
        return _PatchDistogramCrossEntropyTriton.apply(
            output.coarse_logits,
            output.slot_bias,
            output.patch_residue_index,
            true_bins,
            square_mask,
            pair_count,
            float(eps),
            bool(output.symmetrize),
        )

    logits = _dense_logits(output)
    errors = F.cross_entropy(logits.float().movedim(-1, 1), true_bins, reduction="none")
    loss_sum = torch.sum(errors * square_mask.to(dtype=errors.dtype), dim=(-1, -2))
    return loss_sum / (float(eps) + pair_count.to(dtype=torch.float32))


__all__ = [
    "patch_distogram_cross_entropy",
    "patch_distogram_loss_triton_supported",
]
