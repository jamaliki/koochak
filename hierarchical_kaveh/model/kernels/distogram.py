"""Fused compact-to-dense distogram expansion."""

from torch import Tensor


def expand(
    coarse_logits: Tensor,
    slot_bias: Tensor,
    residue_to_patch: Tensor,
    residue_slot: Tensor,
    patch_residue_index: Tensor,
    residue_mask: Tensor,
) -> Tensor:
    from ._distogram_expand_triton import patch_distogram_expand

    return patch_distogram_expand(
        coarse_logits, slot_bias, residue_to_patch, residue_slot,
        patch_residue_index, residue_mask,
    )
