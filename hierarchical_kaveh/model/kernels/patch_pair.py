"""Guarded entrypoint for fused p=4 distance/RBF projection."""

from torch import Tensor


def project_patch_distances(
    ca_slots: Tensor,
    slot_mask: Tensor,
    projection_weight: Tensor,
    distance_min: float,
    distance_max: float,
) -> Tensor:
    from ._patch_pair_rbf_triton import patch_pair_rbf_projection_triton

    return patch_pair_rbf_projection_triton(
        ca_slots, slot_mask, projection_weight, distance_min, distance_max
    )
