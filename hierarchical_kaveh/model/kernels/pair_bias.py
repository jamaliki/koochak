"""Fused pair LayerNorm, head projection, and register padding."""

from torch import Tensor


def project_pair_bias(
    pair: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    projection_weight: Tensor,
    output_size: int,
    registers: int,
    residue_lengths: Tensor,
) -> Tensor:
    from ._pair_bias_projection_triton import pair_bias_projection_triton

    return pair_bias_projection_triton(
        pair, norm_weight, norm_bias, projection_weight, 1e-5,
        output_size, registers, residue_lengths,
    )
