"""Guarded entrypoints for packed pair multiplication and fused residuals."""

from torch import Tensor


def packed_gated_operands(
    combined: Tensor,
    pair_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    from ._triangle_packed_triton import triangle_mul_packed_combined_gated_outgate

    return triangle_mul_packed_combined_gated_outgate(combined, pair_mask)


def gated_residual(
    pair: Tensor,
    update: Tensor,
    gate: Tensor,
    pair_mask: Tensor,
    dropout_values: Tensor,
    scale: Tensor,
) -> Tensor:
    from ._gated_residual_triton import pair_gated_residual_triton

    return pair_gated_residual_triton(
        pair, update, gate, pair_mask, dropout_values, scale,
        columnwise=False, apply_mask=False,
    )
