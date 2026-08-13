"""Fused dual Q/K LayerNorm used at production BF16 sizes."""

from torch import Tensor, nn


def normalize(qkv: Tensor, query_norm: nn.LayerNorm, key_norm: nn.LayerNorm) -> tuple[Tensor, Tensor, Tensor]:
    from ._qk_layernorm_triton import dual_qk_layernorm_triton

    return dual_qk_layernorm_triton(
        qkv, query_norm, key_norm, num_warps=1, backward_impl="two_stage"
    )
