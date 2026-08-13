"""Guarded entrypoint for fused Atom14 residue-window attention."""

from torch import Tensor


def supported(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    atom_mask: Tensor,
    segment_index: Tensor,
    radius: int,
) -> bool:
    try:
        from ._atom_window_triton import atom_window_attention_triton_supported

        return atom_window_attention_triton_supported(
            query, key, value, atom_mask, segment_index, radius=radius
        )[0]
    except (ImportError, RuntimeError):
        return False


def atom_window_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    atom_mask: Tensor,
    segment_index: Tensor,
    radius: int,
    scale: float,
) -> Tensor:
    from ._atom_window_triton import atom_window_attention_triton

    return atom_window_attention_triton(
        query, key, value, atom_mask, segment_index, radius=radius, scale=scale
    )
