import torch

import pytest

from hierarchical_kaveh.model.patch import (
    PATCH_SIZE,
    bucket_patch_capacity,
    build_patch_layout,
)


def _layout(lengths: list[int], padded: int):
    batch = len(lengths)
    positions = torch.arange(padded)[None].expand(batch, -1)
    mask = positions < torch.tensor(lengths)[:, None]
    return build_patch_layout(
        mask,
        torch.zeros(batch, padded, dtype=torch.long),
        positions,
    )


def test_regular_pack_unpack_preserves_nondivisible_tail_and_padding():
    layout = _layout([7, 5], 9)
    values = torch.arange(18).reshape(2, 9, 1).float()

    packed = layout.pack(values)
    unpacked = layout.unpack(packed)

    assert packed.shape == (2, 2, PATCH_SIZE, 1)
    assert layout.slot_mask[0].tolist() == [[True] * 4, [True, True, True, False]]
    assert layout.slot_mask[1].tolist() == [[True] * 4, [True, False, False, False]]
    assert torch.equal(unpacked[0, :7], values[0, :7])
    assert torch.equal(unpacked[1, :5], values[1, :5])
    assert torch.count_nonzero(unpacked[0, 7:]) == 0
    assert torch.count_nonzero(unpacked[1, 5:]) == 0


def test_multichain_patches_never_cross_chain_boundaries():
    residue_count = 8
    mask = torch.ones(1, residue_count, dtype=torch.bool)
    chains = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    residue_index = torch.tensor([[10, 11, 12, 4, 5, 6, 7, 8]])
    layout = build_patch_layout(mask, chains, residue_index)

    assert layout.patch_lengths.tolist() == [3]  # ceil(3/4) + ceil(5/4)
    assert layout.patch_residue_index.tolist() == [[[0, 1, 2, -1], [3, 4, 5, 6], [7, -1, -1, -1]]]
    for patch_chains, valid in zip(layout.slot_chain_index[0], layout.slot_mask[0], strict=True):
        assert torch.unique(patch_chains[valid]).numel() == 1


def test_residue_index_discontinuity_starts_a_fresh_partial_patch():
    mask = torch.ones(1, 6, dtype=torch.bool)
    residue_index = torch.tensor([[0, 1, 9, 10, 11, 12]])
    layout = build_patch_layout(
        mask,
        torch.zeros_like(residue_index),
        residue_index,
    )
    assert layout.patch_residue_index.tolist() == [[[0, 1, -1, -1], [2, 3, 4, 5]]]


def test_pack_unpack_is_a_bijection_for_arbitrary_chain_layout():
    mask = torch.tensor([[True, True, True, True, True, True, False]])
    chains = torch.tensor([[0, 0, 1, 1, 1, 2, 0]])
    residue_index = torch.tensor([[1, 2, 1, 2, 3, 8, 0]])
    layout = build_patch_layout(mask, chains, residue_index)
    values = torch.randn(1, 7, 3, 2)
    result = layout.unpack(layout.pack(values))
    assert torch.equal(result[:, :6], values[:, :6])
    assert torch.count_nonzero(result[:, 6:]) == 0


def test_static_capacity_pads_without_changing_valid_assignments():
    mask = torch.tensor([[True, True, True, True, True, True, False, False]])
    chains = torch.tensor([[0, 0, 1, 1, 1, 2, -1, -1]])
    residue_index = torch.tensor([[1, 2, 1, 2, 3, 8, -1, -1]])
    exact = build_patch_layout(mask, chains, residue_index)
    static = build_patch_layout(
        mask,
        chains,
        residue_index,
        patch_capacity=bucket_patch_capacity(int(exact.patch_lengths.max())),
    )

    assert static.slot_mask.shape == (1, 4, PATCH_SIZE)
    assert torch.equal(static.residue_to_patch, exact.residue_to_patch)
    values = torch.randn(1, 8, 3)
    torch.testing.assert_close(
        static.unpack(static.pack(values)),
        exact.unpack(exact.pack(values)),
    )


def test_static_capacity_rejects_layout_overflow_in_eager_validation():
    mask = torch.ones(1, 8, dtype=torch.bool)
    chains = torch.arange(8)[None]
    residue_index = torch.ones_like(chains)
    with pytest.raises(ValueError, match="smaller"):
        build_patch_layout(
            mask,
            chains,
            residue_index,
            patch_capacity=4,
        )
