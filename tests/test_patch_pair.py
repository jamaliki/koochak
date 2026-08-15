import torch

from hierarchical_kaveh.model.pair import DistogramHead, PairInitializer, expand_distogram
from hierarchical_kaveh.model.patch import build_patch_layout


def _one_patch_layout():
    mask = torch.ones(1, 4, dtype=torch.bool)
    index = torch.arange(4)[None]
    return build_patch_layout(mask, torch.zeros_like(index), index, torch.zeros_like(index))


def test_pair_geometry_contains_all_16_ordered_raw_distances():
    layout = _one_patch_layout()
    ca = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]]]])
    initializer = PairInitializer(pair_dim=3, rbf_bins=5, distance_min=0.05, distance_max=8.0)

    features = initializer.geometry_features(ca, layout)

    assert features.shape == (1, 1, 1, 4, 4, 5)
    assert not torch.equal(features[0, 0, 0, 0, 1], features[0, 0, 0, 0, 2])
    assert torch.equal(features[0, 0, 0, 0, 1], features[0, 0, 0, 1, 0])
    assert torch.count_nonzero(features) == 16 * 5


def test_pair_state_uses_topology_even_when_geometry_scale_is_zero():
    layout = _one_patch_layout()
    initializer = PairInitializer(pair_dim=4, rbf_bins=3, distance_min=0.05, distance_max=8.0)
    initializer.geometry_scale.data.zero_()
    ca = torch.randn(1, 1, 4, 3)
    pair = initializer(ca, layout, torch.float32)
    assert pair.shape == (1, 1, 1, 4)
    assert torch.count_nonzero(pair) > 0


def test_local_center_geometry_separates_intra_patch_and_center_distances():
    mask = torch.ones(1, 8, dtype=torch.bool)
    index = torch.arange(8)[None]
    layout = build_patch_layout(mask, torch.zeros_like(index), index, torch.zeros_like(index))
    ca = torch.zeros(1, 8, 3)
    ca[0, :, 0] = torch.arange(8)
    initializer = PairInitializer(
        pair_dim=4,
        rbf_bins=5,
        distance_min=0.05,
        distance_max=16.0,
        geometry_mode="local_center",
    )

    local, center = initializer.local_center_features(layout.pack(ca), layout)

    assert local.shape == (1, 2, 4, 4, 5)
    assert center.shape == (1, 2, 2, 5)
    assert torch.allclose(center[0, 0, 1], center[0, 1, 0])
    assert torch.allclose(local[0, 0, 0, 1], local[0, 0, 1, 0])
    assert not torch.allclose(center[0, 0, 0], center[0, 0, 1])


def test_local_center_can_add_all_cross_patch_distances():
    mask = torch.ones(1, 8, dtype=torch.bool)
    index = torch.arange(8)[None]
    layout = build_patch_layout(mask, torch.zeros_like(index), index, torch.zeros_like(index))
    ca = layout.pack(torch.randn(1, 8, 14, 3)[..., 1, :])
    initializer = PairInitializer(
        pair_dim=4,
        rbf_bins=5,
        distance_min=0.05,
        distance_max=16.0,
        geometry_mode="local_center",
        cross_patch_geometry=True,
        cross_patch_extrema=True,
    )

    cross = initializer.geometry_features(ca, layout)
    extrema = initializer.cross_patch_extrema_features(ca, layout)
    pair = initializer(ca, layout, torch.float32)

    assert cross.shape == (1, 2, 2, 4, 4, 5)
    assert extrema.shape == (1, 2, 2, 2, 5)
    assert pair.shape == (1, 2, 2, 4)
    assert torch.isfinite(pair).all()


def test_compact_distogram_expands_with_slot_bias_and_symmetry():
    mask = torch.ones(1, 5, dtype=torch.bool)
    index = torch.arange(5)[None]
    layout = build_patch_layout(mask, torch.zeros_like(index), index, torch.zeros_like(index))
    head = DistogramHead(pair_dim=4, bins=3)
    pair = torch.randn(1, 2, 2, 4)
    head.slot_bias.data.copy_(torch.arange(4 * 4 * 3).reshape(4, 4, 3))

    dense = expand_distogram(head(pair, layout))

    assert dense.shape == (1, 5, 5, 3)
    assert torch.allclose(dense, dense.transpose(1, 2))
