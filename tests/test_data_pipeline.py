import json

import numpy as np
import pytest
import torch

from hierarchical_kaveh.config import DataConfig, DiffusionConfig
from hierarchical_kaveh.data import (
    ShardCache,
    build_train_dataloader,
    index_shards,
    load_sample,
    shard_references,
)
from hierarchical_kaveh.diffusion import corrupt_structure


def _ragged_fixture(tmp_path):
    lengths = (5, 7, 3)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    total = int(offsets[-1])
    pos = np.zeros((total, 14, 3), dtype=np.float32)
    mask = np.zeros((total, 14), dtype=bool)
    aatype = np.arange(total, dtype=np.int64) % 20
    chain_idx = np.zeros(total, dtype=np.int64)
    res_idx = np.zeros(total, dtype=np.int64)
    for sample_idx, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        local = np.arange(stop - start)
        pos[start:stop, :, 0] = local[:, None] * 3.0
        mask[start:stop, :4] = True
        res_idx[start:stop] = local + 1
    # A multichain, non-p=4 sample whose chain boundary remains spatially local.
    chain_idx[3:5] = 1
    res_idx[3:5] = (1, 2)
    shard = tmp_path / "shard.npz"
    np.savez_compressed(
        shard,
        pos=pos,
        mask=mask,
        aatype=aatype,
        chain_idx=chain_idx,
        res_idx=res_idx,
        sample_offsets=offsets,
        cond=np.asarray([[95.0], [85.0], [70.0]], dtype=np.float32),
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps([{"shard": shard.name, "count": 3, "ids": [0], "cond_feature_names": ["mean_plddt"]}])
    )
    return metadata


def test_ragged_reader_preserves_multichain_non_divisible_by_four(tmp_path) -> None:
    references = index_shards(_ragged_fixture(tmp_path), min_length=4)
    assert [reference.length for reference in references] == [5, 7]
    sample = load_sample(references[0])
    assert sample["atom14_coordinates"].shape == (5, 14, 3)
    assert sample["atom14_mask"].all()  # unified Atom14 repeats missing atoms at CA
    assert sample["chain_idx"].tolist() == [0, 0, 0, 1, 1]
    assert sample["res_idx"].tolist() == [1, 2, 3, 1, 2]
    # Chain metadata, rather than this distance-only feature, segments patches.
    assert not sample["chain_breaks_per_residue"].any()


def test_reader_distinguishes_virtual_sidechains_from_unresolved_backbone(tmp_path) -> None:
    metadata = _ragged_fixture(tmp_path)
    shard = tmp_path / "shard.npz"
    with np.load(shard) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["mask"][0, 0] = False  # unresolved N is not a virtual atom
    arrays["mask"][1, 1] = False  # a residue without CA is removed
    np.savez_compressed(shard, **arrays)

    sample = load_sample(index_shards(metadata, min_length=4)[0])
    assert sample["res_idx"].tolist() == [1, 3, 1, 2]
    assert not sample["atom14_mask"][0, 0]
    assert sample["atom14_mask"][0, 4:].all()
    assert torch.equal(
        sample["atom14_coordinates"][0, 4:],
        sample["atom14_coordinates"][0, 1].expand(10, -1),
    )


def test_index_excludes_post_filter_short_samples_before_worker_partition(tmp_path) -> None:
    metadata = _ragged_fixture(tmp_path)
    shard = tmp_path / "shard.npz"
    with np.load(shard) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["mask"][:5, 1] = False
    np.savez_compressed(shard, **arrays)

    references = index_shards(metadata, min_length=4)
    assert [reference.index for reference in references] == [1]
    assert references[0].length == 7


def test_index_rejects_dataset_without_enough_resolved_ca(tmp_path) -> None:
    metadata = _ragged_fixture(tmp_path)
    shard = tmp_path / "shard.npz"
    with np.load(shard) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["mask"][:, 1] = False
    np.savez_compressed(shard, **arrays)

    with pytest.raises(ValueError, match="resolved C-alpha"):
        index_shards(metadata, min_length=4)


def test_index_rejects_object_arrays_without_pickle_deserialization(tmp_path) -> None:
    metadata = _ragged_fixture(tmp_path)
    shard = tmp_path / "shard.npz"
    with np.load(shard) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["mask"] = arrays["mask"].astype(object)
    np.savez_compressed(shard, **arrays)

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        index_shards(metadata)


def test_rank_and_worker_partition_is_deterministic_and_exhaustive(tmp_path) -> None:
    references = index_shards(_ragged_fixture(tmp_path))
    partitions = [
        shard_references(references, rank=rank, world_size=2, worker_id=worker, num_workers=2)
        for rank in range(2)
        for worker in range(2)
    ]
    flattened = [reference.index for partition in partitions for reference in partition]
    assert sorted(flattened) == [0, 1, 2]
    assert len(flattened) == len(set(flattened))
    assert partitions == [
        shard_references(references, rank=rank, world_size=2, worker_id=worker, num_workers=2)
        for rank in range(2)
        for worker in range(2)
    ]


def test_corruption_hides_sequence_and_preserves_rigid_internal_geometry(tmp_path) -> None:
    clean = load_sample(index_shards(_ragged_fixture(tmp_path), min_length=4)[0])
    sample = corrupt_structure(
        clean,
        sigma=1.0,
        generator=torch.Generator().manual_seed(7),
        translation_std=1.0,
    )
    assert sample["aatype_input"].eq(20).all()
    before = torch.cdist(clean["atom14_coordinates"][:, 1], clean["atom14_coordinates"][:, 1])
    after = torch.cdist(sample["x0"][:, 1], sample["x0"][:, 1])
    torch.testing.assert_close(before, after, atol=2e-5, rtol=2e-5)


def test_standard_edm_noise_has_configured_variance(tmp_path) -> None:
    clean = load_sample(index_shards(_ragged_fixture(tmp_path), min_length=4)[0])
    sigma = 5.0
    errors = []
    for seed in range(128):
        sample = corrupt_structure(
            clean,
            sigma=sigma,
            generator=torch.Generator().manual_seed(seed),
            translation_std=0.0,
        )
        errors.append((sample["x_t"] - sample["x0"]).square().flatten())
    torch.testing.assert_close(torch.cat(errors).mean(), torch.tensor(sigma**2), atol=0.5, rtol=0.0)


def test_worker_local_shard_cache_decompresses_once(tmp_path, monkeypatch) -> None:
    references = index_shards(_ragged_fixture(tmp_path))
    calls = 0
    original_load = np.load

    def counted_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    cache = ShardCache(1)
    load_sample(references[0], cache=cache)
    load_sample(references[1], cache=cache)
    load_sample(references[0], cache=cache)
    assert calls == 1


def test_loader_buckets_and_prefix_pads(tmp_path) -> None:
    metadata = _ragged_fixture(tmp_path)
    data = DataConfig(
        metadata_path=str(metadata),
        min_length=4,
        max_length=8,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        length_buckets=(4, 8),
    )
    batch = next(iter(build_train_dataloader(data, DiffusionConfig(), sigma_data=16.0)))
    assert batch["x_t"].shape == (2, 8, 14, 3)
    assert batch["t"].shape == (2, 8, 14)
    assert set(batch["lengths"].tolist()) <= {5, 7}
    for row, length in enumerate(batch["lengths"].tolist()):
        assert batch["residue_mask"][row, :length].all()
        assert not batch["residue_mask"][row, length:].any()
        assert batch["aatype_input"][row, length:].eq(21).all()
