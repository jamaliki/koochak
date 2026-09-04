import json
from dataclasses import replace

import numpy as np
import pytest
import torch

import hierarchical_kaveh.data.shards as shard_module
import hierarchical_kaveh.data.pipeline as pipeline_module
from hierarchical_kaveh.config import DataConfig, DataMixtureConfig, DiffusionConfig
from hierarchical_kaveh.data import (
    ShardCache,
    build_train_dataloader,
    index_shards,
    load_sample,
    shard_references,
)
from hierarchical_kaveh.diffusion import corrupt_structure
from hierarchical_kaveh.residue_constants import ATOM14_NAMES, RESTYPES, physical_atom14_mask
from scripts.materialize_ca_distance_exclusions import materialize_metadata


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
        mask[start:stop] = True
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
        cond=np.asarray([[95.0, 0.2], [85.0, 0.4], [70.0, 0.6]], dtype=np.float32),
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            [
                {
                    "shard": shard.name,
                    "count": 3,
                    "ids": [0],
                    "cond_feature_names": ["mean_plddt", "loop_content"],
                    "ca_distance_validation_max": 4.0,
                    "excluded_ca_distance_samples": [],
                }
            ]
        )
    )
    return metadata


def test_cysteine_has_one_sulfur_and_no_phantom_atom14_slot() -> None:
    cysteine = RESTYPES.index("C")

    assert ATOM14_NAMES[cysteine] == ("N", "CA", "C", "O", "CB", "SG")
    assert physical_atom14_mask(torch.tensor([cysteine])).tolist() == [
        [True, True, True, True, True, True, False, False, False, False, False, False, False, False]
    ]


def test_ragged_reader_preserves_multichain_non_divisible_by_four(tmp_path) -> None:
    references = index_shards(_ragged_fixture(tmp_path), min_length=4)
    assert [reference.length for reference in references] == [5, 7]
    sample = load_sample(references[0])
    assert sample["atom14_coordinates"].shape == (5, 14, 3)
    assert sample["model_atom_mask"].all()
    assert sample["coordinate_mask"].all()
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
    arrays["mask"][2, 4] = False  # unresolved real ASN CB is not supervised
    np.savez_compressed(shard, **arrays)

    sample = load_sample(index_shards(metadata, min_length=4)[0])
    assert sample["res_idx"].tolist() == [1, 3, 1, 2]
    assert sample["model_atom_mask"].all()
    assert not sample["coordinate_mask"][0, 0]
    assert not sample["coordinate_mask"][1, 4]
    assert sample["coordinate_mask"][0, 5:].all()
    assert torch.equal(
        sample["atom14_coordinates"][0, 4:],
        sample["atom14_coordinates"][0, 1].expand(10, -1),
    )
    assert torch.equal(sample["atom14_coordinates"][1, 4], sample["atom14_coordinates"][1, 1])


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


def test_quality_filters_are_strict_and_applied_before_partition(tmp_path) -> None:
    metadata = _ragged_fixture(tmp_path)
    shard = tmp_path / "shard.npz"
    with np.load(shard) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["cond"] = np.asarray(
        [[95.0, 0.5], [80.0, 0.4], [85.0, 0.49]],
        dtype=np.float32,
    )
    np.savez_compressed(shard, **arrays)

    references = index_shards(
        metadata,
        mean_plddt_min=80.0,
        loop_content_max=0.5,
    )
    assert [reference.index for reference in references] == [2]


def test_training_dataset_forwards_all_data_filters(monkeypatch) -> None:
    captured = {}

    def fake_index_shards(metadata_path, **filters):
        captured.update(metadata_path=metadata_path, **filters)
        return []

    monkeypatch.setattr(pipeline_module, "index_shards", fake_index_shards)
    data = DataConfig(
        metadata_path="/data/metadata.json",
        min_length=32,
        max_length=256,
        mean_plddt_min=80.0,
        loop_length_max=15,
        loop_content_max=0.4,
        packing_density_min=0.3,
    )
    pipeline_module.TrainingBatchDataset(data, DiffusionConfig(), sigma_data=1.0)

    assert captured == {
        "metadata_path": "/data/metadata.json",
        "min_length": 32,
        "max_length": 256,
        "mean_plddt_min": 80.0,
        "loop_length_max": 15,
        "loop_content_max": 0.4,
        "packing_density_min": 0.3,
    }


def test_mixture_uses_exclusive_pool_and_exact_periodic_source_ratio(monkeypatch, tmp_path) -> None:
    references = [
        shard_module.SampleReference(tmp_path / "s.npz", index, 0, 1, 32)
        for index in range(5)
    ]
    calls = []

    def fake_index_shards(_metadata_path, **filters):
        calls.append(filters)
        return references[:4] if filters["loop_content_max"] == 0.4 else references

    monkeypatch.setattr(pipeline_module, "index_shards", fake_index_shards)
    data = DataConfig(
        metadata_path="/data/metadata.json",
        min_length=32,
        max_length=128,
        mean_plddt_min=80.0,
        loop_length_max=15,
        loop_content_max=0.4,
        packing_density_min=0.3,
        mixture=DataMixtureConfig(strict_probability=0.75, broader_probability=0.25),
    )
    pools = pipeline_module._mixture_references(data)
    schedule = pipeline_module._MixtureSchedule(0.75)
    observed = [schedule.pop() for _ in range(8)]

    assert len(pools["strict"]) == 4
    assert len(pools["broader_exclusive"]) == 1
    assert calls[1] == {
        "min_length": 32,
        "max_length": 128,
        "mean_plddt_min": 80.0,
        "loop_length_max": None,
        "loop_content_max": 0.5,
        "packing_density_min": None,
    }
    assert observed.count("strict") == 6
    assert observed.count("broader_exclusive") == 2


def test_mixture_worker_assignment_is_disjoint_with_imbalanced_shard_coverage(tmp_path) -> None:
    strict = tuple(
        shard_module.SampleReference(tmp_path / "strict-only-shard.npz", index, 0, 1, 32)
        for index in range(8)
    )
    broader = tuple(
        shard_module.SampleReference(tmp_path / "broader-only-shard.npz", index, 0, 1, 32)
        for index in range(8)
    )
    sources = {"strict": strict, "broader_exclusive": broader}

    workers = [
        pipeline_module._assign_mixture_sources(
            sources, worker_index=worker_index, worker_count=4
        )
        for worker_index in range(4)
    ]

    for assigned in workers:
        assert len(assigned["strict"]) == 2
        assert len(assigned["broader_exclusive"]) == 2
    for source in sources:
        observed = [reference for worker in workers for reference in worker[source]]
        assert len(observed) == len(set((reference.shard, reference.index) for reference in observed))
        assert set(observed) == set(sources[source])


def test_mixture_worker_assignment_does_not_fallback_when_a_pool_is_too_small(tmp_path) -> None:
    strict = tuple(
        shard_module.SampleReference(tmp_path / "strict.npz", index, 0, 1, 32)
        for index in range(2)
    )
    broader = tuple(
        shard_module.SampleReference(tmp_path / "broader.npz", index, 0, 1, 32)
        for index in range(8)
    )

    assigned = pipeline_module._assign_mixture_sources(
        {"strict": strict, "broader_exclusive": broader},
        worker_index=3,
        worker_count=4,
    )

    assert assigned["strict"] == ()
    assert assigned["broader_exclusive"]
    assert not set(assigned["broader_exclusive"]) & set(strict)


@pytest.mark.parametrize(
    ("strict_probability", "strict_count"), ((0.5, 256), (0.75, 384))
)
def test_mixture_schedule_is_exact_and_explicit_offset_is_deterministic(
    strict_probability: float, strict_count: int
) -> None:
    full_schedule = pipeline_module._MixtureSchedule(
        strict_probability, seed=42
    )
    full = [full_schedule.pop() for _ in range(512)]
    resumed_schedule = pipeline_module._MixtureSchedule(
        strict_probability, offset=256, seed=42
    )
    resumed = [resumed_schedule.pop() for _ in range(256)]

    assert full.count("strict") == strict_count
    assert full.count("broader_exclusive") == 512 - strict_count
    assert resumed == full[256:]


def test_mixture_dataset_restart_does_not_infer_draw_offset_from_global_step(monkeypatch, tmp_path) -> None:
    references = [
        shard_module.SampleReference(tmp_path / f"shard-{index}.npz", index, 0, 1, 32)
        for index in range(4)
    ]

    def fake_index_shards(_metadata_path, **filters):
        return references[:2] if filters["loop_content_max"] == 0.4 else references

    offsets = []

    class SpySchedule(pipeline_module._MixtureSchedule):
        def __init__(self, strict_probability, *, offset=0, seed=0):
            offsets.append(offset)
            super().__init__(strict_probability, offset=offset, seed=seed)

    monkeypatch.setattr(pipeline_module, "index_shards", fake_index_shards)
    monkeypatch.setattr(pipeline_module, "_MixtureSchedule", SpySchedule)
    monkeypatch.setattr(
        pipeline_module,
        "load_sample",
        lambda *_args, **_kwargs: {
            "aatype": torch.zeros(32, dtype=torch.long),
            "chain_idx": torch.zeros(32, dtype=torch.long),
            "res_idx": torch.arange(32),
        },
    )
    monkeypatch.setattr(pipeline_module, "corrupt_structure", lambda clean, **_kwargs: clean)
    monkeypatch.setattr(pipeline_module, "sample_training_sigma", lambda *_args, **_kwargs: torch.tensor(1.0))
    class EmptyCache:
        hits = misses = resident_bytes = 0

        def __init__(self, _capacity):
            pass

        def preload(self, _shards):
            return None

        def __len__(self):
            return 0

    monkeypatch.setattr(pipeline_module, "ShardCache", EmptyCache)
    monkeypatch.setattr(
        pipeline_module,
        "collate_samples",
        lambda buffer, **_kwargs: {"lengths": torch.tensor([len(buffer[0]["aatype"])])},
    )
    data = DataConfig(
        metadata_path="/data/metadata.json",
        min_length=32,
        max_length=128,
        loop_content_max=0.4,
        batch_size=1,
        length_buckets=(64,),
        mixture=DataMixtureConfig(),
    )

    next(iter(pipeline_module.TrainingBatchDataset(data, DiffusionConfig(), sigma_data=1.0, global_step=123)))

    assert offsets == [0]


def test_index_excludes_sample_with_overlong_consecutive_ca_step(tmp_path) -> None:
    metadata = _ragged_fixture(tmp_path)
    shard = tmp_path / "shard.npz"
    with np.load(shard) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    # Sample 1 begins at global residue 5. Move one residue without changing
    # its chain or consecutive residue number.
    arrays["pos"][7, :, 0] += 10.0
    np.savez_compressed(shard, **arrays)
    contents = json.loads(metadata.read_text())
    contents[0].pop("ca_distance_validation_max")
    contents[0].pop("excluded_ca_distance_samples")
    metadata.write_text(json.dumps(contents))

    validated_metadata = tmp_path / "metadata_ca4.json"
    materialize_metadata(metadata, validated_metadata)
    references = index_shards(validated_metadata, min_length=1)
    assert [reference.index for reference in references] == [0, 2]


def test_ca_filter_is_strict_and_ignores_topological_discontinuities(tmp_path) -> None:
    metadata = _ragged_fixture(tmp_path)
    shard = tmp_path / "shard.npz"
    with np.load(shard) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    # Exactly 4 Angstrom is retained.
    arrays["pos"][5:12, :, 0] = np.arange(7)[:, None] * 4.0
    # A large spatial jump at the existing chain boundary in sample 0 is valid.
    arrays["pos"][3:5, :, 0] += 100.0
    # A large jump across a residue-number gap is also valid.
    arrays["res_idx"][7:12] += 100
    arrays["pos"][7:12, :, 0] += 200.0
    np.savez_compressed(shard, **arrays)
    contents = json.loads(metadata.read_text())
    contents[0].pop("ca_distance_validation_max")
    contents[0].pop("excluded_ca_distance_samples")
    metadata.write_text(json.dumps(contents))

    validated_metadata = tmp_path / "metadata_ca4.json"
    materialize_metadata(metadata, validated_metadata)
    references = index_shards(validated_metadata, min_length=1)
    assert [reference.index for reference in references] == [0, 1, 2]


def test_index_uses_precomputed_ca_distance_exclusions(tmp_path, monkeypatch) -> None:
    metadata = _ragged_fixture(tmp_path)
    contents = json.loads(metadata.read_text())
    contents[0].update(
        {
            "ca_distance_validation_max": 4.0,
            "excluded_ca_distance_samples": [1],
        }
    )
    metadata.write_text(json.dumps(contents))

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("precomputed exclusions must avoid coordinate scanning")

    monkeypatch.setattr(shard_module, "invalid_ca_distance_samples", unexpected_scan)
    references = index_shards(metadata, min_length=1)
    assert [reference.index for reference in references] == [0, 2]


def test_materializer_reuses_existing_ca_validation(
    tmp_path,
    monkeypatch,
) -> None:
    metadata = _ragged_fixture(tmp_path)
    contents = json.loads(metadata.read_text())
    contents[0]["excluded_ca_distance_samples"] = [1]
    metadata.write_text(json.dumps(contents))

    def unexpected_coordinate_scan(*_args, **_kwargs):
        raise AssertionError("prevalidated CA exclusions must not be recomputed")

    monkeypatch.setattr(
        "scripts.materialize_ca_distance_exclusions.invalid_ca_distance_samples",
        unexpected_coordinate_scan,
    )
    output = tmp_path / "metadata_ca4_copy.json"
    summary = materialize_metadata(metadata, output)
    annotated = json.loads(output.read_text())
    assert annotated[0]["excluded_ca_distance_samples"] == [1]
    assert summary["reused_ca_validation_shards"] == 1


def test_index_requires_offline_ca_distance_validation(tmp_path) -> None:
    metadata = _ragged_fixture(tmp_path)
    contents = json.loads(metadata.read_text())
    contents[0].pop("ca_distance_validation_max")
    contents[0].pop("excluded_ca_distance_samples")
    metadata.write_text(json.dumps(contents))

    with pytest.raises(ValueError, match="lacks precomputed CA-distance validation"):
        index_shards(metadata, min_length=1)


def test_quality_filter_requires_named_conditioning_feature(tmp_path) -> None:
    metadata = _ragged_fixture(tmp_path)
    contents = json.loads(metadata.read_text())
    contents[0]["cond_feature_names"] = ["mean_plddt", "helix_content"]
    metadata.write_text(json.dumps(contents))

    with pytest.raises(ValueError, match="loop_content"):
        index_shards(metadata, loop_content_max=0.5)


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


def test_rank_and_worker_partition_owns_whole_shards_and_balances_samples(tmp_path) -> None:
    references = []
    template = index_shards(_ragged_fixture(tmp_path))[0]
    sample_index = 0
    largest_shard = 0
    for shard_index, count in enumerate((11, 9, 7, 5, 3, 2, 1)):
        largest_shard = max(largest_shard, count)
        shard = tmp_path / f"shard_{shard_index}.npz"
        for local_index in range(count):
            references.append(
                replace(
                    template,
                    shard=shard,
                    index=sample_index,
                    start=local_index,
                    stop=local_index + 1,
                    resolved_length=1,
                )
            )
            sample_index += 1

    partitions = [
        shard_references(references, rank=rank, world_size=2, worker_id=worker, num_workers=2)
        for rank in range(2)
        for worker in range(2)
    ]
    owners = {}
    for owner, partition in enumerate(partitions):
        for reference in partition:
            previous = owners.setdefault(reference.shard, owner)
            assert previous == owner
    assert len(owners) == 7
    assert sorted(reference.index for group in partitions for reference in group) == list(
        range(sample_index)
    )
    loads = [len(group) for group in partitions]
    assert max(loads) - min(loads) <= largest_shard


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


def test_corruption_preserves_secondary_structure_targets_and_inputs(tmp_path) -> None:
    clean = load_sample(index_shards(_ragged_fixture(tmp_path), min_length=4)[0])
    clean["secondary_structure"] = torch.tensor([0, 1, 2, 0, 1])
    clean["secondary_structure_input"] = torch.tensor([3, 1, 2, 3, 0])
    sample = corrupt_structure(
        clean,
        sigma=1.0,
        generator=torch.Generator().manual_seed(7),
        translation_std=0.0,
    )
    assert torch.equal(sample["secondary_structure"], clean["secondary_structure"])
    assert torch.equal(sample["secondary_structure_input"], clean["secondary_structure_input"])


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


def test_unbounded_worker_cache_preloads_disjoint_working_set(tmp_path, monkeypatch) -> None:
    references = index_shards(_ragged_fixture(tmp_path))
    second_shard = tmp_path / "second.npz"
    second_shard.write_bytes((tmp_path / "shard.npz").read_bytes())
    second_reference = replace(references[0], shard=second_shard)
    calls = 0
    original_load = np.load

    def counted_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    cache = ShardCache(None)
    cache.preload((references[0].shard, second_reference.shard))
    load_sample(references[0], cache=cache)
    load_sample(second_reference, cache=cache)

    assert calls == 2
    assert len(cache) == 2
    assert cache.misses == 2
    assert cache.hits == 2
    assert cache.resident_bytes > 0


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
    assert batch["patch_capacity"] == 4
    assert set(batch["lengths"].tolist()) <= {5, 7}
    for row, length in enumerate(batch["lengths"].tolist()):
        assert batch["residue_mask"][row, :length].all()
        assert not batch["residue_mask"][row, length:].any()
        assert batch["model_atom_mask"][row, :length].all()
        assert not batch["model_atom_mask"][row, length:].any()
        assert not (batch["coordinate_mask"][row] & ~batch["model_atom_mask"][row]).any()
        assert batch["aatype_input"][row, length:].eq(21).all()
    assert batch["data_owned_shard_count"].item() == 1
    assert batch["data_cached_shard_count"].item() == 1
    assert batch["data_cache_miss_count"].item() == 1
    assert batch["data_cache_hit_count"].item() >= 2


def test_collation_checks_index_derived_patch_capacity(tmp_path) -> None:
    metadata = _ragged_fixture(tmp_path)
    data = DataConfig(
        metadata_path=str(metadata),
        min_length=4,
        max_length=8,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        length_buckets=(8,),
        patch_capacities=(4,),
    )
    loader = build_train_dataloader(data, DiffusionConfig(), sigma_data=16.0)
    assert next(iter(loader))["patch_capacity"] == 4

    shard = tmp_path / "shard.npz"
    with np.load(shard) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["chain_idx"][5:12] = np.arange(7) % 2
    np.savez_compressed(shard, **arrays)
    with pytest.raises(ValueError, match="calibrate data.patch_capacities"):
        next(iter(build_train_dataloader(data, DiffusionConfig(), sigma_data=16.0)))
