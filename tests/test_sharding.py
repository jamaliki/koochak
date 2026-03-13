from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "koochak" / "data" / "sharding.py"
SPEC = importlib.util.spec_from_file_location("_koochak_data_sharding", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
_sharding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_sharding)

is_sharded = _sharding.is_sharded
mark_sharded = _sharding.mark_sharded
shard_dataset = _sharding.shard_dataset
shard_iterable_dataset = _sharding.shard_iterable_dataset
shard_map_dataset = _sharding.shard_map_dataset
warn_if_unsharded = _sharding.warn_if_unsharded


def test_shard_iterable_dataset_disjoint_and_complete():
    data = list(range(30))
    world = 3
    shards = [list(shard_iterable_dataset(data, r, world)) for r in range(world)]
    # Disjoint
    assert len(set(shards[0]).intersection(shards[1])) == 0
    assert len(set(shards[0]).intersection(shards[2])) == 0
    assert len(set(shards[1]).intersection(shards[2])) == 0
    # Complete
    combined = sorted([x for s in shards for x in s])
    assert combined == data


def test_shard_iterable_dataset_idempotent():
    data = list(range(10))
    sharded = shard_iterable_dataset(data, 0, 2)
    again = shard_iterable_dataset(sharded, 0, 2)
    assert sharded is again


def test_shard_map_dataset_len_and_indexing():
    data = list(range(10))
    rank = 1
    world = 3
    sharded = shard_map_dataset(data, rank, world)
    expected_len = max(0, math.ceil((len(data) - rank) / world))
    assert len(sharded) == expected_len
    expected_items = [data[i * world + rank] for i in range(expected_len)]
    assert [sharded[i] for i in range(expected_len)] == expected_items
    with pytest.raises(IndexError):
        _ = sharded[expected_len]


def test_shard_map_dataset_idempotent():
    data = list(range(12))
    sharded = shard_map_dataset(data, 0, 4)
    again = shard_map_dataset(sharded, 0, 4)
    assert sharded is again


def test_shard_dataset_mode_dispatch_and_marker():
    data = list(range(6))
    sharded = shard_dataset(data, rank=0, world_size=2, mode="map")
    assert is_sharded(sharded)
    with pytest.raises(ValueError):
        shard_dataset(data, rank=0, world_size=2, mode="unknown")


def test_warn_if_unsharded_emits_warning():
    data = [1, 2, 3]
    with pytest.warns(UserWarning, match="not sharded"):
        warn_if_unsharded(data, enabled=True, name="train dataset")


def test_loader_is_sharded_when_dataset_is_marked():
    torch = pytest.importorskip("torch")

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return idx

    dataset = TinyDataset()
    mark_sharded(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    assert is_sharded(loader)
    assert shard_dataset(loader, rank=0, world_size=2, mode="iterable") is loader


def test_sharding_outer_loader_emits_warning():
    torch = pytest.importorskip("torch")

    class TinyIterable(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield from range(4)

    loader = torch.utils.data.DataLoader(TinyIterable(), batch_size=1, num_workers=0)

    with pytest.warns(UserWarning, match="outer DataLoader/iterable"):
        sharded = shard_dataset(loader, rank=0, world_size=2, mode="iterable")
    assert is_sharded(sharded)
