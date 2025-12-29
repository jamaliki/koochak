from __future__ import annotations

import math

import pytest

from koochak.data.sharding import (
    is_sharded,
    shard_dataset,
    shard_iterable_dataset,
    shard_map_dataset,
    warn_if_unsharded,
)


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
