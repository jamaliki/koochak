from __future__ import annotations

from koochak.data.sharding import shard_iterable


def test_shard_iterable_disjoint_and_complete():
    data = list(range(30))
    world = 3
    shards = [list(shard_iterable(data, r, world)) for r in range(world)]
    # Disjoint
    assert len(set(shards[0]).intersection(shards[1])) == 0
    assert len(set(shards[0]).intersection(shards[2])) == 0
    assert len(set(shards[1]).intersection(shards[2])) == 0
    # Complete
    combined = sorted([x for s in shards for x in s])
    assert combined == data

