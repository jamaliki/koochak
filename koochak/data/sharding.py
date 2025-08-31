from __future__ import annotations

from typing import Any, Iterable, Iterator

__all__ = ["shard_iterable"]


def shard_iterable(iterable: Iterable[Any], rank: int, world_size: int) -> Iterator[Any]:
    """Yield every world_size-th item from iterable starting at index==rank.

    Works for any iterable (finite or infinite). This is a simple striding
    approach appropriate for IterableDataset-like streams.
    """
    if world_size <= 1:
        yield from iterable
        return
    for idx, item in enumerate(iterable):
        if (idx % world_size) == rank:
            yield item

