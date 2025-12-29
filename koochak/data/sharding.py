from __future__ import annotations

from typing import Any, Iterable, Iterator

__all__ = [
    "is_sharded",
    "mark_sharded",
    "shard_iterable_dataset",
    "shard_map_dataset",
    "shard_dataset",
    "warn_if_unsharded",
]

_KOOCHAK_SHARDED_ATTR = "__koochak_sharded__"


def is_sharded(obj: Any) -> bool:
    return bool(getattr(obj, _KOOCHAK_SHARDED_ATTR, False))


def mark_sharded(obj: Any) -> Any:
    try:
        setattr(obj, _KOOCHAK_SHARDED_ATTR, True)
    except Exception:
        pass
    return obj


class ShardedIterable:
    def __init__(self, iterable: Iterable[Any], rank: int, world_size: int):
        self._iterable = iterable
        self._rank = int(rank)
        self._world_size = int(world_size)
        mark_sharded(self)

    def __iter__(self) -> Iterator[Any]:
        if self._world_size <= 1:
            yield from self._iterable
            return
        for idx, item in enumerate(self._iterable):
            if (idx % self._world_size) == self._rank:
                yield item


class ShardedMapDataset:
    def __init__(self, dataset: Any, rank: int, world_size: int):
        self._dataset = dataset
        self._rank = int(rank)
        self._world_size = int(world_size)
        mark_sharded(self)

    def __len__(self) -> int:
        n = len(self._dataset)
        if self._world_size <= 1:
            return n
        return max(0, (n - self._rank + self._world_size - 1) // self._world_size)

    def __getitem__(self, idx: int) -> Any:
        if self._world_size <= 1:
            return self._dataset[idx]
        src_idx = idx * self._world_size + self._rank
        if src_idx >= len(self._dataset):
            raise IndexError("sharded dataset index out of range")
        return self._dataset[src_idx]


def shard_iterable_dataset(iterable: Iterable[Any], rank: int, world_size: int) -> Iterable[Any]:
    if is_sharded(iterable):
        return iterable
    return ShardedIterable(iterable, rank, world_size)


def shard_map_dataset(dataset: Any, rank: int, world_size: int) -> Any:
    if is_sharded(dataset):
        return dataset
    return ShardedMapDataset(dataset, rank, world_size)


def shard_dataset(dataset: Any, *, rank: int, world_size: int, mode: str) -> Any:
    if dataset is None:
        return None
    mode_key = str(mode).lower()
    if mode_key == "iterable":
        return shard_iterable_dataset(dataset, rank, world_size)
    if mode_key == "map":
        return shard_map_dataset(dataset, rank, world_size)
    raise ValueError(f"Invalid shard_dataset mode: {mode!r} (expected 'iterable' or 'map')")


def warn_if_unsharded(dataset: Any, *, enabled: bool, name: str = "dataset") -> None:
    if not enabled or dataset is None:
        return
    if not is_sharded(dataset):
        import warnings

        warnings.warn(
            f"[koochak][ddp] {name} is not sharded; each rank may replay the full data. "
            "Enable train.shard_dataset/train.shard_eval_dataset or shard explicitly via "
            "koochak.data.sharding.shard_dataset(...).",
            stacklevel=2,
        )
