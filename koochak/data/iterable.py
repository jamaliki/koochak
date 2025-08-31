from __future__ import annotations

from typing import Any, Iterable, Iterator

import torch

__all__ = ["to_device", "cycle", "take"]


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        items = [to_device(v, device) for v in batch]
        return tuple(items) if isinstance(batch, tuple) else items
    return batch


def cycle(iterable: Iterable[Any]) -> Iterator[Any]:
    """Yield from `iterable` forever by cycling.

    If the iterable is finite, this will restart it repeatedly.
    """
    while True:
        for x in iterable:
            yield x


def take(iterable: Iterable[Any], n: int) -> Iterator[Any]:
    """Yield at most `n` items from iterable.

    Useful for quick eval loops or sampling a small subset without materializing.
    """
    if n <= 0:
        return
    it = iter(iterable)
    count = 0
    while count < n:
        try:
            yield next(it)
        except StopIteration:
            return
        count += 1
