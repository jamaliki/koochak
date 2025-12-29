from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Iterator

import torch

__all__ = ["to_device", "prefetch", "cycle", "take"]


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        items = [to_device(v, device) for v in batch]
        return tuple(items) if isinstance(batch, tuple) else items
    return batch


def _record_stream(batch: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(batch, torch.Tensor):
        batch.record_stream(stream)
        return
    if isinstance(batch, dict):
        for v in batch.values():
            _record_stream(v, stream)
        return
    if isinstance(batch, (list, tuple)):
        for v in batch:
            _record_stream(v, stream)


class CudaPrefetcher:
    def __init__(self, iterator: Iterator[Any], device: torch.device, prefetch: int = 2):
        self.iterator = iterator
        self.device = device
        self.prefetch = max(1, int(prefetch))
        self.stream = torch.cuda.Stream()
        self.queue = deque()
        self._done = False
        self._fill()

    def _prefetch_one(self) -> None:
        if self._done:
            return
        try:
            batch = next(self.iterator)
        except StopIteration:
            self._done = True
            return
        with torch.cuda.stream(self.stream):
            batch = to_device(batch, self.device)
        self.queue.append(batch)

    def _fill(self) -> None:
        while len(self.queue) < self.prefetch and not self._done:
            self._prefetch_one()

    def __iter__(self) -> "CudaPrefetcher":
        return self

    def __next__(self) -> Any:
        if not self.queue and self._done:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.queue.popleft()
        _record_stream(batch, torch.cuda.current_stream())
        self._prefetch_one()
        return batch


def prefetch(iterable: Iterator[Any], device: torch.device, prefetch: int = 2) -> Iterator[Any]:
    if getattr(device, "type", "cpu") != "cuda" or prefetch <= 0:
        return iterable
    return CudaPrefetcher(iterable, device=device, prefetch=prefetch)


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
