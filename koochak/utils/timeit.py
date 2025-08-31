from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator

__all__ = ["Timer", "time_block"]


class Timer:
    """Simple scoped wall-clock timer.

    Usage:
        with Timer() as t:
            ...
        print(t.elapsed)
    """

    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._start
        return False


@contextmanager
def time_block(store: Dict[str, float], key: str) -> Iterator[None]:
    """Context manager that records elapsed time into `store[key]`."""
    start = time.perf_counter()
    try:
        yield None
    finally:
        store[key] = store.get(key, 0.0) + (time.perf_counter() - start)

