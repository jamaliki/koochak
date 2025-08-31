from __future__ import annotations

import collections
import time
from typing import Deque, Optional

__all__ = ["SmoothedMeter", "Throughput", "EMA"]


class SmoothedMeter:
    """Track a scalar with a fixed-size window and global stats."""

    def __init__(self, window_size: int = 100):
        self.window_size = int(window_size)
        self.deque: Deque[float] = collections.deque(maxlen=self.window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        v = float(value)
        self.deque.append(v)
        self.total += v * n
        self.count += n

    @property
    def median(self) -> float:
        if not self.deque:
            return 0.0
        d = sorted(self.deque)
        m = len(d) // 2
        if len(d) % 2 == 1:
            return d[m]
        return 0.5 * (d[m - 1] + d[m])

    @property
    def avg(self) -> float:
        if not self.deque:
            return 0.0
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self) -> float:
        return self.total / max(1, self.count)

    @property
    def max(self) -> float:
        return max(self.deque) if self.deque else 0.0

    @property
    def min(self) -> float:
        return min(self.deque) if self.deque else 0.0


class Throughput:
    """Compute items/sec over a sliding window using timestamps."""

    def __init__(self, window_size: int = 50):
        self.times: Deque[float] = collections.deque(maxlen=window_size)
        self.counts: Deque[int] = collections.deque(maxlen=window_size)

    def update(self, n_items: int):
        self.times.append(time.time())
        self.counts.append(int(n_items))

    @property
    def it_per_s(self) -> float:
        if len(self.times) < 2:
            return 0.0
        dt = self.times[-1] - self.times[0]
        if dt <= 0:
            return 0.0
        return sum(self.counts) / dt


class EMA:
    """Simple exponential moving average for scalars."""

    def __init__(self, alpha: float = 0.98):
        self.alpha = float(alpha)
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * x
        return self.value

