from __future__ import annotations

import queue
import threading
import time
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
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device(self.device.type, torch.cuda.current_device())
        self.prefetch = max(1, int(prefetch))
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream()
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=self.prefetch)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._sentinel = object()
        self._thread = threading.Thread(
            target=self._producer_loop,
            name="koochak-cuda-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _put(self, item: Any) -> bool:
        while not self._stop.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _producer_loop(self) -> None:
        try:
            while not self._stop.is_set():
                next_start = time.perf_counter()
                try:
                    batch = next(self.iterator)
                except StopIteration:
                    break
                next_return_time = time.perf_counter()
                producer_next_s = next_return_time - next_start
                next_overhead_s = 0.0
                next_gap_before_collect_s = 0.0
                if isinstance(batch, dict):
                    try:
                        batch_ready_time = float(batch.get("batch_ready_time_s", 0.0))
                        if batch_ready_time > 0.0:
                            next_overhead_s = max(0.0, next_return_time - batch_ready_time)
                    except Exception:
                        next_overhead_s = 0.0
                    try:
                        collect_start_time = float(
                            batch.get("batch_collect_start_time_s", 0.0)
                        )
                        if collect_start_time > 0.0:
                            next_gap_before_collect_s = max(
                                0.0,
                                collect_start_time - next_start,
                            )
                    except Exception:
                        next_gap_before_collect_s = 0.0

                transfer_start = time.perf_counter()
                with torch.cuda.device(self.device):
                    with torch.cuda.stream(self.stream):
                        batch = to_device(batch, self.device)
                        event = torch.cuda.Event()
                        event.record(self.stream)
                transfer_enqueue_s = time.perf_counter() - transfer_start

                timings = {
                    "prefetch_producer_next_s": float(producer_next_s),
                    "prefetch_next_overhead_s": float(next_overhead_s),
                    "prefetch_next_gap_before_collect_s": float(next_gap_before_collect_s),
                    "prefetch_transfer_enqueue_s": float(transfer_enqueue_s),
                }
                put_start = time.perf_counter()
                if not self._put((batch, event, timings)):
                    return
                timings["prefetch_queue_put_s"] = float(time.perf_counter() - put_start)
        except BaseException as exc:
            self._error = exc
        finally:
            self._put(self._sentinel)

    def __iter__(self) -> "CudaPrefetcher":
        return self

    def __next__(self) -> Any:
        queue_wait_start = time.perf_counter()
        item = self.queue.get()
        queue_wait_s = time.perf_counter() - queue_wait_start
        if item is self._sentinel:
            self.close()
            if self._error is not None:
                raise self._error
            raise StopIteration
        batch, event, timings = item
        with torch.cuda.device(self.device):
            current_stream = torch.cuda.current_stream()
        wait_event_start = time.perf_counter()
        current_stream.wait_event(event)
        wait_event_enqueue_s = time.perf_counter() - wait_event_start
        _record_stream(batch, current_stream)
        if isinstance(batch, dict):
            batch["prefetch_queue_wait_s"] = float(queue_wait_s)
            batch["prefetch_wait_event_enqueue_s"] = float(wait_event_enqueue_s)
            batch["prefetch_queue_depth"] = float(self.queue.qsize())
            try:
                ready_time = float(batch.get("batch_ready_time_s", 0.0))
                batch["prefetch_batch_ready_age_s"] = (
                    max(0.0, time.perf_counter() - ready_time)
                    if ready_time > 0.0
                    else 0.0
                )
            except Exception:
                batch["prefetch_batch_ready_age_s"] = 0.0
            for key, value in timings.items():
                batch[key] = float(value)
        return batch

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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
