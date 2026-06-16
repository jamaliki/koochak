from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, Iterable, Iterator

import torch

__all__ = ["to_device", "prefetch", "cycle", "take"]

BatchPrepareFn = Callable[[Any], Any]


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


@dataclass
class _PrefetchedBatch:
    batch: Any
    event: torch.cuda.Event


_STOP = object()


class CudaPrefetcher:
    def __init__(
        self,
        iterator: Iterator[Any],
        device: torch.device,
        prefetch: int = 2,
        *,
        prepare_fn: BatchPrepareFn | None = None,
        threaded: bool = False,
        pipeline: str = "single",
    ):
        self.iterator = iterator
        self.device = device
        self.prefetch = max(1, int(prefetch))
        self.prepare_fn = prepare_fn
        self.threaded = bool(threaded)
        self.pipeline = str(pipeline or "single").strip().lower().replace("-", "_")
        self.stream = torch.cuda.Stream()
        self._exception_lock = Lock()
        self._stop_event = Event()
        self.queue: deque[_PrefetchedBatch] | Queue[Any] = (
            Queue(maxsize=self.prefetch) if self.threaded else deque()
        )
        self.cpu_queue: Queue[Any] | None = None
        self._done = False
        self._closed = False
        self._thread_exception: BaseException | None = None
        if self.threaded:
            if self.pipeline in {"two_stage", "staged"}:
                self.cpu_queue = Queue(maxsize=self.prefetch)
                self._cpu_worker = Thread(target=self._cpu_worker_loop, daemon=True)
                self._cuda_worker = Thread(target=self._cuda_worker_loop, daemon=True)
                self._cpu_worker.start()
                self._cuda_worker.start()
            else:
                self._worker = Thread(target=self._worker_loop, daemon=True)
                self._worker.start()
        else:
            self._fill()

    def _prepare_on_stream(self, batch: Any) -> _PrefetchedBatch:
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            batch = to_device(batch, self.device)
            if self.prepare_fn is not None:
                prepared = self.prepare_fn(batch)
                if prepared is not None:
                    batch = prepared
            event = torch.cuda.Event()
            event.record(self.stream)
        return _PrefetchedBatch(batch=batch, event=event)

    def _prefetch_one(self) -> None:
        if self._done:
            return
        try:
            batch = next(self.iterator)
        except StopIteration:
            self._done = True
            return
        assert isinstance(self.queue, deque)
        self.queue.append(self._prepare_on_stream(batch))

    def _worker_loop(self) -> None:
        assert isinstance(self.queue, Queue)
        try:
            while not self._stop_event.is_set():
                try:
                    batch = next(self.iterator)
                except StopIteration:
                    self._queue_put(self.queue, _STOP)
                    return
                if not self._queue_put(self.queue, self._prepare_on_stream(batch)):
                    return
        except BaseException as exc:
            self._set_thread_exception(exc)
            self._queue_put(self.queue, _STOP)

    def _cpu_worker_loop(self) -> None:
        assert self.cpu_queue is not None
        try:
            while not self._stop_event.is_set():
                try:
                    batch = next(self.iterator)
                except StopIteration:
                    self._queue_put(self.cpu_queue, _STOP)
                    return
                if not self._queue_put(self.cpu_queue, batch):
                    return
        except BaseException as exc:
            self._set_thread_exception(exc)
            self._queue_put(self.cpu_queue, _STOP)

    def _cuda_worker_loop(self) -> None:
        assert self.cpu_queue is not None
        assert isinstance(self.queue, Queue)
        try:
            while not self._stop_event.is_set():
                batch = self._queue_get(self.cpu_queue)
                if batch is _STOP:
                    self._queue_put(self.queue, _STOP)
                    return
                if not self._queue_put(self.queue, self._prepare_on_stream(batch)):
                    return
        except BaseException as exc:
            self._set_thread_exception(exc)
            self._queue_put(self.queue, _STOP)

    def _set_thread_exception(self, exc: BaseException) -> None:
        with self._exception_lock:
            if self._thread_exception is None:
                self._thread_exception = exc

    def _queue_put(self, queue: Queue[Any], item: Any) -> bool:
        while not self._stop_event.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def _queue_get(self, queue: Queue[Any]) -> Any:
        while not self._stop_event.is_set():
            try:
                return queue.get(timeout=0.1)
            except Empty:
                continue
        return _STOP

    def _fill(self) -> None:
        assert isinstance(self.queue, deque)
        while len(self.queue) < self.prefetch and not self._done:
            self._prefetch_one()

    def __iter__(self) -> "CudaPrefetcher":
        return self

    def __next__(self) -> Any:
        if self.threaded:
            assert isinstance(self.queue, Queue)
            item = self._queue_get(self.queue)
            if item is _STOP:
                if self._thread_exception is not None:
                    raise self._thread_exception
                raise StopIteration
            assert isinstance(item, _PrefetchedBatch)
            return self._consume(item)
        assert isinstance(self.queue, deque)
        if not self.queue and self._done:
            raise StopIteration
        item = self.queue.popleft()
        batch = self._consume(item)
        self._prefetch_one()
        return batch

    def _consume(self, item: _PrefetchedBatch) -> Any:
        current_stream = torch.cuda.current_stream()
        current_stream.wait_event(item.event)
        _record_stream(item.batch, current_stream)
        return item.batch

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        for name in ("_cuda_worker", "_cpu_worker", "_worker"):
            worker = getattr(self, name, None)
            if worker is not None:
                worker.join(timeout=5.0)


def prefetch(
    iterable: Iterator[Any],
    device: torch.device,
    prefetch: int = 2,
    *,
    prepare_fn: BatchPrepareFn | None = None,
    threaded: bool = False,
    pipeline: str = "single",
) -> Iterator[Any]:
    if getattr(device, "type", "cpu") != "cuda" or prefetch <= 0:
        return iterable
    return CudaPrefetcher(
        iterable,
        device=device,
        prefetch=prefetch,
        prepare_fn=prepare_fn,
        threaded=threaded,
        pipeline=pipeline,
    )


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
