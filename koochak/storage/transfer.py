"""Move objects between stores: parallel, verified, and resumable.

``copy_objects`` is the one place Koochak moves bytes between stores; archiving,
pulling, staging, and checkpoint replication build on it.  Concurrency comes
from the stores' profiles unless given explicitly:

- ``streams`` items are copied at once;
- an item larger than ``part_bytes`` is read as ``range_streams`` parallel
  byte ranges (in order, at most ``range_streams`` ranges ahead) and written as
  one sequential, create-only ``put``.

Peak memory is about ``streams * range_streams * part_bytes``.

A target that already exists with the expected size is skipped, so rerunning
an interrupted transfer resumes it; the write-once ``put`` protocol means an
existing object is complete.  A target with a different size is refused.
Bytes are hashed in flight: when an item carries an expected SHA256, a
just-written object that does not match is removed and the copy fails.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Sequence

from .store import Store, profile_of, validate_key

__all__ = ["TransferItem", "TransferReport", "copy_objects"]

ProgressFn = Callable[[int, int, int], None]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TransferItem:
    """Copy ``source`` to ``target``; ``size``/``sha256`` come from a manifest when known."""

    source: str
    target: str
    size: Optional[int] = None
    sha256: Optional[str] = None

    def __post_init__(self) -> None:
        validate_key(self.source, "source key")
        validate_key(self.target, "target key")
        if self.size is not None and (type(self.size) is not int or self.size < 0):
            raise ValueError(f"size must be a non-negative integer or None: {self.size!r}")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256)
        ):
            raise ValueError(f"sha256 must be 64 lowercase hex characters: {self.sha256!r}")


@dataclass(frozen=True, slots=True)
class TransferReport:
    copied: int
    skipped: int
    bytes_copied: int
    seconds: float

    @property
    def mb_s(self) -> float:
        return self.bytes_copied / max(self.seconds, 1e-9) / 1e6


def _streamed(source: Store, key: str) -> Iterator[bytes]:
    with source.open(key) as handle:
        while chunk := handle.read(_CHUNK):
            yield chunk


def _ranged(source: Store, key: str, size: int, range_streams: int, part_bytes: int) -> Iterator[bytes]:
    offsets = iter(range(0, size, part_bytes))
    with ThreadPoolExecutor(max_workers=range_streams) as pool:
        pending: deque[tuple[int, Future[bytes]]] = deque()

        def submit_next() -> None:
            offset = next(offsets, None)
            if offset is not None:
                length = min(part_bytes, size - offset)
                pending.append((length, pool.submit(source.get, key, offset, length)))

        for _ in range(range_streams):
            submit_next()
        while pending:
            length, future = pending.popleft()
            data = future.result()
            if len(data) != length:
                raise ValueError(f"{key} is shorter than its expected {size} bytes")
            submit_next()
            yield data


def _copy_one(
    source: Store,
    target: Store,
    item: TransferItem,
    range_streams: int,
    part_bytes: int,
) -> tuple[bool, int]:
    size = item.size
    if size is None:
        info = source.stat(item.source)
        if info is None:
            raise FileNotFoundError(f"source object does not exist: {item.source}")
        size = info.size
    existing = target.stat(item.target)
    if existing is not None:
        if existing.size != size:
            raise FileExistsError(
                f"target {item.target} already exists with {existing.size} bytes, expected {size}"
            )
        return False, 0

    if range_streams > 1 and size > part_bytes:
        chunks = _ranged(source, item.source, size, range_streams, part_bytes)
    else:
        chunks = _streamed(source, item.source)
    digest = hashlib.sha256()
    copied = 0

    def hashed() -> Iterator[bytes]:
        nonlocal copied
        for chunk in chunks:
            digest.update(chunk)
            copied += len(chunk)
            yield chunk

    target.put(item.target, hashed())
    problem = None
    if copied != size:
        problem = f"copied {copied} bytes, expected {size}"
    elif item.sha256 is not None and digest.hexdigest() != item.sha256:
        problem = "SHA256 does not match the expected digest"
    if problem is not None:
        # This call created the target, so it may remove the bad object.
        target.delete(item.target)
        raise ValueError(f"{item.source} -> {item.target}: {problem}")
    return True, copied


def copy_objects(
    source: Store,
    target: Store,
    items: Iterable[TransferItem],
    *,
    streams: Optional[int] = None,
    range_streams: Optional[int] = None,
    part_bytes: Optional[int] = None,
    progress: Optional[ProgressFn] = None,
) -> TransferReport:
    """Copy every item from ``source`` to ``target``; see the module docstring.

    ``progress(items_done, items_total, bytes_copied)`` is called from the
    calling thread after each item.  The first failure stops new work, waits
    for items already running, and is raised.
    """

    work: Sequence[TransferItem] = tuple(items)
    for item in work:
        if not isinstance(item, TransferItem):
            raise TypeError(f"expected TransferItem, got {type(item).__name__}")
    targets = [item.target for item in work]
    if len(set(targets)) != len(targets):
        raise ValueError("transfer items must have unique target keys")
    source_profile = profile_of(source)
    streams = max(source_profile.streams, profile_of(target).streams) if streams is None else streams
    range_streams = source_profile.range_streams if range_streams is None else range_streams
    part_bytes = source_profile.part_bytes if part_bytes is None else part_bytes
    for name, value in (("streams", streams), ("range_streams", range_streams), ("part_bytes", part_bytes)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    start = time.perf_counter()
    copied = skipped = bytes_copied = 0
    first_error: Optional[BaseException] = None
    remaining = iter(work)
    running: set[Future[tuple[bool, int]]] = set()
    with ThreadPoolExecutor(max_workers=streams) as pool:

        def fill() -> None:
            while first_error is None and len(running) < 2 * streams:
                item = next(remaining, None)
                if item is None:
                    return
                running.add(pool.submit(_copy_one, source, target, item, range_streams, part_bytes))

        fill()
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                running.discard(future)
                try:
                    was_copied, size = future.result()
                except BaseException as exc:
                    # Keep the first failure; let items already running finish.
                    if first_error is None:
                        first_error = exc
                    continue
                if was_copied:
                    copied += 1
                    bytes_copied += size
                else:
                    skipped += 1
                if progress is not None:
                    progress(copied + skipped, len(work), bytes_copied)
            fill()
    if first_error is not None:
        raise first_error
    return TransferReport(copied, skipped, bytes_copied, time.perf_counter() - start)
