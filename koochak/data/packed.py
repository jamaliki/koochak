"""Stream the groups of a packed collection to data-loading workers.

``koochak.data archive --groups`` writes a collection whose packs hold whole
groups: every file a training record needs, stored contiguously under its
original path. This module reads such a collection the way object storage
likes to be read, with few large sequential requests:

- ``PackedGroups`` indexes the file table by group and pack and deals whole
  packs to the job's data-loading workers (``plan``), reusing
  ``plan_shards``'s deterministic, record-balanced dealing with groups as the
  records. Each group belongs to the worker owning the pack that holds its
  first file, so workers own disjoint groups.
- ``GroupStream`` is one worker's endless stream of ``LoadedGroup`` objects
  (a group's files as bytes, plus their file-table entries). Each pass orders
  the worker's packs by a seeded hash, fetches whole packs ahead in a
  background thread (bounded by ``prefetch``), verifies them against the
  manifest, and yields the selected groups of each window of ``window`` packs
  in a seeded shuffle. Passes follow one another inside the stream, so ranks
  never run out of data at different times; ``start`` skips groups for
  resumption without reading the skipped packs.
- ``PackCache`` keeps recently fetched packs in memory up to a byte budget and
  can be shared by several streams in one process, so data that is cycled
  more often than the rest is not fetched again.
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Optional, Sequence

from ..storage.collection import Collection, FileEntry, object_key
from ..storage.store import Store
from .shards import Shard, plan_shards

__all__ = ["GroupInfo", "GroupStream", "LoadedGroup", "PackCache", "PackedGroups"]


@dataclass(frozen=True, slots=True)
class GroupInfo:
    """Where one group's files live: its primary pack and any other packs or objects."""

    name: str
    files: tuple[FileEntry, ...]
    pack: Optional[int]
    size_bytes: int


@dataclass(frozen=True, slots=True)
class LoadedGroup:
    """One group's file contents, keyed by collection path."""

    name: str
    files: Mapping[str, bytes]
    entries: Mapping[str, FileEntry]


def _rank(seed: int, epoch: int, key: str) -> bytes:
    return hashlib.sha256(f"{seed}\x00{epoch}\x00{key}".encode("utf-8")).digest()


class PackedGroups:
    """Group and pack view of a ``packed`` collection."""

    def __init__(self, collection: Collection) -> None:
        if not isinstance(collection, Collection):
            raise TypeError("PackedGroups needs a Collection")
        if collection.layout != "packed":
            raise ValueError("PackedGroups needs a collection with the 'packed' layout")
        self.collection = collection
        members: dict[str, list[FileEntry]] = {}
        for entry in collection.files:
            members.setdefault(entry.group, []).append(entry)
        groups: dict[str, GroupInfo] = {}
        primary: dict[int, list[str]] = {}
        for name, entries in members.items():
            entries.sort(key=lambda item: (item.pack is None, item.pack or 0, item.offset or 0, item.path))
            pack = entries[0].pack
            groups[name] = GroupInfo(name, tuple(entries), pack, sum(item.size for item in entries))
            if pack is not None:
                primary.setdefault(pack, []).append(name)
        unpacked = sorted(name for name, info in groups.items() if info.pack is None)
        if unpacked:
            raise ValueError(
                f"{len(unpacked)} groups have no packed file (e.g. {unpacked[:3]}); archive with "
                "--object-bytes at least as large as --pack-bytes"
            )
        self.groups: Mapping[str, GroupInfo] = MappingProxyType(groups)
        # Groups whose first file is in each pack, in pack order.
        self.pack_groups: Mapping[int, tuple[str, ...]] = MappingProxyType(
            {
                pack: tuple(sorted(names, key=lambda name: groups[name].files[0].offset or 0))
                for pack, names in primary.items()
            }
        )

    @classmethod
    def load(cls, store: Store) -> "PackedGroups":
        from ..storage.collection import load_collection

        return cls(load_collection(store))

    def load_group(self, store: Store, name: str, *, verify: bool = True) -> LoadedGroup:
        """Read one group with one range request per pack it occupies.

        A group's files are contiguous in its primary pack, so a group that
        fits in one pack is a single request; standalone objects are read
        whole. Every file is checked against its file-table size and SHA256
        when ``verify`` is set.
        """

        info = self.groups[name]
        by_pack: dict[int, list[FileEntry]] = {}
        for entry in info.files:
            if entry.pack is not None:
                by_pack.setdefault(entry.pack, []).append(entry)
        files: dict[str, bytes] = {}
        for pack, entries in by_pack.items():
            begin = min(entry.offset or 0 for entry in entries)
            end = max((entry.offset or 0) + entry.size for entry in entries)
            span = store.get(self.collection.packs[pack].key, begin, end - begin)
            for entry in entries:
                start = (entry.offset or 0) - begin
                files[entry.path] = span[start : start + entry.size]
        for entry in info.files:
            if entry.pack is None:
                files[entry.path] = store.get(object_key(entry.path))
        for entry in info.files:
            data = files[entry.path]
            if len(data) != entry.size or (verify and hashlib.sha256(data).hexdigest() != entry.sha256):
                raise ValueError(f"{entry.path} does not match its file-table size and SHA256")
        return LoadedGroup(name, MappingProxyType(files), MappingProxyType({e.path: e for e in info.files}))

    def plan(
        self,
        *,
        num_owners: int,
        seed: int = 0,
        select: Optional[Callable[[str], bool]] = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Deal the packs holding selected groups to ``num_owners`` owners.

        Packs are balanced by how many selected groups they hold (see
        ``plan_shards``); a pack's groups all go to its owner. Plans for
        disjoint selections can be combined to balance several kinds of
        groups separately, provided no pack holds groups of two selections.
        """

        shards = []
        for pack, names in sorted(self.pack_groups.items()):
            count = sum(1 for name in names if select is None or select(name))
            if count:
                entry = self.collection.packs[pack]
                shards.append(Shard(entry.key, entry.size, entry.sha256, count))
        if not shards:
            raise ValueError("no pack holds a selected group")
        index = {entry.key: number for number, entry in enumerate(self.collection.packs)}
        owners = plan_shards(shards, num_workers=num_owners, seed=seed)
        return tuple(tuple(sorted(index[shard.key] for shard in owned)) for owned in owners)


@dataclass
class PackCache:
    """Recently fetched packs, kept up to ``max_bytes`` (least recently used first out).

    Thread-safe, so streams in one process and their prefetch threads can share it.
    """

    max_bytes: int
    hits: int = 0
    _packs: "OrderedDict[int, bytes]" = field(default_factory=OrderedDict)
    _bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if type(self.max_bytes) is not int or self.max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")

    def get(self, pack: int) -> Optional[bytes]:
        with self._lock:
            data = self._packs.get(pack)
            if data is not None:
                self._packs.move_to_end(pack)
                self.hits += 1
            return data

    def put(self, pack: int, data: bytes) -> None:
        with self._lock:
            if len(data) > self.max_bytes or pack in self._packs:
                return
            self._packs[pack] = data
            self._bytes += len(data)
            while self._bytes > self.max_bytes:
                _, evicted = self._packs.popitem(last=False)
                self._bytes -= len(evicted)


class GroupStream:
    """One worker's endless, windowed-shuffle stream of the groups in its packs.

    ``packs`` are this worker's pack numbers (``PackedGroups.plan(...)[owner]``)
    and ``select`` keeps a subset of their groups. Pass ``p`` orders the packs
    by ``sha256(seed, p, pack key)``; each run of ``window`` packs yields its
    selected groups shuffled by ``Random(sha256(seed, p, window))``. ``start``
    skips that many groups from the beginning of pass 0. ``retries`` re-reads
    a pack after an ``OSError`` that many times; a pack that does not match its
    manifest size and SHA256 always raises.
    """

    def __init__(
        self,
        store: Store,
        groups: PackedGroups,
        packs: Sequence[int],
        *,
        select: Optional[Callable[[str], bool]] = None,
        seed: int = 0,
        window: int = 2,
        prefetch: int = 2,
        start: int = 0,
        verify: bool = True,
        cache: Optional[PackCache] = None,
        retries: int = 0,
        retry_seconds: float = 5.0,
    ) -> None:
        for label, value, minimum in (("window", window, 1), ("prefetch", prefetch, 0), ("start", start, 0),
                                      ("retries", retries, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{label} must be an integer >= {minimum}")
        self.store = store
        self.groups = groups
        self.select = select
        self.seed = seed
        self.window = window
        self.prefetch = prefetch
        self.start = start
        self.verify = verify
        self.cache = cache
        self.retries = retries
        self.retry_seconds = retry_seconds
        self.selected = {
            pack: tuple(name for name in groups.pack_groups.get(pack, ()) if select is None or select(name))
            for pack in packs
        }
        self.packs = tuple(pack for pack in sorted(set(packs)) if self.selected[pack])
        if not self.packs:
            raise ValueError("this worker owns no selected group")
        self.groups_per_pass = sum(len(self.selected[pack]) for pack in self.packs)
        self.stats = {"packs_fetched": 0, "bytes_fetched": 0, "fetch_seconds": 0.0, "wait_seconds": 0.0,
                      "cache_hits": 0, "retries": 0}

    # ------------------------------------------------------------ reading

    def _fetch(self, pack: int) -> bytes:
        if self.cache is not None:
            cached = self.cache.get(pack)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return cached
        entry = self.groups.collection.packs[pack]
        attempt = 0
        while True:
            started = time.perf_counter()
            try:
                data = self.store.get(entry.key)
                break
            except OSError:
                if attempt >= self.retries:
                    raise
                attempt += 1
                self.stats["retries"] += 1
                time.sleep(self.retry_seconds)
        self.stats["fetch_seconds"] += time.perf_counter() - started
        if len(data) != entry.size or (self.verify and hashlib.sha256(data).hexdigest() != entry.sha256):
            raise ValueError(f"{entry.key} does not match its manifest size and SHA256")
        self.stats["packs_fetched"] += 1
        self.stats["bytes_fetched"] += len(data)
        if self.cache is not None:
            self.cache.put(pack, data)
        return data

    def _materialize(self, name: str, loaded: Mapping[int, bytes]) -> LoadedGroup:
        info = self.groups.groups[name]
        files: dict[str, bytes] = {}
        for entry in info.files:
            if entry.pack in loaded:
                assert entry.offset is not None
                data = loaded[entry.pack][entry.offset : entry.offset + entry.size]
            elif entry.pack is not None:
                key = self.groups.collection.packs[entry.pack].key
                data = self.store.get(key, entry.offset or 0, entry.size)
            else:
                data = self.store.get(object_key(entry.path))
            if self.verify and entry.pack not in loaded and hashlib.sha256(data).hexdigest() != entry.sha256:
                raise ValueError(f"{entry.path} does not match its file-table SHA256")
            files[entry.path] = data
        return LoadedGroup(name, MappingProxyType(files), MappingProxyType({e.path: e for e in info.files}))

    # ------------------------------------------------------------ ordering

    def _windows(self, epoch: int) -> list[tuple[int, ...]]:
        packs = self.groups.collection.packs
        ordered = sorted(self.packs, key=lambda pack: _rank(self.seed, epoch, packs[pack].key))
        return [tuple(ordered[i : i + self.window]) for i in range(0, len(ordered), self.window)]

    def _order(self, epoch: int, number: int, window: Sequence[int]) -> list[str]:
        names = [name for pack in window for name in self.selected[pack]]
        random.Random(_rank(self.seed, epoch, f"window-{number}")).shuffle(names)
        return names

    def _schedule(self) -> Iterator[tuple[int, int, tuple[int, ...], int]]:
        """Endless ``(epoch, window number, packs, groups to skip)``, after ``start``."""

        skip = self.start
        epoch = 0
        while True:
            for number, window in enumerate(self._windows(epoch)):
                count = sum(len(self.selected[pack]) for pack in window)
                if skip >= count:
                    skip -= count
                    continue
                yield epoch, number, window, skip
                skip = 0
            epoch += 1

    def __iter__(self) -> Iterator[LoadedGroup]:
        schedule = self._schedule()
        executor = ThreadPoolExecutor(max_workers=max(1, self.prefetch), thread_name_prefix="koochak-packs")
        upcoming: "deque[tuple[int, int, tuple[int, ...], int]]" = deque()
        inflight: "deque[tuple[int, Future[bytes]]]" = deque()

        def top_up(packs: int) -> None:
            # Schedule whole windows, across passes, until ``packs`` fetches are in flight.
            while len(inflight) < packs:
                item = next(schedule)
                upcoming.append(item)
                inflight.extend((pack, executor.submit(self._fetch, pack)) for pack in item[2])

        try:
            while True:
                top_up(1)
                epoch, number, window, skip = upcoming.popleft()
                top_up(len(window) + self.prefetch)
                loaded = {}
                for _ in window:
                    pack, future = inflight.popleft()
                    started = time.perf_counter()
                    loaded[pack] = future.result()
                    self.stats["wait_seconds"] += time.perf_counter() - started
                top_up(self.prefetch)
                for name in self._order(epoch, number, window)[skip:]:
                    yield self._materialize(name, loaded)
        finally:
            for _, future in inflight:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
