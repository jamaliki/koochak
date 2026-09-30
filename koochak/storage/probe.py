"""Measure the storage semantics and performance that Koochak stores rely on.

    python -m koochak.storage.probe /path/on/a/filesystem
    python -m koochak.storage.probe scheme://bucket/prefix --json

For a local path (including FUSE mounts) the probe first checks POSIX
behaviour: exclusive create, rename-over, hard links, symlinks, fsync,
in-place writes, whether an unclosed file is visible, and how long a closed
file takes to read back with its exact bytes.  From those it recommends
``LocalStore`` settings, then exercises the store API with them: small-object
latency, listing, and single- and multi-stream throughput.

Two optional phases model the real workloads:

- ``checkpoint_bytes``: write one checkpoint-sized payload as 1..N concurrent
  parts, then time until every part reads back with its exact SHA256.
- ``dataset_shards``: build a synthetic dataset with ``ShardWriter``, then read
  it with N spawned worker processes that each read, verify, and decode their
  ``assign_shards`` share, twice (cold, then warm).

Everything happens under a fresh ``koochak-probe-<id>`` prefix that is
removed afterwards unless ``keep=True``.  Run it where the workload runs, on
a compute node rather than a login node: the throughput phase alone moves
about ``large_bytes * max(streams)`` bytes in each direction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import struct
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from ..data import shards as shards_lib
from ..utils.sizes import parse_size
from .atomic import _fsync_directory
from .store import LocalStore, ObjectInfo, Store, open_store

__all__ = ["main", "probe"]

_MB = 1_000_000
_CHUNK = 8 * 1024 * 1024
_BLOCK = 16 * 1024 * 1024
# Spawned readers import Koochak before the first barrier; storage passes can be slow.
_BARRIER_TIMEOUT = 900.0
_READ_TIMEOUT = 4 * 3600.0


def _progress(message: str) -> None:
    """One timestamped line on stderr so long runs show which phase they are in."""

    print(f"[koochak.storage.probe {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _describe_error(exc: OSError) -> str:
    code = os.strerror(exc.errno) if exc.errno else ""
    name = type(exc).__name__
    return f"{name}({exc.errno}: {code})" if exc.errno else f"{name}: {exc}"


def _run_check(checks: Dict[str, Any], name: str, fn: Callable[[], Optional[Dict[str, Any]]]) -> None:
    start = time.perf_counter()
    try:
        detail = fn() or {}
    except OSError as exc:
        # Unsupported operations are the expected outcome on some mounts.
        checks[name] = {"ok": False, "error": _describe_error(exc)}
        return
    checks[name] = {"ok": True, **detail, "seconds": round(time.perf_counter() - start, 6)}


def _write_file(path: str, data: bytes, *, exclusive: bool = False) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _read_file(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _readback(path: str, data: bytes, settle_timeout: float) -> Dict[str, Any]:
    expected = hashlib.sha256(data).hexdigest()
    start = time.perf_counter()
    _write_file(path, data, exclusive=True)
    written = time.perf_counter()
    first = True
    first_matched = False
    while True:
        try:
            matched = hashlib.sha256(_read_file(path)).hexdigest() == expected
        except OSError:
            matched = False
        if first:
            first_matched, first = matched, False
        now = time.perf_counter()
        if matched or now - written >= settle_timeout:
            break
        time.sleep(0.05)
    return {
        "ok": matched,
        "bytes": len(data),
        "write_seconds": round(written - start, 6),
        "write_mb_s": round(len(data) / max(written - start, 1e-9) / _MB, 1),
        "first_read_matched": first_matched,
        "settle_seconds": round(now - written, 3),
    }


def _probe_posix(directory: str, small: bytes, large: bytes, settle_timeout: float) -> Dict[str, Any]:
    os.makedirs(directory)
    checks: Dict[str, Any] = {}
    path = lambda name: os.path.join(directory, name)  # noqa: E731

    def exclusive_create() -> Dict[str, Any]:
        _write_file(path("exclusive"), small, exclusive=True)
        try:
            _write_file(path("exclusive"), small, exclusive=True)
        except FileExistsError:
            return {}
        return {"ok": False, "error": "a second exclusive create succeeded"}

    def rename_replace() -> Dict[str, Any]:
        _write_file(path("rename-target"), b"old")
        _write_file(path("rename-source"), b"new")
        os.replace(path("rename-source"), path("rename-target"))
        if _read_file(path("rename-target")) != b"new":
            return {"ok": False, "error": "replaced file kept its old content"}
        return {}

    def hard_link() -> Dict[str, Any]:
        _write_file(path("link-source"), small)
        os.link(path("link-source"), path("link-target"))
        if _read_file(path("link-target")) != small:
            return {"ok": False, "error": "link target content differs"}
        return {}

    def symlink() -> Dict[str, Any]:
        os.symlink("exclusive", path("symlink"))
        if not os.path.islink(path("symlink")) or os.readlink(path("symlink")) != "exclusive":
            return {"ok": False, "error": "symlink did not round-trip"}
        return {}

    def file_fsync() -> None:
        descriptor = os.open(path("fsync"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(descriptor, small)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def overwrite_in_place() -> Dict[str, Any]:
        _write_file(path("in-place"), b"abc")
        with open(path("in-place"), "r+b") as handle:
            handle.write(b"X")
        if _read_file(path("in-place")) != b"Xbc":
            return {"ok": False, "error": "in-place write not visible"}
        return {}

    def append() -> Dict[str, Any]:
        _write_file(path("append"), b"abc")
        with open(path("append"), "ab") as handle:
            handle.write(b"def")
        if _read_file(path("append")) != b"abcdef":
            return {"ok": False, "error": "append not visible"}
        return {}

    def unclosed_visibility() -> Dict[str, Any]:
        descriptor = os.open(path("unclosed"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(descriptor, small)
            try:
                seen = len(_read_file(path("unclosed")))
            except FileNotFoundError:
                observed = "absent"
            except OSError as exc:
                observed = f"unreadable ({_describe_error(exc)})"
            else:
                observed = "complete" if seen == len(small) else ("empty" if seen == 0 else "partial")
        finally:
            os.close(descriptor)
        return {"observed": observed}

    _run_check(checks, "exclusive_create", exclusive_create)
    _run_check(checks, "rename_replace", rename_replace)
    _run_check(checks, "hard_link", hard_link)
    _run_check(checks, "symlink", symlink)
    _run_check(checks, "file_fsync", file_fsync)
    _run_check(checks, "directory_fsync", lambda: _fsync_directory(directory))
    _run_check(checks, "overwrite_in_place", overwrite_in_place)
    _run_check(checks, "append", append)
    _run_check(checks, "unclosed_visibility", unclosed_visibility)
    for label, data in (("readback_small", small), ("readback_large", large)):
        try:
            checks[label] = _readback(path(label), data, settle_timeout)
        except OSError as exc:
            checks[label] = {"ok": False, "error": _describe_error(exc)}
    return checks


def _recommend(posix: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if posix["hard_link"]["ok"]:
        publish = "link"
    elif posix["exclusive_create"]["ok"]:
        publish = "exclusive"
    else:
        return None
    readbacks = [posix["readback_small"], posix["readback_large"]]
    return {
        "publish": publish,
        "fsync": posix["file_fsync"]["ok"] and posix["directory_fsync"]["ok"],
        "verify_readback": not all(check.get("first_read_matched") for check in readbacks),
    }


def _percentiles(seconds: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(seconds)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]  # noqa: E731
    return {
        "p50_ms": round(pick(0.5) * 1e3, 3),
        "p95_ms": round(pick(0.95) * 1e3, 3),
        "max_ms": round(ordered[-1] * 1e3, 3),
    }


def _timed(fn: Callable[[], Any]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _throughput(total_bytes: int, seconds: float) -> float:
    return round(total_bytes / max(seconds, 1e-9) / _MB, 1)


def _probe_store(
    store: Store,
    prefix: str,
    *,
    small_count: int,
    small: bytes,
    large: bytes,
    streams: Sequence[int],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    key = f"{prefix}/semantics"
    info = store.put(key, small)
    observed = store.stat(key)
    semantics = {
        "put_info_matches": info.size == len(small),
        "get_matches": store.get(key) == small,
        "range_get_matches": store.get(key, 1, 3) == small[1:4],
        "stat_size_matches": observed is not None and observed.size == len(small),
        "listed_after_put": key in store.list(f"{prefix}/"),
    }
    try:
        store.put(key, small)
    except FileExistsError:
        semantics["create_only"] = True
    else:
        semantics["create_only"] = False
    store.delete(key)
    semantics["absent_after_delete"] = store.stat(key) is None
    semantics["ok"] = all(semantics.values())
    results["semantics"] = semantics

    small_keys = [f"{prefix}/small/{index:06d}" for index in range(small_count)]
    results["small_put"] = _percentiles([_timed(lambda k=k: store.put(k, small)) for k in small_keys])
    results["small_stat"] = _percentiles([_timed(lambda k=k: store.stat(k)) for k in small_keys])
    results["small_get"] = _percentiles([_timed(lambda k=k: store.get(k)) for k in small_keys])
    listed: List[str] = []
    list_seconds = _timed(lambda: listed.extend(store.list(f"{prefix}/small/")))
    results["small_list"] = {
        "keys": len(listed),
        "complete": listed == small_keys,
        "ms": round(list_seconds * 1e3, 3),
    }

    # Separate objects for each cold measurement: a read warms what it touches.
    count = max(streams)
    size = len(large)
    single_key = f"{prefix}/large/single"
    ranged_key = f"{prefix}/large/ranged"
    many = [f"{prefix}/large/{index:03d}" for index in range(count)]
    put_single = _timed(lambda: store.put(single_key, _unique(large, 0)))
    with ThreadPoolExecutor(max_workers=count + 1) as pool:
        put_many = _timed(
            lambda: list(
                pool.map(
                    lambda job: store.put(job[1], _unique(large, job[0])),
                    enumerate([ranged_key, *many], start=1),
                )
            )
        )
    for key in (single_key, ranged_key, *many):
        _drop_cached_pages(store, key)

    cold = _timed(lambda: store.get(single_key))
    warm = _timed(lambda: store.get(single_key))
    results["large_single_stream"] = {
        "bytes": size,
        "put_mb_s": _throughput(size, put_single),
        "cold_get_mb_s": _throughput(size, cold),
        "warm_get_mb_s": _throughput(size, warm),
    }
    part = -(-size // count)
    cold = _timed(lambda: _ranged_get(store, ranged_key, size, count, part))
    warm = _timed(lambda: _ranged_get(store, ranged_key, size, count, part))
    results["large_ranged"] = {
        "ranges": count,
        "part_bytes": part,
        "cold_get_mb_s": _throughput(size, cold),
        "warm_get_mb_s": _throughput(size, warm),
    }
    concurrent: Dict[str, Any] = {
        "put_streams": count + 1,
        "put_mb_s": _throughput(size * (count + 1), put_many),
    }
    with ThreadPoolExecutor(max_workers=count) as pool:
        seconds = _timed(lambda: list(pool.map(store.get, many)))
    concurrent[f"cold_get_{count}_streams_mb_s"] = _throughput(size * count, seconds)
    for streams_now in streams:
        with ThreadPoolExecutor(max_workers=streams_now) as pool:
            seconds = _timed(lambda: list(pool.map(store.get, many[:streams_now])))
        concurrent[f"warm_get_{streams_now}_streams_mb_s"] = _throughput(size * streams_now, seconds)
    results["large_concurrent"] = concurrent

    for created in [*small_keys, single_key, ranged_key, *many]:
        store.delete(created)
    return results


def _unique(data: bytes, tag: int) -> List[Any]:
    """``data`` with a distinct 16-byte header, as chunks, without copying it."""

    return [struct.pack("<QQ", 0x6B6F6F6368616B, tag), memoryview(data)[16:]]


def _ranged_get(store: Store, key: str, size: int, ranges: int, part: int) -> None:
    offsets = range(0, size, part)
    with ThreadPoolExecutor(max_workers=ranges) as pool:
        read = sum(pool.map(lambda offset: len(store.get(key, offset, min(part, size - offset))), offsets))
    if read != size:
        raise RuntimeError(f"ranged read of {key} returned {read} bytes, expected {size}")


def _drop_cached_pages(store: Store, key: str) -> bool:
    """Best effort: evict a local file's cached pages so the next read is cold."""

    advise = getattr(os, "posix_fadvise", None)
    path = store.local_path(key)
    if advise is None or path is None:
        return False
    descriptor = os.open(path, os.O_RDONLY)
    try:
        advise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        # Some filesystems reject the hint; the next read may then be warm.
        return False
    finally:
        os.close(descriptor)
    return True


def _recommend_profile(store_results: Dict[str, Any], streams: Sequence[int]) -> Dict[str, Any]:
    """A ``StoreProfile(**...)`` suggestion from the store-phase measurements."""

    count = max(streams)
    single = store_results["large_single_stream"]["cold_get_mb_s"]
    single_warm = store_results["large_single_stream"]["warm_get_mb_s"]
    ranged = store_results["large_ranged"]
    concurrent = store_results["large_concurrent"][f"cold_get_{count}_streams_mb_s"]
    listing = store_results["small_list"]
    # Ranges must speed up cold reads without slowing cached ones: on some
    # object-storage mounts each range pays an open/seek cost that caps them
    # far below a sequential read of cached data.
    ranges_scale = (
        count > 1
        and ranged["cold_get_mb_s"] >= 2 * single
        and ranged["warm_get_mb_s"] >= 0.8 * single_warm
    )
    return {
        "request_seconds": max(round(store_results["small_get"]["p50_ms"] / 1e3, 4), 0.0001),
        "stream_mb_s": max(single, 0.1),
        "streams": count if concurrent >= 2 * single else min(count, 4),
        "range_streams": count if ranges_scale else 1,
        "part_bytes": max(8 * 1024 * 1024, ranged["part_bytes"]) if ranges_scale else 64 * 1024 * 1024,
        "list_is_cheap": listing["ms"] / max(listing["keys"], 1) < 1.0,
    }


def _payload(size: int, block: bytes, tag: int, counter: int) -> bytearray:
    """Incompressible bytes, made unique by a ``(tag, counter)`` header."""

    chunk = bytearray(block[:size])
    header = struct.pack("<QQ", tag, counter)[:size]
    chunk[: len(header)] = header
    return chunk


def _payload_stream(total: int, block: bytes, tag: int) -> Iterator[bytearray]:
    sent = 0
    counter = 0
    while sent < total:
        size = min(len(block), total - sent)
        yield _payload(size, block, tag, counter)
        sent += size
        counter += 1


def _object_digest(store: Store, key: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with store.open(key) as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _settle(store: Store, info: ObjectInfo, timeout: float) -> tuple[bool, float, bool]:
    """Return ``(first_read_matched, seconds_until_match, matched)`` for a new object."""

    start = time.perf_counter()
    first: Optional[bool] = None
    while True:
        try:
            matched = _object_digest(store, info.key) == (info.size, info.sha256)
        except OSError:
            matched = False
        if first is None:
            first = matched
        elapsed = time.perf_counter() - start
        if matched or elapsed >= timeout:
            return first, elapsed, matched
        time.sleep(1.0)


def _probe_checkpoint(
    store: Store,
    prefix: str,
    *,
    total_bytes: int,
    parts_options: Sequence[int],
    block: bytes,
    settle_timeout: float,
) -> Dict[str, Any]:
    """Write one checkpoint-sized payload as 1..N concurrent parts, then read it back."""

    results: Dict[str, Any] = {}
    for parts in parts_options:
        _progress(f"checkpoint: {total_bytes} bytes as {parts} concurrent parts")
        sizes =[total_bytes // parts + (1 if i < total_bytes % parts else 0) for i in range(parts)]
        keys = [f"{prefix}/{parts}-parts/part-{i:03d}" for i in range(parts)]
        with ThreadPoolExecutor(max_workers=parts) as pool:
            start = time.perf_counter()
            infos = list(
                pool.map(
                    lambda job: store.put(job[0], _payload_stream(job[1], block, tag=job[2])),
                    zip(keys, sizes, range(parts)),
                )
            )
            put_seconds = time.perf_counter() - start
            for key in keys:
                _drop_cached_pages(store, key)
            start = time.perf_counter()
            settled = list(pool.map(lambda info: _settle(store, info, settle_timeout), infos))
            readback_seconds = time.perf_counter() - start
        results[f"{parts}_parts"] = {
            "ok": all(matched for _first, _seconds, matched in settled),
            "bytes": total_bytes,
            "put_mb_s": _throughput(total_bytes, put_seconds),
            "put_seconds": round(put_seconds, 3),
            "first_read_matched": all(first for first, _seconds, _matched in settled),
            "readback_mb_s": _throughput(total_bytes, readback_seconds),
            "max_settle_seconds": round(max(seconds for _first, seconds, _matched in settled), 3),
        }
        for key in keys:
            store.delete(key)
    return results


def _store_spec(store: Store, location: str) -> Dict[str, Any]:
    if isinstance(store, LocalStore):
        return {
            "root": store.root,
            "publish": store.publish,
            "fsync": store.fsync,
            "verify_readback": store.verify_readback,
            "settle_seconds": store.settle_seconds,
        }
    return {"location": location}


def _open_spec(spec: Dict[str, Any]) -> Store:
    return LocalStore(**spec) if "root" in spec else open_store(spec["location"])


def _dataset_reader(
    spec: Dict[str, Any],
    index_key: str,
    worker: int,
    num_workers: int,
    passes: int,
    barrier: Any,
    results: Any,
) -> None:
    """Read, verify, and decode one worker's planned shards once per pass."""

    try:
        store = _open_spec(spec)
        index = shards_lib.load_index(store, index_key)
        owned = shards_lib.assign_shards(index, worker=worker, num_workers=num_workers)
        for pass_number in range(passes):
            barrier.wait(timeout=_BARRIER_TIMEOUT)
            start = time.perf_counter()
            size = records = 0
            for shard in owned:
                data = shards_lib.read_shard(store, shard, verify=True)
                size += len(data)
                records += sum(1 for _sample in shards_lib.TAR.decode(data))
            results.put((pass_number, worker, size, records, time.perf_counter() - start, None))
    except BaseException as exc:
        # Hand the failure to the parent, which raises it; then die loudly too.
        results.put((-1, worker, 0, 0, 0.0, f"{type(exc).__name__}: {exc}"))
        raise


def _read_dataset(
    spec: Dict[str, Any],
    index: "shards_lib.ShardIndex",
    readers: int,
    passes: int,
) -> List[Dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(readers + 1)
    queue = context.Queue()
    processes = [
        context.Process(
            target=_dataset_reader,
            args=(spec, index.key, worker, readers, passes, barrier, queue),
            daemon=True,
        )
        for worker in range(readers)
    ]
    for process in processes:
        process.start()
    summary: List[Dict[str, Any]] = []
    try:
        for pass_number in range(passes):
            barrier.wait(timeout=_BARRIER_TIMEOUT)
            start = time.perf_counter()
            rows = [queue.get(timeout=_READ_TIMEOUT) for _ in range(readers)]
            wall = time.perf_counter() - start
            errors = [row[5] for row in rows if row[5]]
            if errors:
                raise RuntimeError(f"dataset readers failed: {errors}")
            size = sum(row[2] for row in rows)
            records = sum(row[3] for row in rows)
            if (size, records) != (index.size_bytes, index.records):
                raise RuntimeError(
                    f"readers read {size} bytes / {records} records, expected "
                    f"{index.size_bytes} / {index.records}"
                )
            elapsed = sorted(row[4] for row in rows)
            summary.append(
                {
                    "pass": pass_number + 1,
                    "mb_s": _throughput(size, wall),
                    "wall_seconds": round(wall, 3),
                    "fastest_reader_seconds": round(elapsed[0], 3),
                    "slowest_reader_seconds": round(elapsed[-1], 3),
                }
            )
    finally:
        barrier.abort()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
    return summary


def _probe_dataset(
    store: Store,
    spec: Dict[str, Any],
    prefix: str,
    *,
    shards: int,
    shard_bytes: int,
    record_bytes: int,
    readers_options: Sequence[int],
    block: bytes,
) -> Dict[str, Any]:
    """Build a synthetic shard dataset, then read it with N parallel worker processes."""

    encoded = len(shards_lib.encode_tar_sample({"__key__": "r000000000", "bin": block[:record_bytes]}))
    per_shard = max(1, shard_bytes // encoded)
    _progress(f"dataset: building {shards} shards of {per_shard} records")
    start = time.perf_counter()
    with shards_lib.ShardWriter(store, prefix, target_bytes=per_shard * encoded) as writer:
        for number in range(shards * per_shard):
            writer.add({"__key__": f"r{number:09d}", "bin": _payload(record_bytes, block, 0, number)})
    build_seconds = time.perf_counter() - start
    index = writer.index
    assert index is not None
    results: Dict[str, Any] = {
        "shards": len(index.shards),
        "records": index.records,
        "bytes": index.size_bytes,
        "build_mb_s": _throughput(index.size_bytes, build_seconds),
        "build_seconds": round(build_seconds, 3),
    }
    # Only the first reader count starts cold; later passes measure cached reads.
    for shard in index.shards:
        _drop_cached_pages(store, shard.key)
    for readers in readers_options:
        _progress(f"dataset: reading with {readers} worker processes")
        results[f"read_{readers}_workers"] = _read_dataset(spec, index, readers, passes=2)
    return results




def _positive_ints(values: Sequence[int], label: str) -> tuple[int, ...]:
    items = tuple(values)
    if not items or any(type(count) is not int or count < 1 for count in items):
        raise ValueError(f"{label} must be positive integers")
    return items


def probe(
    location: str,
    *,
    small_count: int = 64,
    small_bytes: int = 4096,
    large_bytes: int = 64 * 1024 * 1024,
    streams: Sequence[int] = (1, 8),
    settle_timeout: float = 60.0,
    checkpoint_bytes: int = 0,
    checkpoint_parts: Sequence[int] = (1, 4),
    dataset_shards: int = 0,
    shard_bytes: int = 64 * 1024 * 1024,
    record_bytes: int = 1024 * 1024,
    readers: Sequence[int] = (1, 4),
    keep: bool = False,
    stores_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Probe ``location`` and return a JSON-serializable report.

    ``checkpoint_bytes=0`` and ``dataset_shards=0`` skip those phases.
    """

    for label, value, minimum in (
        ("small_count", small_count, 4),
        ("small_bytes", small_bytes, 4),
        ("large_bytes", large_bytes, 64),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    streams = _positive_ints(streams, "streams")
    checkpoint_parts = _positive_ints(checkpoint_parts, "checkpoint_parts")
    readers = _positive_ints(readers, "readers")
    if settle_timeout < 0:
        raise ValueError("settle_timeout must be non-negative")
    if type(checkpoint_bytes) is not int or checkpoint_bytes < 0:
        raise ValueError("checkpoint_bytes must be a non-negative integer")
    if checkpoint_bytes and checkpoint_bytes < max(checkpoint_parts):
        raise ValueError("checkpoint_bytes must be at least max(checkpoint_parts)")
    if type(dataset_shards) is not int or dataset_shards < 0:
        raise ValueError("dataset_shards must be a non-negative integer")
    if dataset_shards:
        if dataset_shards < max(readers):
            raise ValueError("dataset_shards must be at least max(readers)")
        if type(record_bytes) is not int or record_bytes < 1024:
            raise ValueError("record_bytes must be an integer >= 1024")
        if type(shard_bytes) is not int or shard_bytes < record_bytes:
            raise ValueError("shard_bytes must be an integer >= record_bytes")

    started = time.perf_counter()
    run_id = f"koochak-probe-{uuid.uuid4().hex[:12]}"
    small = os.urandom(small_bytes)
    large = os.urandom(large_bytes)
    report: Dict[str, Any] = {
        "location": str(location),
        "run_id": run_id,
        "host": platform.node(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "settings": {
            "small_count": small_count,
            "small_bytes": small_bytes,
            "large_bytes": large_bytes,
            "streams": list(streams),
            "settle_timeout": settle_timeout,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_parts": list(checkpoint_parts),
            "dataset_shards": dataset_shards,
            "shard_bytes": shard_bytes,
            "record_bytes": record_bytes,
            "readers": list(readers),
            "page_cache_eviction": getattr(os, "posix_fadvise", None) is not None,
        },
    }
    target = open_store(location, stores_file=stores_file)
    local_root = target.root if isinstance(target, LocalStore) else None
    if local_root is not None and not os.path.isdir(local_root):
        raise FileNotFoundError(f"probe location is not an existing directory: {local_root}")
    try:
        store: Optional[Store] = target
        if local_root is not None:
            _progress(f"posix semantics under {local_root}/{run_id}")
            report["posix"] = _probe_posix(
                os.path.join(local_root, run_id, "posix"), small, large, settle_timeout
            )
            recommended = _recommend(report["posix"])
            report["recommended_local_store"] = recommended
            store = None
            if recommended is not None:
                store = LocalStore(
                    local_root,
                    **recommended,
                    settle_seconds=settle_timeout if recommended["verify_readback"] else 0.0,
                )
        if store is None:
            report["store"] = {"skipped": "no create-only publish mode works on this filesystem"}
        else:
            report["store_config"] = repr(store)
            _progress(f"store API with {store!r}")
            report["store"] = _probe_store(
                store,
                f"{run_id}/store",
                small_count=small_count,
                small=small,
                large=large,
                streams=streams,
            )
            report["recommended_profile"] = _recommend_profile(report["store"], streams)
            block = os.urandom(max(_BLOCK, record_bytes if dataset_shards else 0))
            if checkpoint_bytes:
                report["checkpoint"] = _probe_checkpoint(
                    store,
                    f"{run_id}/checkpoint",
                    total_bytes=checkpoint_bytes,
                    parts_options=checkpoint_parts,
                    block=block,
                    settle_timeout=settle_timeout,
                )
            if dataset_shards:
                report["dataset"] = _probe_dataset(
                    store,
                    _store_spec(store, str(location)),
                    f"{run_id}/dataset",
                    shards=dataset_shards,
                    shard_bytes=shard_bytes,
                    record_bytes=record_bytes,
                    readers_options=readers,
                    block=block,
                )
    finally:
        if not keep:
            if local_root is not None:
                if os.path.lexists(os.path.join(local_root, run_id)):
                    shutil.rmtree(os.path.join(local_root, run_id))
            else:
                for created in target.list(f"{run_id}/"):
                    target.delete(created)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    return report


def _parse_counts(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(","))


def _format(report: Dict[str, Any]) -> str:
    lines = [f"location  {report['location']}  host={report['host']}  cpus={report['cpus']}"]
    for check, result in report.get("posix", {}).items():
        status = "ok  " if result.get("ok") else "FAIL"
        detail = {k: v for k, v in result.items() if k not in {"ok", "seconds"}}
        lines.append(f"posix  {status}  {check:<20} {json.dumps(detail, sort_keys=True)}")
    if "recommended_local_store" in report:
        lines.append(f"recommended LocalStore settings: {report['recommended_local_store']}")
    if "recommended_profile" in report:
        lines.append(f"recommended StoreProfile: {report['recommended_profile']}")
    if "store_config" in report:
        lines.append(f"store  {report['store_config']}")
    for phase in ("store", "checkpoint", "dataset"):
        for section, values in report.get(phase, {}).items():
            lines.append(f"{phase:<10} {section:<20} {json.dumps(values, sort_keys=True)}")
    lines.append(f"elapsed  {report['elapsed_seconds']}s")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m koochak.storage.probe",
        description="Measure storage semantics and throughput for a Koochak store location.",
    )
    parser.add_argument("location", help="filesystem path, file:// URI, or registered scheme://")
    parser.add_argument("--small-count", type=int, default=64)
    parser.add_argument("--small-bytes", type=parse_size, default=4096)
    parser.add_argument("--large-bytes", type=parse_size, default=64 * 1024**2)
    parser.add_argument(
        "--streams",
        type=_parse_counts,
        default=(1, 8),
        help="comma-separated concurrent read stream counts (default: 1,8)",
    )
    parser.add_argument("--settle-timeout", type=float, default=60.0)
    parser.add_argument(
        "--checkpoint-bytes",
        type=parse_size,
        default=0,
        help="checkpoint-sized payload to write and read back (default: 0, skip)",
    )
    parser.add_argument(
        "--checkpoint-parts",
        type=_parse_counts,
        default=(1, 4),
        help="comma-separated concurrent part counts for the checkpoint payload",
    )
    parser.add_argument(
        "--dataset-shards",
        type=int,
        default=0,
        help="shards in the synthetic dataset (default: 0, skip)",
    )
    parser.add_argument("--shard-bytes", type=parse_size, default=64 * 1024**2)
    parser.add_argument("--record-bytes", type=parse_size, default=1024**2)
    parser.add_argument(
        "--readers",
        type=_parse_counts,
        default=(1, 4),
        help="comma-separated counts of parallel dataset reader processes",
    )
    parser.add_argument("--keep", action="store_true", help="leave probe files in place")
    parser.add_argument("--stores", default=None, help="stores file (default: $KOOCHAK_STORES)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print the full JSON report")
    output.add_argument(
        "--emit-profile",
        action="store_true",
        help="print only the recommended profile, as a stores-file 'profile:' block",
    )
    args = parser.parse_args(argv)
    report = probe(
        args.location,
        small_count=args.small_count,
        small_bytes=args.small_bytes,
        large_bytes=args.large_bytes,
        streams=args.streams,
        settle_timeout=args.settle_timeout,
        checkpoint_bytes=args.checkpoint_bytes,
        checkpoint_parts=args.checkpoint_parts,
        dataset_shards=args.dataset_shards,
        shard_bytes=args.shard_bytes,
        record_bytes=args.record_bytes,
        readers=args.readers,
        keep=args.keep,
        stores_file=args.stores,
    )
    if args.emit_profile:
        profile = report.get("recommended_profile")
        if profile is None:
            raise SystemExit("no profile: the store phase was skipped for this location")
        print("profile:")
        for key in sorted(profile):
            value = profile[key]
            print(f"  {key}: {str(value).lower() if isinstance(value, bool) else value}")
        return 0
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else _format(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
