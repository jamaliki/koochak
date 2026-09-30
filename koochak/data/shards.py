"""Immutable dataset shards: build them, index them, deal them to workers.

A dataset is a set of large, write-once shard objects plus one small index
written last.  The index lists every shard's key, size, SHA256 and record
count, so training never lists storage and can verify every byte it reads;
the index's own SHA256 identifies the dataset version.

``plan_shards`` deals whole shards to every data-loading worker in the job
before anything is read, so each shard is read by exactly one worker per
epoch.  It is a pure function of its arguments: every rank and DataLoader
worker computes the same plan without coordinating.
"""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import re
import tarfile
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Union

from ..storage.store import Store, validate_key

__all__ = [
    "DEFAULT_TARGET_BYTES",
    "INDEX_KIND",
    "INDEX_VERSION",
    "TAR",
    "Shard",
    "ShardFormat",
    "ShardIndex",
    "ShardWriter",
    "assign_shards",
    "decode_tar_samples",
    "describe_shard",
    "encode_tar_sample",
    "load_index",
    "plan_shards",
    "read_shard",
    "worker_coordinates",
    "write_index",
]

INDEX_KIND = "koochak.shard_index"
INDEX_VERSION = 1
DEFAULT_TARGET_BYTES = 256 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
_READ_CHUNK = 8 * 1024 * 1024
_TAR_BLOCK = 512


def _require_int(value: object, label: str, *, minimum: Optional[int] = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        bound = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{label} must be an integer{bound}, got {value!r}")
    return value


def _segment(value: object, label: str) -> str:
    validate_key(value, label)
    if "/" in value:  # type: ignore[operator]
        raise ValueError(f"{label} must be a single path segment: {value!r}")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Shard:
    """One immutable shard object and the facts needed to verify and plan it."""

    key: str
    size_bytes: int
    sha256: str
    records: int

    def __post_init__(self) -> None:
        validate_key(self.key, "shard key")
        _require_int(self.size_bytes, "size_bytes", minimum=0)
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError(f"sha256 must be 64 lowercase hex characters: {self.sha256!r}")
        _require_int(self.records, "records", minimum=1)


@dataclass(frozen=True, slots=True)
class ShardIndex:
    """A committed dataset: its shards in index order and free-form metadata."""

    key: str
    shards: tuple[Shard, ...]
    metadata: Mapping[str, Any]
    sha256: str

    @property
    def records(self) -> int:
        return sum(shard.records for shard in self.shards)

    @property
    def size_bytes(self) -> int:
        return sum(shard.size_bytes for shard in self.shards)


def _unique_shards(shards: Sequence[Shard]) -> tuple[Shard, ...]:
    items = tuple(shards)
    if not items:
        raise ValueError("a dataset needs at least one shard")
    for shard in items:
        if not isinstance(shard, Shard):
            raise TypeError(f"expected Shard, got {type(shard).__name__}")
    keys = [shard.key for shard in items]
    if len(set(keys)) != len(keys):
        raise ValueError("shard keys must be unique")
    return items


def _relative_to(key: str, base: str) -> str:
    if not base:
        return key
    if not key.startswith(base + "/"):
        raise ValueError(f"shard {key!r} must live under the index directory {base!r}")
    return key[len(base) + 1 :]


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _encode_index(key: str, shards: tuple[Shard, ...], metadata: Mapping[str, Any]) -> bytes:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    base = posixpath.dirname(key)
    entries = [
        {
            "key": _relative_to(shard.key, base),
            "size_bytes": shard.size_bytes,
            "sha256": shard.sha256,
            "records": shard.records,
        }
        for shard in shards
    ]
    return _canonical_json(
        {
            "kind": INDEX_KIND,
            "v": INDEX_VERSION,
            "records": sum(shard.records for shard in shards),
            "size_bytes": sum(shard.size_bytes for shard in shards),
            "shards": entries,
            "metadata": dict(metadata),
        }
    )


def _decode_index(key: str, data: bytes) -> ShardIndex:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"shard index is not valid JSON: {key}") from exc
    expected = {"kind", "v", "records", "size_bytes", "shards", "metadata"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"shard index must contain exactly {sorted(expected)}: {key}")
    if value["kind"] != INDEX_KIND or value["v"] != INDEX_VERSION:
        raise ValueError(f"unsupported shard index kind/version in {key}")
    entries = value["shards"]
    if not isinstance(entries, list):
        raise ValueError(f"shard index 'shards' must be a list: {key}")
    base = posixpath.dirname(key)
    shards = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"key", "size_bytes", "sha256", "records"}:
            raise ValueError(f"invalid shard entry in {key}: {entry!r}")
        relative = validate_key(entry["key"], "shard key")
        shards.append(
            Shard(
                posixpath.join(base, relative) if base else relative,
                entry["size_bytes"],
                entry["sha256"],
                entry["records"],
            )
        )
    items = _unique_shards(shards)
    if value["records"] != sum(s.records for s in items) or value["size_bytes"] != sum(
        s.size_bytes for s in items
    ):
        raise ValueError(f"shard index totals do not match its entries: {key}")
    if not isinstance(value["metadata"], dict):
        raise ValueError(f"shard index metadata must be an object: {key}")
    return ShardIndex(
        key,
        items,
        MappingProxyType(value["metadata"]),
        hashlib.sha256(data).hexdigest(),
    )


def write_index(
    store: Store,
    key: str,
    shards: Sequence[Shard],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ShardIndex:
    """Commit a dataset by writing its index; identical re-commits are accepted.

    Shard keys are stored relative to the index's directory, so a dataset
    directory can be copied or staged elsewhere as one unit.
    """

    validate_key(key, "index key")
    items = _unique_shards(shards)
    data = _encode_index(key, items, metadata or {})
    try:
        store.put(key, data)
    except FileExistsError as exc:
        if store.get(key) != data:
            raise FileExistsError(f"a different shard index already exists at {key}") from exc
    return _decode_index(key, data)


def load_index(store: Store, key: str) -> ShardIndex:
    """Read and strictly validate a committed dataset index."""

    return _decode_index(validate_key(key, "index key"), store.get(key))


def _object_digest(store: Store, key: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with store.open(key) as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def describe_shard(store: Store, key: str, *, records: int) -> Shard:
    """Hash an existing object so it can be listed in an index."""

    size, digest = _object_digest(store, key)
    return Shard(key, size, digest, records)


def read_shard(store: Store, shard: Shard, *, verify: bool = True) -> bytes:
    """Read a whole shard, checking its size and (by default) its SHA256."""

    data = store.get(shard.key)
    if len(data) != shard.size_bytes:
        raise ValueError(
            f"shard {shard.key} has {len(data)} bytes, index says {shard.size_bytes}"
        )
    if verify and hashlib.sha256(data).hexdigest() != shard.sha256:
        raise ValueError(f"shard {shard.key} does not match its index SHA256")
    return data


# ---------------------------------------------------------------- formats


@dataclass(frozen=True, slots=True)
class ShardFormat:
    """How records become shard bytes and back.

    A shard is ``encode(r1) + encode(r2) + ... + trailer``, which lets a
    writer stream records without buffering decoded state; ``decode`` turns
    one whole shard back into its records, in order.
    """

    suffix: str
    encode: Callable[[Any], bytes]
    decode: Callable[[bytes], Iterator[Any]]
    trailer: bytes = b""

    def __post_init__(self) -> None:
        if self.suffix:
            _segment("x" + self.suffix, "suffix")
        if not isinstance(self.trailer, bytes):
            raise ValueError("trailer must be bytes")


def _sample_key(value: object) -> str:
    key = validate_key(value, "sample '__key__'")
    if "." in posixpath.basename(key):
        raise ValueError(f"sample '__key__' basename must not contain '.': {key!r}")
    return key


def encode_tar_sample(sample: Mapping[str, Any]) -> bytes:
    """Encode ``{"__key__": k, ext: bytes, ...}`` as tar members ``k.ext``.

    This is the WebDataset layout: members of one sample are adjacent and
    share the name before the first dot.  Headers carry no timestamps or
    owners, so equal samples always encode to equal bytes.
    """

    if not isinstance(sample, Mapping) or "__key__" not in sample:
        raise ValueError("a tar sample must be a mapping with a '__key__' entry")
    key = _sample_key(sample["__key__"])
    fields = sorted((name, value) for name, value in sample.items() if name != "__key__")
    if not fields:
        raise ValueError(f"sample {key!r} has no fields")
    out = bytearray()
    for name, value in fields:
        if not isinstance(name, str) or not name or "/" in name or "\x00" in name:
            raise ValueError(f"sample field names must be non-empty and contain no '/': {name!r}")
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(f"sample field {name!r} must be bytes, got {type(value).__name__}")
        payload = bytes(value)
        info = tarfile.TarInfo(f"{key}.{name}")
        info.size = len(payload)
        info.mtime = 0
        info.mode = 0o444
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        out += info.tobuf(format=tarfile.PAX_FORMAT, encoding="utf-8", errors="strict")
        out += payload
        out += b"\0" * (-len(payload) % _TAR_BLOCK)
    return bytes(out)


def decode_tar_samples(data: bytes) -> Iterator[dict[str, Any]]:
    """Yield ``{"__key__": k, ext: bytes, ...}`` samples from one tar shard."""

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        current: Optional[dict[str, Any]] = None
        for member in archive:
            if not member.isreg():
                raise ValueError(f"unsupported tar member (not a regular file): {member.name}")
            directory, base = posixpath.split(member.name)
            stem, dot, name = base.partition(".")
            if not stem or not dot or not name:
                raise ValueError(f"tar member name must look like 'key.ext': {member.name}")
            key = posixpath.join(directory, stem) if directory else stem
            handle = archive.extractfile(member)
            assert handle is not None  # regular members always have content
            if current is None or current["__key__"] != key:
                if current is not None:
                    yield current
                current = {"__key__": key}
            if name in current:
                raise ValueError(f"duplicate field {name!r} in sample {key!r}")
            current[name] = handle.read()
        if current is not None:
            yield current


TAR = ShardFormat(".tar", encode_tar_sample, decode_tar_samples, trailer=b"\0" * (2 * _TAR_BLOCK))


# ---------------------------------------------------------------- writer


class ShardWriter:
    """Pack records into shards of about ``target_bytes`` and commit an index last.

    Use as a context manager.  Shards are ``{prefix}/{name}-{n:06d}{suffix}``
    and the index is ``{prefix}/{index_name}``.  Leaving the ``with`` block by
    an exception writes no index, so a partial dataset is never committed.
    Encoding is deterministic, so rerunning an interrupted build accepts the
    identical shards it left behind and refuses any that differ.
    """

    def __init__(
        self,
        store: Store,
        prefix: str,
        *,
        format: ShardFormat = TAR,
        target_bytes: int = DEFAULT_TARGET_BYTES,
        name: str = "shard",
        index_name: str = "index.json",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if prefix != "":
            validate_key(prefix, "prefix")
        if not isinstance(format, ShardFormat):
            raise TypeError("format must be a ShardFormat")
        self.store = store
        self.prefix = prefix
        self.format = format
        self.target_bytes = _require_int(target_bytes, "target_bytes", minimum=1)
        self.name = _segment(name, "name")
        self.index_name = _segment(index_name, "index_name")
        self.metadata = dict(metadata or {})
        self.index: Optional[ShardIndex] = None
        self._shards: list[Shard] = []
        self._chunks: list[bytes] = []
        self._size = 0
        self._records = 0
        self._digest = hashlib.sha256()
        self._state = "open"

    def _key(self, leaf: str) -> str:
        return f"{self.prefix}/{leaf}" if self.prefix else leaf

    def _require_open(self) -> None:
        if self._state != "open":
            raise RuntimeError(f"ShardWriter is {self._state}")

    def add(self, record: Any) -> None:
        """Append one record, starting a new shard once the target size is reached."""

        self._require_open()
        encoded = self.format.encode(record)
        if not isinstance(encoded, bytes):
            raise TypeError(f"format.encode must return bytes, got {type(encoded).__name__}")
        self._chunks.append(encoded)
        self._digest.update(encoded)
        self._size += len(encoded)
        self._records += 1
        if self._size + len(self.format.trailer) >= self.target_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._records:
            return
        chunks = [*self._chunks, self.format.trailer]
        self._digest.update(self.format.trailer)
        shard = Shard(
            self._key(f"{self.name}-{len(self._shards):06d}{self.format.suffix}"),
            self._size + len(self.format.trailer),
            self._digest.hexdigest(),
            self._records,
        )
        try:
            self.store.put(shard.key, chunks)
        except FileExistsError as exc:
            if _object_digest(self.store, shard.key) != (shard.size_bytes, shard.sha256):
                raise FileExistsError(
                    f"{shard.key} already exists with different content; delete the "
                    "uncommitted shards or choose a new prefix"
                ) from exc
        self._shards.append(shard)
        self._chunks = []
        self._size = 0
        self._records = 0
        self._digest = hashlib.sha256()

    def close(self) -> ShardIndex:
        """Write the last shard and the index; return the committed dataset."""

        if self._state == "closed":
            assert self.index is not None
            return self.index
        self._require_open()
        try:
            self._flush()
            if not self._shards:
                raise ValueError("no records were added; refusing to commit an empty dataset")
            self.index = write_index(
                self.store, self._key(self.index_name), self._shards, metadata=self.metadata
            )
        except BaseException:
            self._state = "aborted"
            raise
        self._state = "closed"
        return self.index

    def abort(self) -> None:
        """Stop without committing; shards already written stay uncommitted."""

        if self._state == "open":
            self._state = "aborted"
            self._chunks = []

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


# ---------------------------------------------------------------- planning

ShardsLike = Union[ShardIndex, Sequence[Shard]]


def _shuffle_rank(seed: int, epoch: int, key: str) -> bytes:
    return hashlib.sha256(f"{seed}\x00{epoch}\x00{key}".encode("utf-8")).digest()


def plan_shards(
    shards: ShardsLike,
    *,
    num_workers: int,
    epoch: int = 0,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[tuple[Shard, ...], ...]:
    """Deal every shard to exactly one of ``num_workers`` workers.

    Shards are ordered by a hash of ``(seed, epoch, key)`` (index order when
    ``shuffle=False``), which is reproducible across machines and Python
    versions.  Each worker then owns one contiguous run of that order holding
    about ``1/num_workers`` of the records.  If any worker would own nothing,
    every caller raises the same ``ValueError`` rather than silently skewing.
    """

    _require_int(num_workers, "num_workers", minimum=1)
    _require_int(epoch, "epoch", minimum=0)
    _require_int(seed, "seed")
    if not isinstance(shuffle, bool):
        raise ValueError("shuffle must be a boolean")
    items = _unique_shards(shards.shards if isinstance(shards, ShardIndex) else shards)
    if len(items) < num_workers:
        raise ValueError(
            f"{len(items)} shards cannot feed {num_workers} workers; write smaller shards "
            "or use fewer workers"
        )
    ordered = sorted(items, key=lambda s: _shuffle_rank(seed, epoch, s.key)) if shuffle else items
    total = sum(shard.records for shard in ordered)
    owners: list[list[Shard]] = [[] for _ in range(num_workers)]
    before = 0
    for shard in ordered:
        # The worker whose record range contains this shard's midpoint, in
        # exact integer arithmetic; the result is always < num_workers.
        owners[(2 * before + shard.records) * num_workers // (2 * total)].append(shard)
        before += shard.records
    idle = [worker for worker, owned in enumerate(owners) if not owned]
    if idle:
        raise ValueError(
            f"shard record counts are too uneven to give all {num_workers} workers a shard "
            f"(idle workers include {idle[:8]}); write smaller shards or use fewer workers"
        )
    return tuple(tuple(owned) for owned in owners)


def assign_shards(
    shards: ShardsLike,
    *,
    worker: int,
    num_workers: int,
    epoch: int = 0,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[Shard, ...]:
    """Return the shards ``worker`` owns in ``plan_shards(...)``."""

    _require_int(num_workers, "num_workers", minimum=1)
    if type(worker) is not int or not 0 <= worker < num_workers:
        raise ValueError(f"worker must be in [0, {num_workers}), got {worker!r}")
    return plan_shards(shards, num_workers=num_workers, epoch=epoch, seed=seed, shuffle=shuffle)[
        worker
    ]


def worker_coordinates(*, rank: int, world_size: int) -> tuple[int, int]:
    """Return ``(worker, num_workers)`` over every rank and DataLoader worker.

    Call it inside ``IterableDataset.__iter__``.  Pass the rank and world size
    captured when the dataset was built in the main process: DataLoader
    worker processes may not have an initialized process group.
    """

    _require_int(world_size, "world_size", minimum=1)
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank!r}")
    import torch.utils.data

    info = torch.utils.data.get_worker_info()
    local, per_rank = (0, 1) if info is None else (info.id, info.num_workers)
    return rank * per_rank + local, world_size * per_rank
