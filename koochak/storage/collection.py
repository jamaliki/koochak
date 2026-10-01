"""Manifested collections: many files described by one manifest written last.

A collection lives under one store prefix::

    manifest.json            commit point: layout, packs, totals, file-table digest
    files.jsonl.gz           one row per file: path, size, sha256, mode, mtime_ns,
                             group, and either (pack, offset) or object
    packs/pack-000000.tar    plain tar packs of small files (layout "packed")
    packs/pack-000000.tar.json   per-pack record, written right after its pack
    objects/<path>           files stored as standalone objects

Readers never list storage: the file table answers listing, ``stat``, and
where each byte lives. A file's bytes are ``size`` bytes at ``offset`` in its
pack (an ordinary tar member) or the whole standalone object. Files of one
group are contiguous within a pack, so a group loads with one range read.
Tables are deterministic (sorted rows, no timestamps, gzip mtime 0), so
committing the same collection twice is accepted and a different one refused.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import tarfile
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from .store import Store, validate_key

__all__ = [
    "COLLECTION_KIND",
    "FILES_KEY",
    "LAYOUTS",
    "MANIFEST_KEY",
    "Collection",
    "FileEntry",
    "PackEntry",
    "PackRange",
    "load_collection",
    "member_header",
    "object_key",
    "pack_key",
    "pack_ranges",
    "put_once",
    "read_pack_range",
    "read_pack_ranges",
    "tar_padding",
    "write_collection",
]

COLLECTION_KIND = "koochak.collection"
VERSION = 1
MANIFEST_KEY = "manifest.json"
FILES_KEY = "files.jsonl.gz"
LAYOUTS = ("packed", "objects")
TAR_BLOCK = 512
TAR_END = b"\0" * (2 * TAR_BLOCK)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def pack_key(index: int) -> str:
    return f"packs/pack-{index:06d}.tar"


def object_key(path: str) -> str:
    return f"objects/{path}"


def tar_padding(size: int) -> bytes:
    return b"\0" * (-size % TAR_BLOCK)


def member_header(path: str, size: int, mode: int, mtime_ns: int) -> bytes:
    """Deterministic tar header for one regular file (PAX for long or non-ASCII names)."""

    info = tarfile.TarInfo(path)
    info.size = size
    info.mode = mode & 0o7777
    info.mtime = mtime_ns // 1_000_000_000
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.type = tarfile.REGTYPE
    return info.tobuf(format=tarfile.PAX_FORMAT, encoding="utf-8", errors="strict")


def _int(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return value


@dataclass(frozen=True, slots=True)
class FileEntry:
    """One file of a collection and where its bytes live."""

    path: str
    size: int
    sha256: str
    mode: int
    mtime_ns: int
    group: str
    pack: Optional[int] = None
    offset: Optional[int] = None
    object: Optional[str] = None

    def __post_init__(self) -> None:
        validate_key(self.path, "file path")
        _int(self.size, "size")
        _sha(self.sha256, "sha256")
        _int(self.mode, "mode")
        _int(self.mtime_ns, "mtime_ns")
        if not isinstance(self.group, str) or not self.group:
            raise ValueError(f"group of {self.path} must be a non-empty string")
        packed = self.pack is not None or self.offset is not None
        if packed == (self.object is not None):
            raise ValueError(f"{self.path} must live in exactly one of a pack or an object")
        if packed:
            _int(self.pack, "pack")
            _int(self.offset, "offset")
        else:
            validate_key(self.object, "object key")

    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "mode": self.mode,
            "mtime_ns": self.mtime_ns,
            "group": self.group,
        }
        if self.object is not None:
            row["object"] = self.object
        else:
            row["pack"] = self.pack
            row["offset"] = self.offset
        return row

    @classmethod
    def from_row(cls, row: object) -> "FileEntry":
        base = {"path", "size", "sha256", "mode", "mtime_ns", "group"}
        if not isinstance(row, dict) or set(row) not in (base | {"object"}, base | {"pack", "offset"}):
            raise ValueError(f"invalid file-table row: {row!r}")
        return cls(**row)


@dataclass(frozen=True, slots=True)
class PackEntry:
    key: str
    size: int
    sha256: str
    files: int

    def __post_init__(self) -> None:
        validate_key(self.key, "pack key")
        _int(self.size, "pack size")
        _sha(self.sha256, "pack sha256")
        _int(self.files, "pack file count", minimum=1)


@dataclass(frozen=True)
class Collection:
    """A committed collection: its layout, packs, and files sorted by path."""

    layout: str
    packs: tuple[PackEntry, ...]
    files: tuple[FileEntry, ...]
    metadata: Mapping[str, Any]
    manifest_sha256: str

    @property
    def size_bytes(self) -> int:
        return sum(entry.size for entry in self.files)

    def select(self, patterns: Sequence[str] = ()) -> Iterator[FileEntry]:
        """Files whose path matches any glob (all files when ``patterns`` is empty)."""

        from fnmatch import fnmatchcase

        for entry in self.files:
            if not patterns or any(fnmatchcase(entry.path, pattern) for pattern in patterns):
                yield entry


@dataclass(frozen=True)
class PackRange:
    """One read of ``length`` bytes at ``start`` in a pack, covering whole member files."""

    pack: PackEntry
    start: int
    length: int
    members: tuple[FileEntry, ...]


def pack_ranges(
    files: Iterable[FileEntry], packs: Sequence[PackEntry], *, merge_gap: int, max_range: int
) -> list[PackRange]:
    """Cover packed ``files`` with few reads, in pack and offset order.

    Neighbours in one pack share a read when the gap between them is at most
    ``merge_gap`` bytes and the read stays within ``max_range`` bytes.
    """

    by_pack: Dict[int, list[FileEntry]] = {}
    for entry in files:
        if entry.pack is None:
            raise ValueError(f"{entry.path} is a standalone object, not a pack member")
        by_pack.setdefault(entry.pack, []).append(entry)
    ranges: list[PackRange] = []
    for index, members in sorted(by_pack.items()):
        members.sort(key=lambda entry: entry.offset)
        current: list[FileEntry] = []
        start = end = 0
        for entry in members:
            entry_end = entry.offset + entry.size
            if current and (entry.offset - end > merge_gap or entry_end - start > max_range):
                ranges.append(PackRange(packs[index], start, end - start, tuple(current)))
                current = []
            if not current:
                start = entry.offset
            current.append(entry)
            end = entry_end
        if current:
            ranges.append(PackRange(packs[index], start, end - start, tuple(current)))
    return ranges


def read_pack_range(store: Store, part: PackRange, *, verify: bool = True) -> Dict[str, bytes]:
    """Read one range and split it into its files, checking each file's SHA256 when ``verify``."""

    return _split(part, store.get(part.pack.key, part.start, part.length), verify)


def read_pack_ranges(store: Store, parts: Sequence[PackRange], *, verify: bool = True) -> Dict[str, bytes]:
    """Read several ranges of one pack through a single open handle.

    On object-storage mounts opening a file can cost far more than reading a
    few kilobytes from it, so scattered files of one pack share one open.
    """

    if len({part.pack.key for part in parts}) != 1:
        raise ValueError("read_pack_ranges reads ranges of exactly one pack")
    files: Dict[str, bytes] = {}
    with store.open(parts[0].pack.key) as handle:
        for part in sorted(parts, key=lambda item: item.start):
            handle.seek(part.start)
            files.update(_split(part, handle.read(part.length), verify))
    return files


def _split(part: PackRange, data: bytes, verify: bool) -> Dict[str, bytes]:
    if len(data) != part.length:
        raise ValueError(f"{part.pack.key} is shorter than its manifest says")
    files: Dict[str, bytes] = {}
    for entry in part.members:
        begin = entry.offset - part.start
        blob = data[begin : begin + entry.size]
        if verify and hashlib.sha256(blob).hexdigest() != entry.sha256:
            raise ValueError(f"{entry.path} in {part.pack.key} does not match its SHA256")
        files[entry.path] = blob
    return files


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _encode_table(files: Sequence[FileEntry]) -> bytes:
    lines = b"".join(_canonical_json(entry.to_row()) for entry in files)
    return gzip.compress(lines, compresslevel=6, mtime=0)


def put_once(store: Store, key: str, data: bytes) -> None:
    """Create ``key``; an identical existing object is accepted, a different one refused."""

    try:
        store.put(key, data)
    except FileExistsError as exc:
        if store.get(key) != data:
            raise FileExistsError(f"{key} already exists with different content") from exc


def _validate(layout: str, packs: Sequence[PackEntry], files: Sequence[FileEntry]) -> None:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    paths = [entry.path for entry in files]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ValueError("file table must be sorted by path with unique paths")
    counts = [0] * len(packs)
    for entry in files:
        if entry.pack is None:
            continue
        if entry.pack >= len(packs):
            raise ValueError(f"{entry.path} refers to missing pack {entry.pack}")
        if entry.offset + entry.size > packs[entry.pack].size:
            raise ValueError(f"{entry.path} extends past the end of its pack")
        counts[entry.pack] += 1
    for pack, count in zip(packs, counts):
        if pack.files != count:
            raise ValueError(f"{pack.key} lists {pack.files} files but the table has {count}")


def write_collection(
    store: Store,
    *,
    layout: str,
    packs: Sequence[PackEntry],
    files: Iterable[FileEntry],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Collection:
    """Write the file table, then the manifest (the commit point)."""

    ordered = tuple(sorted(files, key=lambda entry: entry.path))
    packs = tuple(packs)
    _validate(layout, packs, ordered)
    table = _encode_table(ordered)
    put_once(store, FILES_KEY, table)
    manifest = _canonical_json(
        {
            "kind": COLLECTION_KIND,
            "v": VERSION,
            "layout": layout,
            "files_table": {
                "key": FILES_KEY,
                "size": len(table),
                "sha256": hashlib.sha256(table).hexdigest(),
            },
            "packs": [
                {"key": pack.key, "size": pack.size, "sha256": pack.sha256, "files": pack.files}
                for pack in packs
            ],
            "totals": {
                "files": len(ordered),
                "bytes": sum(entry.size for entry in ordered),
                "packs": len(packs),
                "objects": sum(1 for entry in ordered if entry.object is not None),
            },
            "metadata": dict(metadata or {}),
        }
    )
    put_once(store, MANIFEST_KEY, manifest)
    return Collection(
        layout, packs, ordered, MappingProxyType(dict(metadata or {})), hashlib.sha256(manifest).hexdigest()
    )


def load_collection(store: Store) -> Collection:
    """Read and strictly validate a committed collection."""

    raw = store.get(MANIFEST_KEY)
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("collection manifest is not valid JSON") from exc
    keys = {"kind", "v", "layout", "files_table", "packs", "totals", "metadata"}
    if not isinstance(manifest, dict) or set(manifest) != keys:
        raise ValueError(f"collection manifest must contain exactly {sorted(keys)}")
    if manifest["kind"] != COLLECTION_KIND or manifest["v"] != VERSION:
        raise ValueError("unsupported collection kind or version")
    table_ref = manifest["files_table"]
    if not isinstance(table_ref, dict) or set(table_ref) != {"key", "size", "sha256"}:
        raise ValueError("invalid files_table reference")
    table = store.get(validate_key(table_ref["key"], "files table key"))
    if len(table) != table_ref["size"] or hashlib.sha256(table).hexdigest() != table_ref["sha256"]:
        raise ValueError("file table does not match the manifest's size and SHA256")
    files = tuple(FileEntry.from_row(json.loads(line)) for line in gzip.decompress(table).splitlines())
    if not isinstance(manifest["packs"], list):
        raise ValueError("manifest packs must be a list")
    packs = tuple(PackEntry(**pack) if isinstance(pack, dict) else None for pack in manifest["packs"])
    if any(pack is None for pack in packs):
        raise ValueError("manifest packs must be objects")
    _validate(manifest["layout"], packs, files)
    totals = manifest["totals"]
    expected = {
        "files": len(files),
        "bytes": sum(entry.size for entry in files),
        "packs": len(packs),
        "objects": sum(1 for entry in files if entry.object is not None),
    }
    if totals != expected:
        raise ValueError("manifest totals do not match the file table")
    if not isinstance(manifest["metadata"], dict):
        raise ValueError("manifest metadata must be an object")
    return Collection(
        manifest["layout"],
        packs,  # type: ignore[arg-type]
        files,
        MappingProxyType(manifest["metadata"]),
        hashlib.sha256(raw).hexdigest(),
    )
