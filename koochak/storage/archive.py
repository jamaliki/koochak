"""Archive local trees into collections; pull, verify, and list them.

``archive`` walks a local directory (where listing is cheap), packs files
smaller than ``object_bytes`` into ~``pack_bytes`` tar packs, stores larger
files as standalone objects, and commits the collection manifest last:

- A groups table (``path,group[,order]``) keeps each group's files contiguous
  in one pack, ordered by ``order`` then group; a group larger than a pack
  spills into the next. Unlisted files group by directory, after listed ones.
- The plan depends only on file metadata, so every pack's layout is known
  before reading. Each pack's record (``<pack>.json``, file digests included)
  is written right after the pack; a rerun reuses finished packs without
  reading them again, rebuilds a missing record from the stored pack, and
  refuses packs that differ from the plan.
- Sources are checked while read; a file whose size or mtime changed fails.
- Symlinks and special files are refused; exclude them with globs.

``pull`` restores all or some files (globs) by merging nearby pack members
into range reads, checks every SHA256, and restores mode and exact mtime on
local targets. Existing targets of the right size are skipped, so reruns
resume. ``verify`` checks sizes, or with ``deep=True`` re-reads every byte.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, BinaryIO, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .collection import (
    TAR_END,
    FileEntry,
    PackEntry,
    load_collection,
    member_header,
    object_key,
    pack_key,
    put_once,
    tar_padding,
    write_collection,
)
from .store import LocalStore, Store, min_object_bytes, profile_of, validate_key
from .transfer import TransferItem, copy_objects

__all__ = [
    "ArchivePlan",
    "ArchiveReport",
    "PullReport",
    "SourceFile",
    "VerifyReport",
    "archive",
    "load_file_list",
    "load_groups",
    "plan_archive",
    "pull",
    "scan_source",
    "verify",
]

DEFAULT_PACK_BYTES = 512 * 1024 * 1024
_UNGROUPED_ORDER = sys.maxsize
_CHUNK = 8 * 1024 * 1024

ProgressFn = Callable[[str, int, int, int], None]
GroupsLike = Union[str, os.PathLike, Mapping[str, tuple[str, int]], None]


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: str
    size: int
    mode: int
    mtime_ns: int
    group: str
    order: int


@dataclass(frozen=True, slots=True)
class PlannedMember:
    file: SourceFile
    header: bytes
    offset: int


@dataclass(frozen=True, slots=True)
class PlannedPack:
    index: int
    members: tuple[PlannedMember, ...]
    size: int

    @property
    def key(self) -> str:
        return pack_key(self.index)


@dataclass(frozen=True)
class ArchivePlan:
    layout: str
    packs: tuple[PlannedPack, ...]
    objects: tuple[SourceFile, ...]
    object_bytes: int

    @property
    def files(self) -> int:
        return sum(len(pack.members) for pack in self.packs) + len(self.objects)

    @property
    def stored_bytes(self) -> int:
        return sum(pack.size for pack in self.packs) + sum(item.size for item in self.objects)


@dataclass(frozen=True)
class ArchiveReport:
    files: int
    bytes: int
    packs: int
    objects: int
    packs_reused: int
    requests: int
    seconds: float
    dry_run: bool
    manifest_sha256: Optional[str] = None
    deleted_sources: int = 0
    changed_sources: tuple[str, ...] = ()
    missing_sources: int = 0


@dataclass(frozen=True)
class PullReport:
    files: int
    copied: int
    skipped: int
    bytes_copied: int
    range_reads: int
    seconds: float


@dataclass
class VerifyReport:
    files: int
    packs: int
    objects: int
    deep: bool
    bytes_checked: int = 0
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


# ---------------------------------------------------------------- sources


def load_groups(path: Union[str, os.PathLike]) -> Dict[str, tuple[str, int]]:
    """Read ``path,group[,order]`` rows; ``order`` is an integer (default 0)."""

    groups: Dict[str, tuple[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        if not {"path", "group"} <= columns or columns - {"path", "group", "order"}:
            raise ValueError(f"{path}: a groups table has columns path,group and optionally order")
        for line, row in enumerate(reader, start=2):
            where = f"{path}:{line}"
            relative = validate_key(row["path"], f"{where} path")
            group = row["group"]
            if not group:
                raise ValueError(f"{where}: empty group")
            order_text = (row.get("order") or "").strip()
            try:
                order = int(order_text) if order_text else 0
            except ValueError as exc:
                raise ValueError(f"{where}: order must be an integer") from exc
            previous = groups.setdefault(relative, (group, order))
            if previous != (group, order):
                raise ValueError(f"{where}: conflicting group for {relative}")
    return groups


def _source_file(
    relative: str,
    observed: os.stat_result,
    groups: Optional[Mapping[str, tuple[str, int]]],
) -> SourceFile:
    if stat.S_ISLNK(observed.st_mode):
        raise ValueError(f"{relative} is a symlink; archives store no links (use --exclude)")
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{relative} is not a regular file")
    validate_key(relative, "source path")
    if groups is not None and relative in groups:
        group, order = groups[relative]
    else:
        parent = relative.rpartition("/")[0]
        group, order = f"dir:{parent or '.'}", _UNGROUPED_ORDER
    return SourceFile(
        relative, observed.st_size, stat.S_IMODE(observed.st_mode), observed.st_mtime_ns, group, order
    )


def load_file_list(path: Union[str, os.PathLike]) -> List[str]:
    """Relative paths, one per line; blank lines and ``#`` comments are ignored."""

    with open(path, encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]
    return [line for line in lines if line.strip() and not line.lstrip().startswith("#")]


def scan_source(
    root: Union[str, os.PathLike],
    *,
    groups: Optional[Mapping[str, tuple[str, int]]] = None,
    exclude: Sequence[str] = (),
    files: Optional[Sequence[str]] = None,
) -> List[SourceFile]:
    """Regular files under ``root`` (all of them, or exactly ``files``), sorted, with groups.

    With ``files`` nothing is walked: each listed path must exist and be a
    regular file, so a stale list fails instead of silently archiving less.
    """

    root = os.path.abspath(os.fspath(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"archive source is not a directory: {root}")

    def excluded(relative: str) -> bool:
        return any(fnmatchcase(relative, pattern) for pattern in exclude)

    found: List[SourceFile] = []
    if files is not None:
        missing = []
        for relative in sorted(set(files)):
            validate_key(relative, "listed path")
            if excluded(relative):
                continue
            try:
                observed = os.lstat(_local(root, relative))
            except FileNotFoundError:
                missing.append(relative)
                continue
            found.append(_source_file(relative, observed, groups))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} listed paths do not exist under {root}, e.g. {missing[:5]}"
            )
        return found

    for directory, child_directories, filenames in os.walk(root, followlinks=False):
        relative_dir = os.path.relpath(directory, root)
        prefix = "" if relative_dir == "." else relative_dir.replace(os.sep, "/") + "/"
        kept = []
        for name in sorted(child_directories):
            relative = prefix + name
            if excluded(relative):
                continue
            if os.path.islink(os.path.join(directory, name)):
                raise ValueError(f"{relative} is a symlink; archives store no links (use --exclude)")
            kept.append(name)
        child_directories[:] = kept
        for name in sorted(filenames):
            relative = prefix + name
            if not excluded(relative):
                found.append(_source_file(relative, os.lstat(os.path.join(directory, name)), groups))
    if groups:
        present = {item.path for item in found}
        missing = [path for path in groups if path not in present and not excluded(path)]
        if missing:
            raise ValueError(
                f"the groups table lists {len(missing)} paths that are not under {root}, "
                f"e.g. {missing[:5]}"
            )
    found.sort(key=lambda item: item.path)
    return found


# ---------------------------------------------------------------- planning


def _member_bytes(header: bytes, size: int) -> int:
    return len(header) + size + len(tar_padding(size))


def plan_archive(
    files: Sequence[SourceFile],
    *,
    layout: str = "packed",
    pack_bytes: int = DEFAULT_PACK_BYTES,
    object_bytes: int,
) -> ArchivePlan:
    """Assign files to packs or standalone objects, from metadata alone."""

    if layout not in ("packed", "objects"):
        raise ValueError(f"layout must be 'packed' or 'objects', got {layout!r}")
    if type(pack_bytes) is not int or pack_bytes < 4 * 1024:
        raise ValueError("pack_bytes must be an integer >= 4096")
    if type(object_bytes) is not int or object_bytes < 1:
        raise ValueError("object_bytes must be a positive integer")
    if layout == "objects":
        return ArchivePlan(layout, (), tuple(files), object_bytes)
    objects = tuple(item for item in files if item.size >= object_bytes)
    packed = sorted(
        (item for item in files if item.size < object_bytes),
        key=lambda item: (item.order, item.group, item.path),
    )
    packs: List[PlannedPack] = []
    members: List[PlannedMember] = []
    used = 0

    def close() -> None:
        nonlocal members, used
        if members:
            packs.append(PlannedPack(len(packs), tuple(members), used + len(TAR_END)))
            members, used = [], 0

    start = 0
    while start < len(packed):
        end = start
        while end < len(packed) and packed[end].group == packed[start].group:
            end += 1
        run = [(item, member_header(item.path, item.size, item.mode, item.mtime_ns)) for item in packed[start:end]]
        run_bytes = sum(_member_bytes(header, item.size) for item, header in run)
        if members and used + run_bytes + len(TAR_END) > pack_bytes:
            close()
        for item, header in run:
            size = _member_bytes(header, item.size)
            if members and used + size + len(TAR_END) > pack_bytes:
                close()  # only a group larger than one pack reaches this
            members.append(PlannedMember(item, header, used + len(header)))
            used += size
        start = end
    close()
    return ArchivePlan(layout, tuple(packs), objects, object_bytes)


# ---------------------------------------------------------------- archive


def _local(root: str, relative: str) -> str:
    return os.path.join(root, *relative.split("/"))


def _read_source(root: str, item: SourceFile) -> bytes:
    with open(_local(root, item.path), "rb") as handle:
        observed = os.fstat(handle.fileno())
        data = handle.read()
    if len(data) != item.size or observed.st_size != item.size or observed.st_mtime_ns != item.mtime_ns:
        raise ValueError(f"source changed during the archive: {item.path}")
    return data


def _hash_source(root: str, item: SourceFile) -> str:
    digest = hashlib.sha256()
    size = 0
    with open(_local(root, item.path), "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    if size != item.size:
        raise ValueError(f"source changed during the archive: {item.path}")
    return digest.hexdigest()


def _read_exact(handle: BinaryIO, count: int, *digests: Any) -> None:
    while count:
        chunk = handle.read(min(count, _CHUNK))
        if not chunk:
            raise ValueError("stored object ended early")
        for digest in digests:
            digest.update(chunk)
        count -= len(chunk)


def _digest_members(handle: BinaryIO, spans: Sequence[tuple[int, int]]) -> tuple[str, List[str], int]:
    """Stream an object once: its SHA256, each span's SHA256, and its size."""

    whole = hashlib.sha256()
    position = 0
    digests: List[str] = []
    for offset, size in spans:
        if offset < position:
            raise ValueError("pack members overlap")
        _read_exact(handle, offset - position, whole)
        member = hashlib.sha256()
        _read_exact(handle, size, whole, member)
        digests.append(member.hexdigest())
        position = offset + size
    while chunk := handle.read(_CHUNK):
        whole.update(chunk)
        position += len(chunk)
    return whole.hexdigest(), digests, position


def _entries(pack: PlannedPack, digests: Sequence[str]) -> List[FileEntry]:
    return [
        FileEntry(
            member.file.path,
            member.file.size,
            digest,
            member.file.mode,
            member.file.mtime_ns,
            member.file.group,
            pack=pack.index,
            offset=member.offset,
        )
        for member, digest in zip(pack.members, digests)
    ]


def _record(pack: PlannedPack, sha256: str, digests: Sequence[str]) -> bytes:
    body = {
        "v": 1,
        "key": pack.key,
        "size": pack.size,
        "sha256": sha256,
        "files": [
            {"path": member.file.path, "size": member.file.size, "offset": member.offset, "sha256": digest}
            for member, digest in zip(pack.members, digests)
        ],
    }
    return (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _upload(target: Store, pack: PlannedPack, contents: Sequence[bytes]) -> tuple[PackEntry, List[FileEntry]]:
    chunks: List[bytes] = []
    digests: List[str] = []
    for member, data in zip(pack.members, contents):
        chunks.extend((member.header, data, tar_padding(len(data))))
        digests.append(hashlib.sha256(data).hexdigest())
    chunks.append(TAR_END)
    whole = hashlib.sha256()
    for chunk in chunks:
        whole.update(chunk)
    target.put(pack.key, chunks)
    put_once(target, pack.key + ".json", _record(pack, whole.hexdigest(), digests))
    return PackEntry(pack.key, pack.size, whole.hexdigest(), len(pack.members)), _entries(pack, digests)


def _reuse(target: Store, pack: PlannedPack) -> Optional[tuple[PackEntry, List[FileEntry]]]:
    """A finished pack from an earlier run, or ``None`` when it must be written."""

    info = target.stat(pack.key)
    record_key = pack.key + ".json"
    planned = [(m.file.path, m.file.size, m.offset) for m in pack.members]
    if target.stat(record_key) is not None:
        record = json.loads(target.get(record_key))
        stored = [(f["path"], f["size"], f["offset"]) for f in record["files"]]
        if record["size"] != pack.size or stored != planned or info is None or info.size != pack.size:
            raise FileExistsError(
                f"{pack.key} from an earlier run does not match this plan (the source or options "
                "changed); use a new destination"
            )
        digests = [f["sha256"] for f in record["files"]]
        return PackEntry(pack.key, pack.size, record["sha256"], len(pack.members)), _entries(pack, digests)
    if info is None:
        return None
    if info.size != pack.size:
        raise FileExistsError(
            f"{pack.key} exists with {info.size} bytes but the plan needs {pack.size}; it was left by "
            "an interrupted or different archive: delete it or use a new destination"
        )
    # The pack is complete but its record was not written: rebuild it from the stored bytes.
    with target.open(pack.key) as handle:
        whole, digests, size = _digest_members(handle, [(m.offset, m.file.size) for m in pack.members])
    if size != pack.size:
        raise ValueError(f"{pack.key} changed size while being read")
    put_once(target, record_key, _record(pack, whole, digests))
    return PackEntry(pack.key, pack.size, whole, len(pack.members)), _entries(pack, digests)


def archive(
    source: Store,
    target: Store,
    *,
    groups: GroupsLike = None,
    exclude: Sequence[str] = (),
    layout: str = "packed",
    pack_bytes: int = DEFAULT_PACK_BYTES,
    object_bytes: Optional[int] = None,
    streams: Optional[int] = None,
    read_streams: int = 8,
    pending_packs: int = 8,
    metadata: Optional[Mapping[str, Any]] = None,
    files: Optional[Sequence[str]] = None,
    delete_source: bool = False,
    dry_run: bool = False,
    progress: Optional[ProgressFn] = None,
) -> ArchiveReport:
    """Archive the local tree under ``source`` into a collection at ``target``.

    ``files`` archives exactly those relative paths instead of walking the
    tree. ``object_bytes`` defaults to ``min_object_bytes(profile_of(target))``
    (at least 1 MiB, at most ``pack_bytes``). Memory peaks near
    ``pending_packs * pack_bytes``.

    ``delete_source`` turns the archive into a move: after the collection is
    committed it is re-read and every SHA256 checked (``verify(deep=True)``);
    only then are source files deleted, and only those whose size and mtime
    still match the file table. Failed verification deletes nothing; changed
    files are kept and reported. Directories are left in place.
    """

    if not isinstance(source, LocalStore):
        raise TypeError("archive sources must be local directories (LocalStore)")
    for name, value in (("read_streams", read_streams), ("pending_packs", pending_packs)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    started = time.perf_counter()
    group_map = load_groups(groups) if isinstance(groups, (str, os.PathLike)) else groups
    listed = files
    files = scan_source(source.root, groups=group_map, exclude=exclude, files=listed)
    if not files:
        raise ValueError(f"nothing to archive under {source.root}")
    if object_bytes is None:
        object_bytes = min(max(min_object_bytes(profile_of(target)), 1024 * 1024), pack_bytes)
    plan = plan_archive(files, layout=layout, pack_bytes=pack_bytes, object_bytes=object_bytes)
    requests = 2 * len(plan.packs) + len(plan.objects) + 2
    total_bytes = sum(item.size for item in files)
    if dry_run:
        return ArchiveReport(
            len(files), total_bytes, len(plan.packs), len(plan.objects), 0, requests,
            time.perf_counter() - started, True,
        )

    entries: List[FileEntry] = []
    if plan.objects:
        with ThreadPoolExecutor(max_workers=read_streams) as pool:
            digests = list(pool.map(lambda item: _hash_source(source.root, item), plan.objects))
        copy_objects(
            source,
            target,
            [
                TransferItem(item.path, object_key(item.path), item.size, digest)
                for item, digest in zip(plan.objects, digests)
            ],
            streams=streams,
            progress=(lambda done, total, size: progress("objects", done, total, size)) if progress else None,
        )
        entries.extend(
            FileEntry(item.path, item.size, digest, item.mode, item.mtime_ns, item.group, object=object_key(item.path))
            for item, digest in zip(plan.objects, digests)
        )

    pack_entries: List[Optional[PackEntry]] = [None] * len(plan.packs)
    reused = 0
    written_bytes = 0
    first_error: Optional[BaseException] = None
    running: Dict[Future[tuple[PackEntry, List[FileEntry]]], int] = {}

    def collect(finished: Sequence[Future[tuple[PackEntry, List[FileEntry]]]]) -> None:
        nonlocal first_error, written_bytes
        for future in finished:
            index = running.pop(future)
            try:
                pack_entry, pack_files = future.result()
            except BaseException as exc:
                # Keep the first failure; uploads already running finish first.
                first_error = first_error or exc
                continue
            pack_entries[index] = pack_entry
            entries.extend(pack_files)
            written_bytes += pack_entry.size
            if progress:
                progress("packs", sum(e is not None for e in pack_entries), len(plan.packs), written_bytes)

    with ThreadPoolExecutor(max_workers=read_streams) as readers, ThreadPoolExecutor(
        max_workers=pending_packs
    ) as uploads:
        for pack in plan.packs:
            if first_error is not None:
                break
            finished = _reuse(target, pack)
            if finished is not None:
                pack_entries[pack.index] = finished[0]
                entries.extend(finished[1])
                reused += 1
                continue
            contents = list(readers.map(lambda member: _read_source(source.root, member.file), pack.members))
            while len(running) >= pending_packs:
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                collect(list(done))
            if first_error is not None:
                break
            running[uploads.submit(_upload, target, pack, contents)] = pack.index
        while running:
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            collect(list(done))
    if first_error is not None:
        raise first_error
    collection = write_collection(
        target,
        layout=plan.layout,
        packs=[entry for entry in pack_entries if entry is not None],
        files=entries,
        metadata=metadata,
    )
    deleted, changed, missing = 0, (), 0
    if delete_source:
        checked = verify(target, deep=True, streams=streams)
        if not checked.ok:
            raise ValueError(
                f"deep verification of the new collection failed; no source file was deleted: "
                f"{checked.problems[:5]}"
            )
        deleted, changed, missing = _delete_sources(source.root, collection.files)
    return ArchiveReport(
        len(files), total_bytes, len(plan.packs), len(plan.objects), reused, requests,
        time.perf_counter() - started, False, collection.manifest_sha256,
        deleted, changed, missing,
    )


def _delete_sources(root: str, entries: Sequence[FileEntry]) -> tuple[int, tuple[str, ...], int]:
    """Delete archived sources that are unchanged since the archive read them."""

    deleted = missing = 0
    changed: List[str] = []
    for entry in entries:
        path = _local(root, entry.path)
        try:
            observed = os.lstat(path)
        except FileNotFoundError:
            missing += 1
            continue
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_size != entry.size
            or observed.st_mtime_ns != entry.mtime_ns
        ):
            changed.append(entry.path)
            continue
        os.remove(path)
        deleted += 1
    return deleted, tuple(changed), missing


# ---------------------------------------------------------------- pull


@dataclass(frozen=True)
class _Range:
    pack: PackEntry
    start: int
    length: int
    members: tuple[FileEntry, ...]


def _ranges(files: Sequence[FileEntry], packs: Sequence[PackEntry], merge_gap: int, max_range: int) -> List[_Range]:
    by_pack: Dict[int, List[FileEntry]] = {}
    for entry in files:
        by_pack.setdefault(entry.pack, []).append(entry)  # type: ignore[arg-type]
    ranges: List[_Range] = []
    for index, members in sorted(by_pack.items()):
        members.sort(key=lambda entry: entry.offset)
        current: List[FileEntry] = []
        start = end = 0
        for entry in members:
            entry_end = entry.offset + entry.size
            if current and (entry.offset - end > merge_gap or entry_end - start > max_range):
                ranges.append(_Range(packs[index], start, end - start, tuple(current)))
                current = []
            if not current:
                start = entry.offset
            current.append(entry)
            end = entry_end
        if current:
            ranges.append(_Range(packs[index], start, end - start, tuple(current)))
    return ranges


def _restore_metadata(target: Store, entry: FileEntry) -> None:
    path = target.local_path(entry.path)
    if path is not None:
        os.chmod(path, entry.mode & 0o7777)
        os.utime(path, ns=(entry.mtime_ns, entry.mtime_ns))


def _pull_range(source: Store, target: Store, part: _Range) -> int:
    data = source.get(part.pack.key, part.start, part.length)
    if len(data) != part.length:
        raise ValueError(f"{part.pack.key} is shorter than its manifest says")
    for entry in part.members:
        begin = entry.offset - part.start
        blob = data[begin : begin + entry.size]
        if hashlib.sha256(blob).hexdigest() != entry.sha256:
            raise ValueError(f"{entry.path} in {part.pack.key} does not match its SHA256")
        target.put(entry.path, blob)
    return sum(entry.size for entry in part.members)


def pull(
    source: Store,
    target: Store,
    *,
    include: Sequence[str] = (),
    streams: Optional[int] = None,
    merge_gap: int = 4 * 1024 * 1024,
    max_range: int = 64 * 1024 * 1024,
    restore_metadata: bool = True,
    progress: Optional[ProgressFn] = None,
) -> PullReport:
    """Restore a collection's files (or those matching ``include`` globs) into ``target``."""

    started = time.perf_counter()
    collection = load_collection(source)
    selected = list(collection.select(include))
    if include and not selected:
        raise ValueError(f"no files in the collection match {list(include)}")
    streams = profile_of(source).streams if streams is None else streams
    with ThreadPoolExecutor(max_workers=min(32, max(1, streams))) as pool:
        present = list(pool.map(lambda entry: target.stat(entry.path), selected))
    todo: List[FileEntry] = []
    skipped = 0
    for entry, info in zip(selected, present):
        if info is None:
            todo.append(entry)
        elif info.size == entry.size:
            skipped += 1
        else:
            raise FileExistsError(f"{entry.path} already exists with {info.size} bytes, expected {entry.size}")
    copied_bytes = 0
    objects = [entry for entry in todo if entry.object is not None]
    if objects:
        report = copy_objects(
            source,
            target,
            [TransferItem(entry.object, entry.path, entry.size, entry.sha256) for entry in objects],  # type: ignore[arg-type]
            streams=streams,
        )
        copied_bytes += report.bytes_copied
    ranges = _ranges([entry for entry in todo if entry.pack is not None], collection.packs, merge_gap, max_range)
    with ThreadPoolExecutor(max_workers=max(1, streams)) as pool:
        futures = [pool.submit(_pull_range, source, target, part) for part in ranges]
        for done, future in enumerate(as_completed(futures), start=1):
            copied_bytes += future.result()
            if progress:
                progress("ranges", done, len(ranges), copied_bytes)
    if restore_metadata:
        with ThreadPoolExecutor(max_workers=min(32, max(1, streams))) as pool:
            list(pool.map(lambda entry: _restore_metadata(target, entry), selected))
    return PullReport(
        len(selected), len(todo), skipped, copied_bytes, len(ranges), time.perf_counter() - started
    )


# ---------------------------------------------------------------- verify


def _verify_pack(store: Store, pack: PackEntry, members: Sequence[FileEntry]) -> List[str]:
    ordered = sorted(members, key=lambda entry: entry.offset)
    with store.open(pack.key) as handle:
        whole, digests, size = _digest_members(handle, [(entry.offset, entry.size) for entry in ordered])
    problems = [f"{entry.path}: SHA256 mismatch" for entry, digest in zip(ordered, digests) if digest != entry.sha256]
    if size != pack.size or whole != pack.sha256:
        problems.append(f"{pack.key}: size or SHA256 differs from the manifest")
    return problems


def _verify_object(store: Store, entry: FileEntry) -> List[str]:
    with store.open(entry.object) as handle:  # type: ignore[arg-type]
        whole, _, size = _digest_members(handle, [])
    if size != entry.size or whole != entry.sha256:
        return [f"{entry.path}: object {entry.object} differs from the manifest"]
    return []


def verify(store: Store, *, deep: bool = False, streams: Optional[int] = None) -> VerifyReport:
    """Check a collection: stored sizes, or with ``deep`` every byte's SHA256."""

    collection = load_collection(store)
    objects = [entry for entry in collection.files if entry.object is not None]
    report = VerifyReport(len(collection.files), len(collection.packs), len(objects), deep)
    streams = min(8, profile_of(store).streams) if streams is None else streams
    with ThreadPoolExecutor(max_workers=max(1, streams)) as pool:
        sizes = list(pool.map(lambda key: store.stat(key), [pack.key for pack in collection.packs]))
        for pack, info in zip(collection.packs, sizes):
            if info is None or info.size != pack.size:
                report.problems.append(f"{pack.key}: missing or wrong size")
        object_sizes = list(pool.map(lambda entry: store.stat(entry.object), objects))
        for entry, info in zip(objects, object_sizes):
            if info is None or info.size != entry.size:
                report.problems.append(f"{entry.path}: object {entry.object} missing or wrong size")
        if deep and report.ok:
            members: Dict[int, List[FileEntry]] = {}
            for entry in collection.files:
                if entry.pack is not None:
                    members.setdefault(entry.pack, []).append(entry)
            futures = [
                pool.submit(_verify_pack, store, pack, members.get(index, []))
                for index, pack in enumerate(collection.packs)
            ]
            futures += [pool.submit(_verify_object, store, entry) for entry in objects]
            for future in futures:
                report.problems.extend(future.result())
            report.bytes_checked = sum(pack.size for pack in collection.packs) + sum(e.size for e in objects)
    return report
