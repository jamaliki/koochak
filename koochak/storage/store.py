"""Immutable object stores: one narrow interface over POSIX and object storage.

A store holds write-once objects addressed by relative ``/``-separated keys.
``put`` is create-only and returns only once the object is readable with the
exact bytes written, so a writer can publish a small manifest last and treat
it as the commit point on any backend.  The contract deliberately has no
rename, append, overwrite, or symlink: object stores and FUSE mounts over them
cannot provide those atomically, and nothing Koochak persists needs them.

Backends are selected by location.  Plain paths and ``file://`` URIs map to
:class:`LocalStore`; any other ``scheme://`` resolves through
:func:`register_store` or the ``koochak.stores`` entry-point group, so a
private package can add a backend without Koochak naming it.
"""

from __future__ import annotations

import errno
import hashlib
import math
import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass
from importlib import metadata
from typing import BinaryIO, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Union, runtime_checkable
from urllib.parse import unquote, urlparse

from ..utils.paths import canonical_dir
from .atomic import _fsync_directory

__all__ = [
    "DEFAULT_PROFILE",
    "ENTRY_POINT_GROUP",
    "LOCAL_PROFILE",
    "LocalStore",
    "ObjectInfo",
    "Store",
    "StoreProfile",
    "min_object_bytes",
    "open_store",
    "profile_of",
    "register_store",
    "validate_key",
]

ENTRY_POINT_GROUP = "koochak.stores"

PutData = Union[bytes, bytearray, memoryview, Iterable[Union[bytes, bytearray, memoryview]]]
StoreFactory = Callable[[str], "Store"]

_TEMP_MARKER = ".koochak-put-"
_BAD_SEGMENTS = frozenset({"", ".", ".."})
_SCHEME = re.compile(r"[a-z][a-z0-9+.-]*")
_PUBLISH_MODES = ("link", "exclusive")
_READ_CHUNK = 8 * 1024 * 1024
# errno values meaning "this filesystem cannot make hard links at all".
_NO_LINK_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, "EPERM", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "ENOSYS", None),
        getattr(errno, "EXDEV", None),
    )
    if code is not None
)


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    """Size of a stored object, plus its SHA256 when the store computed it."""

    key: str
    size: int
    sha256: Optional[str] = None


@dataclass(frozen=True, slots=True)
class StoreProfile:
    """Measured performance of a store; layers above derive their defaults from it.

    - ``request_seconds``: fixed cost of one small request.
    - ``stream_mb_s``: uncached bandwidth of one stream, in MB/s.
    - ``streams``: concurrent transfers worth running against the store.
    - ``range_streams``: parallel byte ranges worth reading from one large
      object (1 when ranges of one object do not scale).
    - ``part_bytes``: size of those ranges and of checkpoint parts.
    - ``list_is_cheap``: whether listing is acceptable at all; when False,
      callers must use manifests.

    ``python -m koochak.storage.probe`` reports a recommended profile.
    """

    request_seconds: float = 0.001
    stream_mb_s: float = 500.0
    streams: int = 4
    range_streams: int = 1
    part_bytes: int = 64 * 1024 * 1024
    list_is_cheap: bool = True

    def __post_init__(self) -> None:
        for name in ("request_seconds", "stream_mb_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("streams", "range_streams", "part_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.list_is_cheap, bool):
            raise ValueError("list_is_cheap must be a boolean")


# Local disks: cheap requests and listings, little to gain from ranges.
LOCAL_PROFILE = StoreProfile()
# Backends that declare nothing are treated like remote object storage.
DEFAULT_PROFILE = StoreProfile(
    request_seconds=0.1,
    stream_mb_s=50.0,
    streams=16,
    range_streams=1,
    part_bytes=32 * 1024 * 1024,
    list_is_cheap=False,
)


def profile_of(store: "Store") -> StoreProfile:
    """Return the store's declared profile, or the conservative remote default."""

    profile = getattr(store, "profile", None)
    if profile is None:
        return DEFAULT_PROFILE
    if not isinstance(profile, StoreProfile):
        raise TypeError(f"{type(store).__name__}.profile must be a StoreProfile")
    return profile


def min_object_bytes(profile: StoreProfile, *, max_overhead: float = 0.25) -> int:
    """Smallest file worth storing as its own object rather than packing it.

    Per-request cost wastes ``t / (t + size / bandwidth)`` of a read; this is
    the size at which that fraction falls to ``max_overhead``.
    """

    if not 0 < max_overhead < 1:
        raise ValueError("max_overhead must be between 0 and 1")
    bytes_per_second = profile.stream_mb_s * 1e6
    return int(profile.request_seconds * bytes_per_second * (1 - max_overhead) / max_overhead)


@runtime_checkable
class Store(Protocol):
    """Write-once objects under relative keys.

    - ``get`` reads a whole object or the byte range ``[offset, offset+length)``
      (short at end of object).
    - ``open`` streams an object.
    - ``put`` creates an object and raises ``FileExistsError`` if the key
      exists.  It returns only when other clients can read exactly these bytes.
    - ``stat`` returns ``None`` for a missing key.
    - ``list`` returns the sorted keys that start with ``prefix``.
    - ``delete`` removes an object; a missing key is not an error.
    - ``local_path`` returns a POSIX path for the object when the backend has
      one (for ``mmap`` or libraries that need a filename), else ``None``.
    """

    def get(self, key: str, offset: int = 0, length: Optional[int] = None) -> bytes: ...

    def open(self, key: str) -> BinaryIO: ...

    def put(self, key: str, data: PutData) -> ObjectInfo: ...

    def stat(self, key: str) -> Optional[ObjectInfo]: ...

    def list(self, prefix: str = "") -> List[str]: ...

    def delete(self, key: str) -> None: ...

    def local_path(self, key: str) -> Optional[str]: ...


def validate_key(key: object, label: str = "key") -> str:
    """Return ``key`` if it is a relative, normalized ``/``-separated path."""

    if not isinstance(key, str) or not key:
        raise ValueError(f"{label} must be a non-empty string")
    if "\x00" in key or "\\" in key:
        raise ValueError(f"{label} must not contain NUL or backslash: {key!r}")
    for segment in key.split("/"):
        if segment in _BAD_SEGMENTS:
            raise ValueError(
                f"{label} must be relative, without empty, '.' or '..' segments: {key!r}"
            )
        if _TEMP_MARKER in segment:
            raise ValueError(f"{label} must not contain {_TEMP_MARKER!r}: {key!r}")
    return key


def _validate_prefix(prefix: object) -> str:
    if prefix == "":
        return ""
    if not isinstance(prefix, str):
        raise ValueError("prefix must be a string")
    validate_key(prefix[:-1] if prefix.endswith("/") else prefix, "prefix")
    return prefix


def _check_range(offset: object, length: object) -> None:
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if length is not None and (type(length) is not int or length < 0):
        raise ValueError("length must be a non-negative integer or None")


def _chunks(data: PutData) -> Iterator[Union[bytes, bytearray, memoryview]]:
    if isinstance(data, (bytes, bytearray, memoryview)):
        yield data
        return
    if isinstance(data, str):
        raise TypeError("put data must be bytes or an iterable of bytes, not str")
    for chunk in data:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError(f"put data chunks must be bytes-like, got {type(chunk).__name__}")
        yield chunk


def _file_digest(path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


class LocalStore:
    """Store backed by a POSIX directory tree.

    ``publish="link"`` (default) writes a hidden temporary file and hard-links
    it into place, so readers never observe a partial object and an existing
    key is never replaced.  ``publish="exclusive"`` creates the final name with
    ``O_EXCL`` and writes it directly; use it on mounts that support neither
    rename nor hard links (typically FUSE over object storage) and publish a
    file when it is closed.  Readers must then trust objects only through a
    manifest written last, which is how Koochak uses every store anyway.

    ``fsync=False`` skips file and directory fsync on mounts that reject or
    ignore it.  ``verify_readback=True`` re-reads every new object until its
    size and SHA256 match what was written, retrying for up to
    ``settle_seconds`` on mounts whose close completes asynchronously.
    ``profile`` records the mount's measured performance (default:
    ``LOCAL_PROFILE``). ``python -m koochak.storage.probe`` reports which
    settings and profile a mount needs.
    """

    def __init__(
        self,
        root: Union[str, os.PathLike[str]],
        *,
        publish: str = "link",
        fsync: bool = True,
        verify_readback: bool = False,
        settle_seconds: float = 0.0,
        file_mode: int = 0o444,
        profile: Optional[StoreProfile] = None,
    ) -> None:
        if publish not in _PUBLISH_MODES:
            raise ValueError(f"publish must be one of {_PUBLISH_MODES}, got {publish!r}")
        if not isinstance(fsync, bool) or not isinstance(verify_readback, bool):
            raise ValueError("fsync and verify_readback must be booleans")
        settle = float(settle_seconds)
        if not math.isfinite(settle) or settle < 0:
            raise ValueError("settle_seconds must be a finite, non-negative number")
        if settle > 0 and not verify_readback:
            raise ValueError("settle_seconds requires verify_readback=True")
        if type(file_mode) is not int or not 0 <= file_mode <= 0o777:
            raise ValueError("file_mode must be a permission mode between 0 and 0o777")
        if profile is not None and not isinstance(profile, StoreProfile):
            raise TypeError("profile must be a StoreProfile")
        self.profile = LOCAL_PROFILE if profile is None else profile
        self.root = canonical_dir(os.fspath(root))
        self.publish = publish
        self.fsync = fsync
        self.verify_readback = verify_readback
        self.settle_seconds = settle
        self.file_mode = file_mode

    def __repr__(self) -> str:
        return (
            f"LocalStore({self.root!r}, publish={self.publish!r}, fsync={self.fsync}, "
            f"verify_readback={self.verify_readback}, settle_seconds={self.settle_seconds})"
        )

    def _path(self, key: str) -> str:
        return os.path.join(self.root, *validate_key(key).split("/"))

    def get(self, key: str, offset: int = 0, length: Optional[int] = None) -> bytes:
        _check_range(offset, length)
        with open(self._path(key), "rb") as handle:
            if offset:
                handle.seek(offset)
            return handle.read() if length is None else handle.read(length)

    def open(self, key: str) -> BinaryIO:
        return open(self._path(key), "rb")

    def put(self, key: str, data: PutData) -> ObjectInfo:
        path = self._path(key)
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        if self.publish == "link":
            size, digest = self._put_link(path, directory, data)
        else:
            size, digest = self._put_exclusive(path, data)
        if self.fsync:
            _fsync_directory(directory)
        if self.verify_readback:
            self._await_readback(path, size, digest)
        return ObjectInfo(key, size, digest)

    def _write(self, handle: BinaryIO, data: PutData) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        for chunk in _chunks(data):
            handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        handle.flush()
        if self.fsync:
            os.fsync(handle.fileno())
        return size, digest.hexdigest()

    def _put_link(self, path: str, directory: str, data: PutData) -> tuple[int, str]:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}{_TEMP_MARKER}", dir=directory
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                result = self._write(handle, data)
            os.chmod(temporary, self.file_mode)
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise
            except OSError as exc:
                if exc.errno in _NO_LINK_ERRNOS:
                    raise OSError(
                        exc.errno,
                        f"{self.root} does not support hard links; use "
                        "LocalStore(..., publish='exclusive') on this mount",
                    ) from exc
                raise
            return result
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _put_exclusive(self, path: str, data: PutData) -> tuple[int, str]:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(path, flags, self.file_mode)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                return self._write(handle, data)
        except BaseException:
            # This call created the inode, so it owns the partial object.
            # Remove it rather than truncating: some mounts cannot truncate.
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            raise

    def _await_readback(self, path: str, size: int, digest: str) -> None:
        deadline = time.monotonic() + self.settle_seconds
        last_error: Optional[OSError] = None
        while True:
            try:
                if _file_digest(path) == (size, digest):
                    return
            except OSError as exc:
                # A mount that closes asynchronously may briefly fail reads of
                # a just-written object; keep polling until the deadline.
                last_error = exc
            if time.monotonic() >= deadline:
                detail = f" (last error: {last_error})" if last_error is not None else ""
                raise OSError(
                    f"{path} did not read back with the written size and SHA256 "
                    f"within {self.settle_seconds}s{detail}"
                )
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

    def stat(self, key: str) -> Optional[ObjectInfo]:
        try:
            observed = os.stat(self._path(key))
        except (FileNotFoundError, NotADirectoryError):
            return None
        if not stat.S_ISREG(observed.st_mode):
            return None
        return ObjectInfo(key, observed.st_size)

    def list(self, prefix: str = "") -> List[str]:
        prefix = _validate_prefix(prefix)
        head = prefix.rpartition("/")[0]
        base = os.path.join(self.root, *head.split("/")) if head else self.root
        if not os.path.isdir(base):
            return []
        keys: List[str] = []
        for directory, child_directories, filenames in os.walk(base):
            relative = os.path.relpath(directory, self.root)
            parent = "" if relative == "." else relative.replace(os.sep, "/") + "/"
            child_directories[:] = [
                name
                for name in child_directories
                if (parent + name).startswith(prefix) or prefix.startswith(parent + name + "/")
            ]
            for name in filenames:
                if name.startswith(".") and _TEMP_MARKER in name:
                    continue
                key = parent + name
                if key.startswith(prefix):
                    keys.append(key)
        keys.sort()
        return keys

    def delete(self, key: str) -> None:
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass

    def local_path(self, key: str) -> Optional[str]:
        return self._path(key)


_FACTORIES: Dict[str, StoreFactory] = {}


def register_store(scheme: str, factory: StoreFactory) -> None:
    """Map ``scheme://...`` locations to ``factory(location) -> Store``."""

    if not isinstance(scheme, str) or not _SCHEME.fullmatch(scheme):
        raise ValueError(f"invalid store scheme {scheme!r}")
    if scheme == "file":
        raise ValueError("the 'file' scheme is built in")
    if not callable(factory):
        raise TypeError("store factory must be callable")
    existing = _FACTORIES.get(scheme)
    if existing is not None and existing is not factory:
        raise ValueError(f"store scheme {scheme!r} is already registered")
    _FACTORIES[scheme] = factory


def _factory(scheme: str) -> StoreFactory:
    factory = _FACTORIES.get(scheme)
    if factory is not None:
        return factory
    installed = list(metadata.entry_points(group=ENTRY_POINT_GROUP))
    matches = [entry for entry in installed if entry.name == scheme]
    if not matches:
        known = sorted({"file", *_FACTORIES, *(entry.name for entry in installed)})
        raise ValueError(f"no store is registered for scheme {scheme!r}; known schemes: {known}")
    if len(matches) > 1:
        sources = sorted(entry.value for entry in matches)
        raise ValueError(f"several installed packages provide store scheme {scheme!r}: {sources}")
    factory = matches[0].load()
    register_store(scheme, factory)
    return factory


def open_store(location: Union[str, os.PathLike[str], Store]) -> Store:
    """Return the store for a path, ``file://`` URI, or registered ``scheme://`` URI."""

    if isinstance(location, Store):
        return location
    text = os.fspath(location)
    if not isinstance(text, str) or not text:
        raise ValueError("store location must be a non-empty string or path")
    if "://" not in text:
        return LocalStore(text)
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme == "file":
        if parsed.netloc not in ("", "localhost") or parsed.params or parsed.query or parsed.fragment:
            raise ValueError(f"file URIs must be file:///absolute/path, got {text!r}")
        return LocalStore(unquote(parsed.path))
    store = _factory(scheme)(text)
    if not isinstance(store, Store):
        raise TypeError(
            f"store factory for {scheme!r} returned {type(store).__name__}, which is not a Store"
        )
    return store
