from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import re
import shutil
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

__all__ = [
    "BackgroundPublisher",
    "add_module_prefix",
    "best",
    "checkpoint_path",
    "checkpoint_store",
    "highest_valid_published",
    "latest_valid",
    "latest",
    "load",
    "match_state_dict_to_model",
    "prune",
    "publication",
    "publication_path",
    "publish",
    "resolve_auto_resume",
    "save",
    "serialize",
    "strip_module_prefix",
]

from ..utils.paths import canonical_dir
from . import fs as fs_utils
from .store import LocalStore, Store, open_store

PUBLICATION_SUFFIX = ".ready.json"
_NUMBERED = re.compile(r"step(\d+)\.pt")
_STEP_RE = re.compile(r"step(\d+)\.pt$")
# A candidate that is absent or not a regular file is skipped. Other I/O errors
# propagate: an unreadable newest checkpoint must not silently roll training back.
_MISSING = (FileNotFoundError, IsADirectoryError, NotADirectoryError)
_INVALID = (
    EOFError,
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    IndexError,
    KeyError,
    ImportError,
    MemoryError,
    OverflowError,
    pickle.PickleError,
)

Location = Union[str, "os.PathLike[str]", Store]


@dataclass(frozen=True, slots=True)
class AutoResumeSelection:
    """Validated auto-resume evidence and its first unexecuted update."""

    path: str
    checkpoint: Dict[str, Any]
    next_step: int


def serialize(ckpt: Dict[str, Any]) -> bytes:
    """Return ``torch.save`` bytes for a checkpoint dict."""

    buffer = io.BytesIO()
    torch.save(ckpt, buffer)
    return buffer.getvalue()


def checkpoint_store(location: Location) -> Store:
    """Return the store that holds a run's numbered checkpoints.

    ``location`` is a directory, a ``file://`` URI, a named ``scheme://`` URI
    from the stores file (``koochak.storage.stores_file``), or a ``Store``.  A
    plain directory keeps the historical behaviour: hard-link publication,
    owner-only files, and a ``latest.pt`` symlink.  A named scheme brings its
    own settings, e.g. ``publish: exclusive`` and ``read_settle_seconds`` for an
    object-storage mount.
    """

    if isinstance(location, Store):
        return location
    text = os.fspath(location)
    if "://" in text:
        return open_store(text)
    return LocalStore(canonical_dir(text), file_mode=0o600)


def checkpoint_path(store: Store, name: str) -> str:
    """Return the filesystem path of checkpoint ``name`` in ``store``."""

    path = store.local_path(name)
    if path is None:
        raise TypeError(
            f"{store!r} exposes no filesystem paths; checkpoint stores must be filesystem-backed"
        )
    return path


def publication_path(path: str) -> str:
    """Return the sidecar written only after an immutable checkpoint is ready."""

    return os.path.abspath(path) + PUBLICATION_SUFFIX


def _publication(path: str, size: int, sha256: str) -> dict[str, Any]:
    absolute = os.path.abspath(path)
    return {
        "v": 1,
        "artifact_id": f"checkpoint/{os.path.basename(absolute)}",
        "path": absolute,
        "size_bytes": size,
        "sha256": sha256,
        "manifest_path": publication_path(absolute),
    }


def publication(path: str) -> dict[str, Any]:
    """Read the small immutable publication record for a saved checkpoint."""

    manifest = publication_path(path)
    with open(manifest, encoding="utf-8") as handle:
        record = json.load(handle)
    expected = {"v", "artifact_id", "path", "size_bytes", "sha256", "manifest_path"}
    if not isinstance(record, dict) or set(record) != expected or record.get("v") != 1:
        raise ValueError(f"invalid checkpoint publication manifest: {manifest}")
    return record


def _maybe_symlink_latest(step_path: str) -> Optional[str]:
    """Create or update a `latest.pt` symlink next to the given step file.

    If symlink creation is not supported, falls back to copying. Returns the
    path to the `latest.pt` file if created, else None.
    """

    directory = os.path.dirname(os.path.abspath(step_path))
    latest_path = os.path.join(directory, "latest.pt")
    if os.path.islink(latest_path) or os.path.exists(latest_path):
        # An existing symlink/file may refuse atomic replace on some filesystems;
        # unlink eagerly and treat absence as success.
        try:
            os.remove(latest_path)
        except FileNotFoundError:
            pass
        except OSError:
            # Permission/in-use error: skip symlink, try copy below.
            return _copy_latest(step_path, latest_path)
    try:
        os.symlink(os.path.basename(step_path), latest_path)
        return latest_path
    except (OSError, NotImplementedError):
        return _copy_latest(step_path, latest_path)


def _copy_latest(step_path: str, latest_path: str) -> Optional[str]:
    try:
        shutil.copy2(step_path, latest_path)
        return latest_path
    except OSError:
        return None


def _numbered(store: Store) -> List[Tuple[int, str]]:
    """Return ``(step, key)`` for every top-level ``step<N>.pt`` object."""

    found = []
    for key in store.list("step"):
        match = _NUMBERED.fullmatch(key)
        if match is not None:
            found.append((int(match.group(1)), key))
    return sorted(found)


def _read_valid_published_numbered_checkpoint(
    store: Store, name: str
) -> Optional[AutoResumeSelection]:
    """Fail closed unless a numbered checkpoint and its exact manifest agree.

    Reads go through the store, so a mount configured with
    ``read_settle_seconds`` waits out files that another node closed moments
    ago instead of mistaking them for invalid ones.
    """

    match = _NUMBERED.fullmatch(name)
    if match is None:
        return None
    absolute = os.path.abspath(checkpoint_path(store, name))
    manifest_path = publication_path(absolute)
    try:
        for path in (absolute, manifest_path):
            if not os.path.isfile(path) or os.path.islink(path):
                return None
        record = json.loads(store.get(name + PUBLICATION_SUFFIX).decode("utf-8"))
        expected = {"v", "artifact_id", "path", "size_bytes", "sha256", "manifest_path"}
        if not isinstance(record, dict) or set(record) != expected:
            return None
        if record["v"] != 1:
            return None
        if record["artifact_id"] != f"checkpoint/{name}":
            return None
        if record["path"] != absolute or record["manifest_path"] != manifest_path:
            return None
        info = store.stat(name)
        if info is None:
            return None
        if type(record["size_bytes"]) is not int or record["size_bytes"] != info.size:
            return None
        payload = store.get(name)
        if len(payload) != info.size:
            return None
        if record["sha256"] != hashlib.sha256(payload).hexdigest():
            return None
        checkpoint = torch.load(io.BytesIO(payload), weights_only=False, map_location="cpu")
        if not isinstance(checkpoint, dict):
            return None
        step = checkpoint.get("step")
        next_step = checkpoint.get("next_step")
        if type(step) is not int or type(next_step) is not int or next_step < 0:
            return None
        if step != int(match.group(1)):
            return None
        # Periodic checkpoints store the last zero-based update; terminal
        # checkpoints store the completed-update cursor.  Both are valid.
        if next_step not in (step, step + 1):
            return None
    except _MISSING:
        return None
    except _INVALID:
        return None
    return AutoResumeSelection(absolute, checkpoint, next_step)


def highest_valid_published(location: Location) -> Optional[str]:
    """Return the highest numbered checkpoint with valid immutable evidence.

    ``latest.pt`` and directories that merely look like checkpoint scaffolding
    are intentionally ignored.  Invalid candidates do not poison a lower,
    valid checkpoint; I/O errors other than a missing file are raised.
    """

    selection = _highest_valid(checkpoint_store(location))
    return None if selection is None else selection.path


latest_valid = highest_valid_published


def _highest_valid(store: Store) -> Optional[AutoResumeSelection]:
    for _step, key in reversed(_numbered(store)):
        selection = _read_valid_published_numbered_checkpoint(store, key)
        if selection is not None:
            return selection
    return None


def resolve_auto_resume(location: Location) -> Optional[tuple[str, Dict[str, Any]]]:
    """Return validated checkpoint bytes and cursor for first-run-safe resume.

    The checkpoint payload is loaded from the same bytes whose size and SHA256
    were validated against the ready manifest.  Callers that need the resume
    cursor before constructing a dataset can use this API without reimplementing
    Koochak's evidence checks.  ``location`` is anything ``checkpoint_store``
    accepts.
    """

    selection = _highest_valid(checkpoint_store(location))
    return None if selection is None else (selection.path, selection.checkpoint)


def prune(store: Store, keep_last_k: int) -> None:
    """Delete numbered checkpoints older than the newest ``keep_last_k``.

    Each manifest is deleted before its checkpoint, so a checkpoint is
    uncommitted before any of its bytes disappear.
    """

    if keep_last_k <= 0:
        return
    for _step, key in _numbered(store)[:-keep_last_k]:
        store.delete(key + PUBLICATION_SUFFIX)
        store.delete(key)


def publish(store: Store, name: str, data: bytes, *, keep_last_k: int = 3) -> str:
    """Commit serialized checkpoint bytes as ``name`` and return its path.

    The checkpoint is written create-only, then its ready manifest; readers
    that trust only manifests never see a partial checkpoint.  An existing
    checkpoint of the same name is uncommitted (manifest first) and replaced.
    Stores that publish with hard links also keep a ``latest.pt`` symlink;
    write-once stores get none, since without symlinks it would be a full
    copy.  Numbered checkpoints beyond ``keep_last_k`` are then pruned.
    """

    path = checkpoint_path(store, name)
    manifest = name + PUBLICATION_SUFFIX
    if store.stat(name) is not None or store.stat(manifest) is not None:
        store.delete(manifest)
        store.delete(name)
    info = store.put(name, data)
    digest = info.sha256 if info.sha256 is not None else hashlib.sha256(data).hexdigest()
    record = _publication(path, info.size, digest)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    store.put(manifest, (encoded + "\n").encode("utf-8"))
    if isinstance(store, LocalStore) and store.publish == "link":
        _maybe_symlink_latest(path)
    prune(store, keep_last_k)
    return path


def save(ckpt: Dict[str, Any], path: str, keep_last_k: int = 3) -> str:
    """Save a checkpoint dict to `path` and prune old ones; return `path`.

    See ``publish``: the file is committed by its ready manifest, an existing
    file of the same name is replaced, and only the newest ``keep_last_k``
    numbered checkpoints in the directory are kept.
    """

    directory = os.path.dirname(os.path.abspath(path)) or "."
    publish(
        checkpoint_store(directory),
        os.path.basename(path),
        serialize(ckpt),
        keep_last_k=keep_last_k,
    )
    return path


class BackgroundPublisher:
    """Publish serialized checkpoints from one background thread, one at a time.

    ``submit`` hands over the bytes and returns the checkpoint's path at once.
    ``poll`` returns ``(path, token)`` once that publication has committed (or
    ``None``), ``wait`` blocks for it, and both re-raise a failed publication on
    the caller's thread.  Only one publication may be pending; ``wait`` for it
    before submitting the next.  ``close`` lets a pending publication finish,
    discards its outcome, and stops the thread.
    """

    def __init__(self, store: Store, *, keep_last_k: int) -> None:
        self.store = store
        self.keep_last_k = keep_last_k
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="koochak-checkpoint")
        self._pending: Optional[Tuple[Future[str], Any]] = None

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def submit(self, name: str, data: bytes, token: Any = None) -> str:
        if self._pending is not None:
            raise RuntimeError("a checkpoint publication is already pending; wait() for it first")
        path = checkpoint_path(self.store, name)
        future = self._executor.submit(publish, self.store, name, data, keep_last_k=self.keep_last_k)
        self._pending = (future, token)
        return path

    def poll(self) -> Optional[Tuple[str, Any]]:
        if self._pending is None or not self._pending[0].done():
            return None
        return self.wait()

    def wait(self) -> Optional[Tuple[str, Any]]:
        if self._pending is None:
            return None
        future, token = self._pending
        self._pending = None
        return future.result(), token

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self._pending = None


def load(path: str) -> Dict[str, Any]:
    """Load a checkpoint dict from `path` (map to CPU)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, weights_only=False, map_location="cpu")


def _directory(location: Location) -> str:
    if isinstance(location, LocalStore):
        return location.root
    text = os.fspath(location)
    if "://" not in text:
        return text
    store = checkpoint_store(text)
    if not isinstance(store, LocalStore):
        raise TypeError(f"{text} does not resolve to a filesystem directory")
    return store.root


def latest(location: Location) -> Optional[str]:
    return fs_utils.latest(_directory(location), pattern=_STEP_RE.pattern)


def best(location: Location, key: str = "val_loss") -> Optional[str]:
    return fs_utils.best(_directory(location), key=key, pattern=_STEP_RE.pattern)


def _has_module_prefix(sd: "OrderedDict[str, torch.Tensor] | Dict[str, torch.Tensor]") -> bool:
    """Treat a state dict as DDP-wrapped only when every key carries the prefix.

    Mixed dicts (some prefixed, some not) are returned unchanged by
    `match_state_dict_to_model`, since either transformation would silently drop keys.
    """
    keys = list(sd.keys())
    if not keys:
        return False
    return all(isinstance(k, str) and k.startswith("module.") for k in keys)


def strip_module_prefix(sd: Dict[str, Any]) -> "OrderedDict[str, Any]":
    out: "OrderedDict[str, Any]" = OrderedDict()
    for k, v in sd.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        out[nk] = v
    return out


def add_module_prefix(sd: Dict[str, Any]) -> "OrderedDict[str, Any]":
    out: "OrderedDict[str, Any]" = OrderedDict()
    for k, v in sd.items():
        nk = f"module.{k}" if isinstance(k, str) and not k.startswith("module.") else k
        out[nk] = v
    return out


def match_state_dict_to_model(model: Any, sd: Dict[str, Any]) -> "OrderedDict[str, Any]":
    """Return a state_dict whose keys match the target model.

    If model expects `module.*` keys but sd doesn't, add them; if the reverse, strip them.
    Otherwise return sd unchanged.
    """
    state_dict_fn = getattr(model, "state_dict", None)
    if state_dict_fn is None:
        return OrderedDict(sd)
    model_keys = list(state_dict_fn().keys())
    model_expects_module = any(isinstance(k, str) and k.startswith("module.") for k in model_keys)
    sd_has_module = _has_module_prefix(sd)
    if model_expects_module and not sd_has_module:
        return add_module_prefix(sd)
    if (not model_expects_module) and sd_has_module:
        return strip_module_prefix(sd)
    return OrderedDict(sd)
