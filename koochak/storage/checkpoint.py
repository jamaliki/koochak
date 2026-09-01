from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import re
import shutil
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

__all__ = [
    "add_module_prefix",
    "best",
    "highest_valid_published",
    "latest_valid",
    "latest",
    "load",
    "match_state_dict_to_model",
    "publication",
    "publication_path",
    "save",
    "strip_module_prefix",
]

from . import fs as fs_utils
from .atomic import atomic_write
from .pruning import prune_keep_last_k

PUBLICATION_SUFFIX = ".ready.json"


def _torch_save_to_bytes(obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def publication_path(path: str) -> str:
    """Return the sidecar written only after an immutable checkpoint is ready."""

    return os.path.abspath(path) + PUBLICATION_SUFFIX


def _publication(path: str, data: bytes) -> dict[str, Any]:
    absolute = os.path.abspath(path)
    return {
        "v": 1,
        "artifact_id": f"checkpoint/{os.path.basename(absolute)}",
        "path": absolute,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
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


_STEP_RE = re.compile(r"step(\d+)\.pt$")


def _valid_published_numbered_checkpoint(path: str) -> bool:
    """Fail closed unless a numbered checkpoint and its exact manifest agree."""

    absolute = os.path.abspath(path)
    name = os.path.basename(absolute)
    match = re.fullmatch(r"step(\d+)\.pt", name)
    if match is None or not os.path.isfile(absolute) or os.path.islink(absolute):
        return False
    manifest_path = publication_path(absolute)
    try:
        if not os.path.isfile(manifest_path) or os.path.islink(manifest_path):
            return False
        with open(manifest_path, encoding="utf-8") as handle:
            record = json.load(handle)
        expected = {"v", "artifact_id", "path", "size_bytes", "sha256", "manifest_path"}
        if not isinstance(record, dict) or set(record) != expected:
            return False
        if record["v"] != 1:
            return False
        if record["artifact_id"] != f"checkpoint/{name}":
            return False
        if record["path"] != absolute or record["manifest_path"] != manifest_path:
            return False
        size = os.path.getsize(absolute)
        if type(record["size_bytes"]) is not int or record["size_bytes"] != size:
            return False
        with open(absolute, "rb") as handle:
            payload = handle.read()
        if len(payload) != size:
            return False
        digest = hashlib.sha256(payload).hexdigest()
        if record["sha256"] != digest:
            return False
        checkpoint = torch.load(io.BytesIO(payload), weights_only=False, map_location="cpu")
        if not isinstance(checkpoint, dict):
            return False
        step = checkpoint.get("step")
        next_step = checkpoint.get("next_step")
        if type(step) is not int or type(next_step) is not int or next_step < 0:
            return False
        if step != int(match.group(1)):
            return False
        # Periodic checkpoints store the last zero-based update; terminal
        # checkpoints store the completed-update cursor.  Both are valid.
        if next_step not in (step, step + 1):
            return False
    except (
        OSError,
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
    ):
        return False
    return True


def highest_valid_published(directory: str) -> Optional[str]:
    """Return the highest numbered checkpoint with valid immutable evidence.

    ``latest.pt`` and directories that merely look like checkpoint scaffolding
    are intentionally ignored.  Invalid candidates do not poison a lower,
    valid checkpoint.
    """

    if not os.path.isdir(directory):
        return None
    candidates = []
    for name in os.listdir(directory):
        match = re.fullmatch(r"step(\d+)\.pt", name)
        if match is not None:
            candidates.append((int(match.group(1)), os.path.join(directory, name)))
    for _step, path in sorted(candidates, reverse=True):
        if _valid_published_numbered_checkpoint(path):
            return os.path.abspath(path)
    return None


latest_valid = highest_valid_published


def save(ckpt: Dict[str, Any], path: str, keep_last_k: int = 3) -> str:
    """Save a checkpoint dict to `path` atomically and prune old ones.

    Returns the saved path. If the filename matches `step*.pt`, retains only
    the last `keep_last_k` matching files in the same directory. Also updates
    a `latest.pt` pointer next to the file for convenience.
    """

    directory = os.path.dirname(os.path.abspath(path)) or "."
    fs_utils.mkdir_p(directory)

    data = _torch_save_to_bytes(ckpt)
    atomic_write(path, data)
    record = _publication(path, data)
    encoded = (
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    atomic_write(record["manifest_path"], encoded)

    # Best-effort: maintain a latest pointer and prune
    _maybe_symlink_latest(path)
    prune_keep_last_k(
        directory,
        pattern=_STEP_RE.pattern,
        k=keep_last_k,
        companion_suffixes=(PUBLICATION_SUFFIX,),
    )
    return path


def load(path: str) -> Dict[str, Any]:
    """Load a checkpoint dict from `path` (map to CPU)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, weights_only=False, map_location="cpu")


def latest(directory: str) -> Optional[str]:
    return fs_utils.latest(directory, pattern=_STEP_RE.pattern)


def best(directory: str, key: str = "val_loss") -> Optional[str]:
    return fs_utils.best(directory, key=key, pattern=_STEP_RE.pattern)


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
