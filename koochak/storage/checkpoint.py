from __future__ import annotations

import io
import os
import re
import shutil
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

__all__ = [
    "save",
    "load",
    "latest",
    "best",
    "strip_module_prefix",
    "add_module_prefix",
    "match_state_dict_to_model",
]

from . import fs as fs_utils
from .atomic import atomic_write
from .pruning import prune_keep_last_k


def _torch_save_to_bytes(obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


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

    # Best-effort: maintain a latest pointer and prune
    _maybe_symlink_latest(path)
    prune_keep_last_k(directory, pattern=_STEP_RE.pattern, k=keep_last_k)
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
