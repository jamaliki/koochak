from __future__ import annotations

import io
import os
import re
import shutil
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

from .atomic import atomic_write
from .pruning import prune_keep_last_k
from . import fs as fs_utils


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
    try:
        if os.path.islink(latest_path) or os.path.exists(latest_path):
            try:
                os.remove(latest_path)
            except OSError:
                pass
        os.symlink(os.path.basename(step_path), latest_path)
        return latest_path
    except (OSError, NotImplementedError):
        # Fallback: copy file (not atomic but acceptable as a convenience)
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
    return torch.load(path, map_location="cpu")


def latest(directory: str) -> Optional[str]:
    return fs_utils.latest(directory, pattern=_STEP_RE.pattern)


def best(directory: str, key: str = "val_loss") -> Optional[str]:
    return fs_utils.best(directory, key=key, pattern=_STEP_RE.pattern)


# ---------- DDP compatibility helpers ----------
from collections import OrderedDict


def _has_module_prefix(sd: "OrderedDict[str, torch.Tensor] | Dict[str, torch.Tensor]") -> bool:
    try:
        for k in sd.keys():
            if isinstance(k, str) and k.startswith("module."):
                return True
    except Exception:
        pass
    return False


def strip_module_prefix(sd: Dict[str, Any]) -> "OrderedDict[str, Any]":
    out: "OrderedDict[str, Any]" = OrderedDict()
    for k, v in sd.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        out[nk] = v
    return out


def add_module_prefix(sd: Dict[str, Any]) -> "OrderedDict[str, Any]":
    out: "OrderedDict[str, Any]" = OrderedDict()
    for k, v in sd.items():
        nk = (f"module.{k}" if isinstance(k, str) and not k.startswith("module.") else k)
        out[nk] = v
    return out


def match_state_dict_to_model(model: Any, sd: Dict[str, Any]) -> "OrderedDict[str, Any]":
    """Return a state_dict whose keys match the target model.

    If model expects `module.*` keys but sd doesn't, add them; if the reverse, strip them.
    Otherwise return sd unchanged.
    """
    try:
        model_keys = list(getattr(model, "state_dict")().keys())
    except Exception:
        return OrderedDict(sd)
    model_expects_module = any(isinstance(k, str) and k.startswith("module.") for k in model_keys)
    sd_has_module = _has_module_prefix(sd)
    if model_expects_module and not sd_has_module:
        return add_module_prefix(sd)
    if (not model_expects_module) and sd_has_module:
        return strip_module_prefix(sd)
    return OrderedDict(sd)
