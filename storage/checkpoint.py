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
