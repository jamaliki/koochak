from __future__ import annotations

import io
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import torch

__all__ = [
    "save",
    "load",
    "latest",
    "best",
]


_STEP_RE = re.compile(r"step(\d+)\.pt$")


def _mkdir_p(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_bytes(data: bytes, path: str) -> None:
    """Atomically write bytes to path using a temporary file + rename.

    This avoids partial writes on interruption. The temporary file is placed
    in the same directory to ensure `os.replace` is atomic on POSIX.
    """

    directory = os.path.dirname(os.path.abspath(path)) or "."
    _mkdir_p(directory)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        # If replace failed, make sure the tmp is removed
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


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


def _list_step_checkpoints(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    files = []
    for name in os.listdir(directory):
        if _STEP_RE.search(name):
            files.append(os.path.join(directory, name))
    return sorted(files)


def _extract_step(path: str) -> int:
    m = _STEP_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def _prune_keep_last_k(directory: str, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    files = _list_step_checkpoints(directory)
    if len(files) <= keep_last_k:
        return
    files_sorted = sorted(files, key=_extract_step)
    to_delete = files_sorted[:-keep_last_k]
    for path in to_delete:
        try:
            os.remove(path)
        except OSError:
            pass


def save(ckpt: Dict[str, Any], path: str, keep_last_k: int = 3) -> str:
    """Save a checkpoint dict to `path` atomically and prune old ones.

    Returns the saved path. If the filename matches `step*.pt`, retains only
    the last `keep_last_k` matching files in the same directory. Also updates
    a `latest.pt` pointer next to the file for convenience.
    """

    directory = os.path.dirname(os.path.abspath(path)) or "."
    _mkdir_p(directory)

    data = _torch_save_to_bytes(ckpt)
    _atomic_write_bytes(data, path)

    # Best-effort: maintain a latest pointer and prune
    _maybe_symlink_latest(path)
    _prune_keep_last_k(directory, keep_last_k)
    return path


def load(path: str) -> Dict[str, Any]:
    """Load a checkpoint dict from `path` (map to CPU)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def latest(directory: str) -> Optional[str]:
    """Return the most recent step checkpoint or `latest.pt` if it exists."""
    latest_path = os.path.join(directory, "latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    files = _list_step_checkpoints(directory)
    if not files:
        return None
    files_sorted = sorted(files, key=_extract_step)
    return files_sorted[-1]


def best(directory: str, key: str = "val_loss") -> Optional[str]:
    """Return the path to the checkpoint with the best metric (min).

    Loads each step checkpoint's `metrics[key]` and selects the minimum.
    Returns None if no checkpoints or key not found anywhere.
    """

    files = _list_step_checkpoints(directory)
    best_path: Optional[str] = None
    best_val: float = float("inf")
    for p in files:
        try:
            ckpt = torch.load(p, map_location="cpu")
        except Exception:
            continue
        metrics = ckpt.get("metrics") or {}
        if key in metrics:
            try:
                v = float(metrics[key])
            except Exception:
                continue
            if v < best_val:
                best_val = v
                best_path = p
    return best_path

