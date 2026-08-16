from __future__ import annotations

import os
import tempfile


def _fsync_directory(directory: str) -> None:
    """Persist a same-directory rename before reporting publication complete."""

    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: str, data: bytes, tmp_suffix: str = ".tmp") -> None:
    """Atomically write bytes to `path` using a same-dir temp file + replace."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=tmp_suffix, dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(directory)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
