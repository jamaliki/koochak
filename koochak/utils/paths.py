from __future__ import annotations

import os

__all__ = ["canonical_dir"]


def canonical_dir(path: str) -> str:
    """Expand ``~`` and resolve to an absolute path. Used for `out_dir`-style settings."""
    return os.path.abspath(os.path.expanduser(str(path)))
