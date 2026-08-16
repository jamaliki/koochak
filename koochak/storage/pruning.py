from __future__ import annotations

import os
import re
from typing import List

from .naming import _step_from_match


def _extract_step(name: str, pattern: re.Pattern[str]) -> int:
    """Return the integer step from `name`, or -1 if it cannot be parsed."""
    m = pattern.search(name)
    if not m:
        return -1
    step = _step_from_match(m)
    return -1 if step is None else step


def prune_keep_last_k(
    directory: str,
    pattern: str,
    k: int,
    *,
    companion_suffixes: tuple[str, ...] = (),
) -> None:
    """Delete older matching files and their named companion records."""
    if k <= 0 or not os.path.isdir(directory):
        return
    rx = re.compile(pattern)
    files: List[str] = [
        os.path.join(directory, f) for f in os.listdir(directory) if rx.search(f)
    ]
    if len(files) <= k:
        return
    files_sorted = sorted(files, key=lambda p: _extract_step(os.path.basename(p), rx))
    for p in files_sorted[:-k]:
        try:
            os.remove(p)
        except OSError:
            continue
        for suffix in companion_suffixes:
            try:
                os.remove(p + suffix)
            except FileNotFoundError:
                pass
            except OSError:
                # Pruning is best-effort. The checkpoint itself is already
                # absent, so a stale small sidecar cannot be loaded as data.
                pass
