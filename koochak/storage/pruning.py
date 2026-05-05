from __future__ import annotations

import os
import re
from typing import List


def _extract_step(name: str, pattern: re.Pattern[str]) -> int:
    m = pattern.search(name)
    if not m:
        return -1
    # Choose the last captured group that is an int
    for g in reversed(m.groups()):
        try:
            return int(g)
        except ValueError:
            continue
    return -1


def prune_keep_last_k(directory: str, pattern: str, k: int) -> None:
    """Delete older matching files, keeping the last `k` by numeric step."""
    if k <= 0:
        return
    if not os.path.isdir(directory):
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
            pass
