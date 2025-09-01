from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional

__all__ = [
    "parse_step_from_name",
    "parse_step_from_path",
    "make_checkpoint_aliases",
]


_STEP_RX = re.compile(r"step(\d+)\.pt$")


def parse_step_from_name(name: str, pattern: str = r"step(\d+)\.pt$") -> Optional[int]:
    """Extract numeric step from a checkpoint filename.

    Returns None if the pattern doesn't match.
    """
    rx = re.compile(pattern)
    m = rx.search(name)
    if not m:
        return None
    try:
        # Use the last captured group as the step if multiple are present
        return int([g for g in m.groups() if g][-1])
    except Exception:
        return None


def parse_step_from_path(path: str, pattern: str = r"step(\d+)\.pt$") -> Optional[int]:
    return parse_step_from_name(os.path.basename(path), pattern)


def make_checkpoint_aliases(
    step: Optional[int],
    *,
    include_latest: bool = True,
    best_keys: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return a list of standard aliases for a checkpoint artifact.

    - Always includes "latest" if include_latest is True.
    - Includes a step alias like "step-123" if step is not None.
    - If best_keys provided, adds "best" and key-specific aliases like "best-val_loss".
    """
    aliases: List[str] = []
    if include_latest:
        aliases.append("latest")
    if step is not None:
        aliases.append(f"step-{step}")
    if best_keys:
        # Generic best alias plus per-key markers
        aliases.append("best")
        for k in best_keys:
            aliases.append(f"best-{k}")
    return aliases

