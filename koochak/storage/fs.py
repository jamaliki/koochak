from __future__ import annotations

import os
import re
from typing import List, Optional

import torch


def mkdir_p(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _matching_step_files(directory: str, pattern: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    rx = re.compile(pattern)
    files = [os.path.join(directory, f) for f in os.listdir(directory) if rx.search(f)]
    files.sort(key=lambda p: int(rx.search(os.path.basename(p)).group(1)))  # type: ignore[union-attr]
    return files


def latest(directory: str, pattern: str = r"step(\d+)\.pt$") -> Optional[str]:
    latest_path = os.path.join(directory, "latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    files = _matching_step_files(directory, pattern)
    if not files:
        return None
    return files[-1]


def best(directory: str, key: str = "val_loss", pattern: str = r"step(\d+)\.pt$") -> Optional[str]:
    best_path: Optional[str] = None
    best_val: float = float("inf")
    for p in _matching_step_files(directory, pattern):
        try:
            ckpt = torch.load(p, weights_only=False, map_location="cpu")
        except (OSError, RuntimeError, EOFError):
            continue
        metrics = ckpt.get("metrics") or {}
        if key in metrics:
            try:
                v = float(metrics[key])
            except (TypeError, ValueError):
                continue
            if v < best_val:
                best_val = v
                best_path = p
    return best_path
