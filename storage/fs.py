from __future__ import annotations

import os
import re
from typing import Optional, List

import torch


def mkdir_p(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def latest(directory: str, pattern: str = r"step(\d+)\.pt$") -> Optional[str]:
    latest_path = os.path.join(directory, "latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    if not os.path.isdir(directory):
        return None
    rx = re.compile(pattern)
    files: List[str] = [
        os.path.join(directory, f) for f in os.listdir(directory) if rx.search(f)
    ]
    if not files:
        return None
    files.sort(key=lambda p: int(rx.search(os.path.basename(p)).group(1)))  # type: ignore[union-attr]
    return files[-1]


def best(directory: str, key: str = "val_loss", pattern: str = r"step(\d+)\.pt$") -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    rx = re.compile(pattern)
    files: List[str] = [
        os.path.join(directory, f) for f in os.listdir(directory) if rx.search(f)
    ]
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

