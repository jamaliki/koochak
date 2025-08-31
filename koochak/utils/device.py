from __future__ import annotations

from typing import Any, Mapping

import torch

__all__ = ["get_device", "get_lr"]


def get_device(cfg: Mapping[str, Any] | Any) -> torch.device:
    dev = None
    if isinstance(cfg, Mapping):
        dev = cfg.get("device")
    else:
        dev = getattr(cfg, "device", None)
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def get_lr(optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))

