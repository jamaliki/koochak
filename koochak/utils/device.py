from __future__ import annotations

import os
from typing import Any, Mapping

import torch

__all__ = ["get_device", "ensure_process_device", "get_lr"]


def get_device(cfg: Mapping[str, Any] | Any) -> torch.device:
    dev = None
    if isinstance(cfg, Mapping):
        dev = cfg.get("device")
    else:
        dev = getattr(cfg, "device", None)
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def _detect_local_rank(default: int = 0) -> int:
    """Best-effort local-rank detection across common launchers."""

    env_keys = (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
    )
    for key in env_keys:
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue

    if torch.cuda.is_available():
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                world_rank = dist.get_rank()
                device_count = max(1, torch.cuda.device_count())
                return world_rank % device_count
        except Exception:
            pass

    return default


def ensure_process_device(device: torch.device) -> torch.device:
    """Ensure each distributed rank uses a unique CUDA device.

    - If the device is CUDA without an explicit index, map it to the process's
      detected local rank and call ``torch.cuda.set_device``.
    - Returns the possibly-updated device so callers can keep references in sync.
    """

    if device.type != "cuda":
        return device

    index = device.index
    if index is None:
        index = _detect_local_rank()
        device = torch.device("cuda", index)

    try:
        torch.cuda.set_device(device)
    except Exception:
        # Fall back to best-effort assignment; subsequent CUDA ops will raise if invalid.
        pass
    return device


def get_lr(optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))
