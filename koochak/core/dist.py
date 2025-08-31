from __future__ import annotations

import torch

__all__ = ["rank", "world_size", "is_initialized"]


def is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    if is_initialized():
        return torch.distributed.get_rank()
    return 0


def world_size() -> int:
    if is_initialized():
        return torch.distributed.get_world_size()
    return 1

