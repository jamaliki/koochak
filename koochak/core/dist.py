from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "is_initialized",
    "init_process_group",
    "barrier",
    "rank",
    "world_size",
    "rank0",
]


def is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def init_process_group(backend: str = "nccl", init_method: Optional[str] = None, timeout: Optional[float] = None) -> None:
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed not available in this build")
    if is_initialized():
        return
    kwargs = {"backend": backend}
    if init_method is not None:
        kwargs["init_method"] = init_method
    if timeout is not None:
        import datetime

        kwargs["timeout"] = datetime.timedelta(seconds=float(timeout))
    torch.distributed.init_process_group(**kwargs)


def barrier() -> None:
    if is_initialized():
        torch.distributed.barrier()


def rank() -> int:
    if is_initialized():
        return torch.distributed.get_rank()
    return 0


def world_size() -> int:
    if is_initialized():
        return torch.distributed.get_world_size()
    return 1


def rank0() -> bool:
    return rank() == 0

def print0(string: str):
    if rank0():
        print(string)
