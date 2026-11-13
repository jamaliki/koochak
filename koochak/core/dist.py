from __future__ import annotations

import datetime
import os
from typing import Optional

import torch

__all__ = [
    "is_initialized",
    "init_process_group",
    "barrier",
    "rank",
    "world_size",
    "rank0",
    "print0",
]


def is_initialized() -> bool:
    try:
        _dist_available = torch.distributed.is_available() and torch.distributed.is_initialized()
    except ImportError:
        _dist_available = False
    return _dist_available


def torchelastic_attempt_store(timeout: datetime.timedelta) -> Optional["torch.distributed.Store"]:
    """Return torchrun's worker store under a per-attempt prefix, or ``None`` outside torchrun.

    torchrun (at least through torch 2.15) keeps one worker store for the whole
    launch, and ``env://`` connects to it without a per-attempt prefix.  After a
    ``--max-restarts`` restart the new workers can then read the previous
    attempt's NCCL or Gloo rendezvous keys and dial peers that no longer exist.
    """

    if os.environ.get("TORCHELASTIC_USE_AGENT_STORE") != "True":
        return None
    store = torch.distributed.TCPStore(
        os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), is_master=False, timeout=timeout
    )
    return torch.distributed.PrefixStore(
        f"/koochak/attempt_{os.environ['TORCHELASTIC_RESTART_COUNT']}", store
    )


def init_process_group(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    timeout: Optional[float] = None,
    *,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> None:
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed not available in this build")
    if is_initialized():
        return
    kwargs = {"backend": backend}
    if timeout is not None:
        kwargs["timeout"] = datetime.timedelta(seconds=float(timeout))
    store = None
    if init_method in (None, "env://"):
        from torch.distributed.constants import default_pg_timeout

        store = torchelastic_attempt_store(kwargs.get("timeout", default_pg_timeout))
    if store is not None:
        kwargs["store"] = store
        rank = int(os.environ["RANK"]) if rank is None else rank
        world_size = int(os.environ["WORLD_SIZE"]) if world_size is None else world_size
    elif init_method is not None:
        kwargs["init_method"] = init_method
    if rank is not None:
        kwargs["rank"] = int(rank)
    if world_size is not None:
        kwargs["world_size"] = int(world_size)
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

def print0(string: str) -> None:
    if rank0():
        print(string)
