from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

import torch

from . import dist as dist_lib

__all__ = ["initialize"]


def _as_int(value: Any, *, name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"[dist] Expected an integer for {name}, got {value!r}") from exc


def _get_first_env_int(keys: tuple[str, ...]) -> Optional[int]:
    for key in keys:
        if key in os.environ:
            return _as_int(os.environ.get(key), name=key)
    return None


def _extract_dist_cfg(config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    if "dist" in config and isinstance(config.get("dist"), Mapping):
        return config["dist"]  # type: ignore[index]
    if "distributed" in config and isinstance(config.get("distributed"), Mapping):
        return config["distributed"]  # type: ignore[index]
    return config


def _resolve_backend(dist_cfg: Mapping[str, Any]) -> str:
    backend = dist_cfg.get("backend") or os.environ.get("TORCH_DIST_BACKEND")
    if backend:
        return str(backend)
    return "nccl" if torch.cuda.is_available() else "gloo"


def _resolve_init_method(dist_cfg: Mapping[str, Any]) -> Optional[str]:
    init_method = dist_cfg.get("init_method") or os.environ.get("TORCH_DIST_INIT_METHOD")
    if init_method:
        return str(init_method)
    # Default to env:// when master addr/port are provided by launchers
    if os.environ.get("MASTER_ADDR") or os.environ.get("MASTER_PORT"):
        return "env://"
    return None


def _resolve_timeout(dist_cfg: Mapping[str, Any]) -> Optional[float]:
    timeout = dist_cfg.get("timeout") or os.environ.get("TORCH_DIST_TIMEOUT")
    if timeout is None:
        return None
    try:
        return float(timeout)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"[dist] Invalid timeout value: {timeout!r}") from exc


def _resolve_rank_info(dist_cfg: Mapping[str, Any]) -> tuple[Optional[int], Optional[int], Optional[int]]:
    rank = _as_int(dist_cfg.get("rank"), name="rank")
    world_size = _as_int(dist_cfg.get("world_size"), name="world_size")
    local_rank = _as_int(dist_cfg.get("local_rank"), name="local_rank")

    rank = rank or _get_first_env_int(("RANK", "SLURM_PROCID"))
    world_size = world_size or _get_first_env_int(("WORLD_SIZE", "SLURM_NTASKS"))
    local_rank = local_rank or _get_first_env_int(
        (
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "SLURM_LOCALID",
        )
    )
    return rank, world_size, local_rank


def _set_cuda_device(local_rank: Optional[int]) -> Optional[int]:
    if not torch.cuda.is_available():
        return None

    device_count = torch.cuda.device_count()
    if local_rank is not None:
        if local_rank < 0 or local_rank >= device_count:
            raise RuntimeError(
                f"[dist] local_rank={local_rank} is out of range for {device_count} CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        return local_rank

    # Fall back to current device, or default to 0 when CUDA is present
    try:
        current = torch.cuda.current_device()
    except Exception:
        current = 0
    torch.cuda.set_device(current)
    return int(current)


def initialize(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Initialize distributed process groups from config/env and return launch context.

    Args:
        config: Optional mapping with ``dist``/``distributed`` settings. Supported keys:
            ``backend``, ``init_method``, ``timeout``, ``rank``, ``world_size``,
            and ``local_rank``.

    Returns:
        Dict with ``rank``, ``world_size``, ``is_rank0``, and ``device`` (CUDA index or None).
    """

    dist_cfg = _extract_dist_cfg(config)
    backend = _resolve_backend(dist_cfg)
    init_method = _resolve_init_method(dist_cfg)
    timeout = _resolve_timeout(dist_cfg)
    rank, world_size, local_rank = _resolve_rank_info(dist_cfg)

    device_id = _set_cuda_device(local_rank)

    if dist_lib.is_initialized():
        rank = dist_lib.rank()
        world_size = dist_lib.world_size()
    else:
        dist_available = torch.distributed.is_available()
        should_init = any(
            [
                os.environ.get("MASTER_ADDR"),
                os.environ.get("MASTER_PORT"),
                rank is not None,
                world_size is not None,
            ]
        )

        if not dist_available and should_init:
            raise RuntimeError("[dist] torch.distributed requested but not available in this build")

        if dist_available and should_init:
            try:
                dist_lib.init_process_group(
                    backend=backend,
                    init_method=init_method,
                    timeout=timeout,
                    rank=rank,
                    world_size=world_size,
                )
            except Exception as exc:
                details = f"backend={backend}, init_method={init_method or 'default'}, rank={rank}, world_size={world_size}"
                raise RuntimeError(f"[dist] Failed to initialize process group ({details})") from exc

            rank = dist_lib.rank()
            world_size = dist_lib.world_size()

    # Fallback defaults when initialization was not requested
    rank = 0 if rank is None else int(rank)
    world_size = 1 if world_size is None else int(world_size)

    return {
        "rank": rank,
        "world_size": world_size,
        "is_rank0": rank == 0,
        "device": device_id,
    }

