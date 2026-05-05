from __future__ import annotations

import random
from typing import Any, Dict

import torch

try:
    import numpy as np
except ImportError:  # numpy is optional for bare Koochak installs.
    np = None  # type: ignore[assignment]

__all__ = ["get_rng_state", "set_rng_state", "set_all_seeds", "make_worker_init_fn"]


def get_rng_state() -> Dict[str, Any]:
    return {
        "numpy": np.random.get_state() if np is not None else None,
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state previously produced by get_rng_state()."""
    if np is not None and state.get("numpy") is not None:
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])  # type: ignore[arg-type]


def set_all_seeds(seed: int) -> None:
    """Seed python, numpy (if present), and torch for reproducibility."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def make_worker_init_fn(base_seed: int, *, rank: int = 0, worker_seed_offset: int = 1000):
    """Return a DataLoader worker_init_fn using base_seed, rank, and worker id.

    Each worker gets a unique deterministic seed:
      seed = base_seed + rank * worker_seed_offset + worker_id
    """
    def _fn(worker_id: int):
        s = int(base_seed) + int(rank) * int(worker_seed_offset) + int(worker_id)
        random.seed(s)
        if np is not None:
            np.random.seed(s)
        torch.manual_seed(s)

    return _fn
