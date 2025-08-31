from __future__ import annotations

import random
from typing import Any, Dict

import torch

__all__ = ["get_rng_state", "set_rng_state", "set_all_seeds", "make_worker_init_fn"]


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    try:
        import numpy as np  # type: ignore

        state["numpy"] = np.random.get_state()
    except Exception:
        state["numpy"] = None
    state.update(
        {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    )
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state previously produced by get_rng_state()."""
    try:
        import numpy as np  # type: ignore

        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        import torch

        if state.get("torch_cpu") is not None:
            torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])  # type: ignore[arg-type]
    except Exception:
        pass


def set_all_seeds(seed: int) -> None:
    """Seed python, numpy (if present), and torch for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Optional for strict determinism (may hurt performance)
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    except Exception:
        pass


def make_worker_init_fn(base_seed: int, *, rank: int = 0, worker_seed_offset: int = 1000):
    """Return a DataLoader worker_init_fn using base_seed, rank, and worker id.

    Each worker gets a unique deterministic seed:
      seed = base_seed + rank * worker_seed_offset + worker_id
    """
    def _fn(worker_id: int):
        s = int(base_seed) + int(rank) * int(worker_seed_offset) + int(worker_id)
        random.seed(s)
        try:
            import numpy as np  # type: ignore

            np.random.seed(s)
        except Exception:
            pass
        try:
            import torch

            torch.manual_seed(s)
        except Exception:
            pass

    return _fn
