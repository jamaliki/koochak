from __future__ import annotations

import random
from typing import Any, Dict

import torch

__all__ = ["get_rng_state"]


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

