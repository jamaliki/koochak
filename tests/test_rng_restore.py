from __future__ import annotations

import torch

from koochak.utils.seed import get_rng_state, set_rng_state, set_all_seeds


def test_rng_state_roundtrip_restores_sequence():
    set_all_seeds(123)
    # Advance RNG and capture state
    a1 = torch.rand(3)
    state = get_rng_state()
    # Advance further
    _ = torch.rand(5)
    # Restore state
    set_rng_state(state)
    # Next draws should match what would have come next from saved point
    a2 = torch.rand(3)
    # Recompute expected from same state
    set_rng_state(state)
    expected = torch.rand(3)
    assert torch.allclose(a2, expected)

