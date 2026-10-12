from __future__ import annotations

import random

import pytest
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


def test_rng_state_restores_python_numpy_and_torch():
    np = pytest.importorskip("numpy")
    set_all_seeds(123)
    state = get_rng_state()
    expected = random.random(), np.random.random(), torch.rand(3)
    set_all_seeds(456)
    set_rng_state(state)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
