from __future__ import annotations

import pytest
import torch

from scripts.audit_activation_spikes import _sigma_grid, _synthetic_input


def test_sigma_grid_is_positive_log_uniform() -> None:
    values = _sigma_grid(0.01, 1000.0, 6)
    assert values[0] == pytest.approx(0.01)
    assert values[-1] == pytest.approx(1000.0)
    ratios = [right / left for left, right in zip(values, values[1:])]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_synthetic_input_is_atom14_and_self_conditioning_off() -> None:
    inputs = _synthetic_input(
        3.0, device=torch.device("cpu"), residues=32, seed=7
    )
    assert inputs.coordinates.shape == (1, 32, 14, 3)
    assert inputs.sigma.shape == (1, 32, 14)
    assert inputs.atom_mask.all()
    assert inputs.self_conditioning_mask is not None
    assert not inputs.self_conditioning_mask.any()
    assert inputs.self_conditioned_coordinates is not None
    assert torch.count_nonzero(inputs.self_conditioned_coordinates) == 0
