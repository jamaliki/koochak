from __future__ import annotations

import torch
from torch import nn
import pytest

from scripts.diagnose_checkpoint_stability import (
    ActivationRecorder,
    _tensor_stats,
    parameter_statistics,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.gate = nn.Linear(2, 2)

    def forward(self, value):
        return self.block.gate(value)


def test_tensor_stats_reports_nonfinite_fraction() -> None:
    stats = _tensor_stats(torch.tensor([3.0, 4.0, float("inf")]))
    assert stats["rms"] == pytest.approx((12.5) ** 0.5)
    assert stats["abs_max"] == 4.0
    assert stats["finite_fraction"] == pytest.approx(2 / 3)


def test_parameter_statistics_groups_top_level_modules() -> None:
    model = TinyModel()
    stats = parameter_statistics(model)
    assert set(stats) == {"all", "top_level"}
    assert set(stats["top_level"]) == {"block"}
    assert stats["all"]["finite_fraction"] == 1.0


def test_activation_recorder_measures_gate_probabilities() -> None:
    model = TinyModel()
    recorder = ActivationRecorder(model)
    try:
        model(torch.ones(1, 2))
        row = recorder.snapshot()["block.gate"]
    finally:
        recorder.close()
    assert row["calls"] == 1
    assert 0.0 <= row["sigmoid_mean"] <= 1.0
    assert 0.0 <= row["saturated_low_fraction"] <= 1.0
    assert 0.0 <= row["saturated_high_fraction"] <= 1.0
