from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.pop("koochak", None)

from koochak.optim.muon import muon_update, normuon_update


def test_muon_updates_accept_noncontiguous_conv3d_gradients() -> None:
    grad = torch.randn(4, 3, 3, 3, 3).to(memory_format=torch.channels_last_3d)
    assert not grad.is_contiguous()

    momentum = torch.zeros_like(grad)
    update = muon_update(grad, momentum)

    assert update.shape == (4, 81)
    assert torch.isfinite(update).all()


def test_normuon_updates_accept_noncontiguous_conv3d_gradients() -> None:
    grad = torch.randn(4, 3, 3, 3, 3).to(memory_format=torch.channels_last_3d)
    assert not grad.is_contiguous()

    momentum = torch.zeros_like(grad)
    second_momentum = torch.zeros(4, 1, dtype=grad.dtype)
    update = normuon_update(grad, momentum, second_momentum)

    assert update.shape == (4, 81)
    assert torch.isfinite(update).all()
