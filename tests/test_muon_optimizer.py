from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.pop("koochak", None)

from koochak.optim.build import build_optimizer
from koochak.optim.muon import muon_update, normuon_update
from koochak.utils.nn_utils import apply_wd


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


def test_muon_builder_forwards_foreach_adam_update_to_aux_adam_groups() -> None:
    model = apply_wd(torch.nn.Linear(4, 4))

    optimizer = build_optimizer(model, {"name": "Muon", "foreach_adam_update": True})

    adam_groups = [group for group in optimizer.param_groups if not group.get("use_muon", False)]
    muon_groups = [group for group in optimizer.param_groups if group.get("use_muon", False)]
    assert adam_groups
    assert muon_groups
    assert all(group["foreach_adam_update"] is True for group in adam_groups)
    assert all("foreach_adam_update" not in group for group in muon_groups)


def test_muon_builder_parses_string_false_for_foreach_adam_update() -> None:
    model = apply_wd(torch.nn.Linear(4, 4))

    optimizer = build_optimizer(model, {"name": "Muon", "foreach_adam_update": "false"})

    adam_groups = [group for group in optimizer.param_groups if not group.get("use_muon", False)]
    assert adam_groups
    assert all(group["foreach_adam_update"] is False for group in adam_groups)
