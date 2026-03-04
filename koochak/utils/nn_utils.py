from __future__ import annotations

from typing import Any

import torch.nn as nn


def prepare_param_groups_for_muon(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Build simple Muon/Adam parameter groups from a module.

    Parameters with ndim >= 2 are marked as Muon-compatible by default.
    """
    muon_params = []
    adam_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adam_params.append(p)

    groups: list[dict[str, Any]] = []
    if adam_params:
        groups.append(
            {
                "params": adam_params,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "use_muon": False,
            }
        )
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "use_muon": True,
            }
        )
    return groups
