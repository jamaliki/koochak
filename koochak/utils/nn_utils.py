from __future__ import annotations

from typing import Any

import torch.nn as nn


def tag_param(param: nn.Parameter, tag: str) -> nn.Parameter:
    tags = getattr(param, "_tags", None)
    if tags is None:
        param._tags = {tag}
    else:
        tags.add(tag)
    return param


def tag_module(module: nn.Module, tag: str) -> nn.Module:
    for param in module.parameters():
        tag_param(param, tag)
    return module


def apply_wd(module: nn.Module) -> nn.Module:
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


def prepare_param_groups_for_muon(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Build Muon/Adam parameter groups using parameter tags.

    Uses tags attached via ``tag_module()`` / ``apply_wd()``:
    - ``wd``: parameter should receive weight decay.
    - ``mapping``: exclude from Muon even if it is 2D+ and weight-decayed.
    """
    def has_tag(param: nn.Parameter, tag: str) -> bool:
        return tag in getattr(param, "_tags", set())

    muon_params = []
    adam_wd = []
    adam_nowd = []

    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim >= 2
        has_wd = has_tag(p, "wd")
        is_mapping = has_tag(p, "mapping")

        if is_matrix and has_wd and not is_mapping:
            muon_params.append(p)
        elif has_wd:
            adam_wd.append(p)
        else:
            adam_nowd.append(p)

    groups: list[dict[str, Any]] = []
    if adam_nowd:
        groups.append(
            {
                "params": adam_nowd,
                "betas": (0.9, 0.95),
                "eps": 1e-10,
                "lr": float(lr),
                "weight_decay": 0.0,
                "use_muon": False,
            }
        )
    if adam_wd:
        groups.append(
            {
                "params": adam_wd,
                "betas": (0.9, 0.95),
                "eps": 1e-10,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "use_muon": False,
            }
        )
    if muon_params:
        groups.append(
            {
                "params": sorted(muon_params, key=lambda x: x.size(), reverse=True),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "momentum": 0.95,
                "use_muon": True,
            }
        )
    return groups
