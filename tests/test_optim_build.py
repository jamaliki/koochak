from __future__ import annotations

import torch

from koochak.optim.build import build_scheduler


def test_warmup_constant_scheduler_ramps_then_holds() -> None:
    """Linear warmup reaches max LR at warmup_steps and does not anneal after."""

    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.AdamW([param], lr=3e-4)
    scheduler = build_scheduler(
        opt,
        {"name": "warmup_constant", "warmup_steps": 2000},
        {"max_steps": 50001},
    )

    assert scheduler is not None
    assert opt.param_groups[0]["lr"] == 0.0

    for _ in range(1000):
        opt.step()
        scheduler.step()
    assert abs(opt.param_groups[0]["lr"] - 1.5e-4) < 1e-12

    for _ in range(1000):
        opt.step()
        scheduler.step()
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12

    for _ in range(100):
        opt.step()
        scheduler.step()
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12
