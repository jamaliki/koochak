from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.pop("koochak", None)

from koochak.loop import training_loop


class _FiniteLossNonfiniteGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.shape = value.shape
        return value.sum() * 0.0 + 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.full(ctx.shape, float("nan"), device=grad_output.device),)


def test_training_loop_skips_nonfinite_gradient_step_before_clip_poisoning(tmp_path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    logged: list[dict[str, float]] = []

    def step_fn(model, _batch, _ctx):
        return {"loss": _FiniteLossNonfiniteGrad.apply(model.weight)}

    def on_log(logs, _ctx):
        logged.append(dict(logs))

    result = training_loop(
        model=model,
        dataset=[{"x": 0}],
        step_fn=step_fn,
        optimizer=optimizer,
        train_cfg={
            "ddp": False,
            "log_every": 1,
            "max_steps": 1,
            "device": "cpu",
            "grad_clip_norm": 1.0,
            "nonfinite_grad_check_every": 0,
            "out_dir": str(tmp_path / "run"),
        },
        hooks={"on_log": [on_log]},
    )

    assert result["step"] == 1
    assert torch.isfinite(model.weight).all()
    assert model.weight.item() == 2.0
    assert logged
    assert logged[0]["grad_step_skipped"] == 1.0
    assert logged[0]["nonfinite_grad_tensor_count"] == 1.0
    assert math.isnan(logged[0]["grad_clip_total_norm"])
