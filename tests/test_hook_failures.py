from __future__ import annotations

import pytest
import torch

from koochak.loop import training_loop


def _loss_step_fn(*_args, **_kwargs):
    return {"loss": torch.tensor(0.0, requires_grad=True)}


def test_training_loop_raises_when_on_train_start_hook_fails(tmp_path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def fail_on_start(_ctx):
        raise RuntimeError("wandb init failed")

    with pytest.raises(RuntimeError, match="wandb init failed"):
        training_loop(
            model=model,
            dataset=[],
            step_fn=_loss_step_fn,
            optimizer=optimizer,
            train_cfg={
                "ddp": False,
                "max_steps": 0,
                "device": "cpu",
                "out_dir": str(tmp_path / "run"),
            },
            hooks={"on_train_start": [fail_on_start]},
        )


def test_training_loop_still_swallows_non_start_hook_failures(tmp_path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    steps_seen: list[int] = []

    def fail_on_log(_logs, ctx):
        steps_seen.append(int(ctx["step"]))
        raise RuntimeError("log hook failed")

    result = training_loop(
        model=model,
        dataset=[{"x": 0}],
        step_fn=_loss_step_fn,
        optimizer=optimizer,
        train_cfg={
            "ddp": False,
            "log_every": 1,
            "max_steps": 1,
            "device": "cpu",
            "out_dir": str(tmp_path / "run"),
        },
        hooks={"on_log": [fail_on_log]},
    )

    assert steps_seen == [0]
    assert result["step"] == 1
