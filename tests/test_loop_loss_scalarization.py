from __future__ import annotations

from pathlib import Path

import torch

from koochak.loop import training_loop


def _run_scalarization_case(tmp_path: Path, *, scalarize_loss_every_step: bool):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    dataset = [{"x": torch.ones(1, 1), "y": torch.zeros(1, 1)} for _ in range(3)]
    step_end_losses = []
    log_losses = []

    def step_fn(module, batch, _ctx):
        pred = module(batch["x"])
        return {"loss": torch.nn.functional.mse_loss(pred, batch["y"])}

    def record_step(row, _ctx):
        step_end_losses.append((int(row["step"]), row["loss"]))

    def record_log(row, _ctx):
        log_losses.append((int(row["step"]), row["loss"]))

    training_loop(
        model=model,
        dataset=dataset,
        step_fn=step_fn,
        optimizer=optimizer,
        train_cfg={
            "max_steps": len(dataset),
            "log_every": 2,
            "eval_every": 1000,
            "ckpt_every": 1000,
            "grad_clip_norm": None,
            "device": "cpu",
            "out_dir": str(tmp_path / "run"),
            "scalarize_loss_every_step": scalarize_loss_every_step,
        },
        hooks={"on_step_end": [record_step], "on_log": [record_log]},
    )
    return step_end_losses, log_losses


def test_training_loop_loss_is_scalar_every_step_by_default(tmp_path):
    step_end_losses, log_losses = _run_scalarization_case(
        tmp_path,
        scalarize_loss_every_step=True,
    )

    assert step_end_losses
    assert all(isinstance(loss, float) for _step, loss in step_end_losses)
    assert log_losses
    assert all(isinstance(loss, float) for _step, loss in log_losses)


def test_training_loop_can_defer_non_log_loss_scalarization(tmp_path):
    step_end_losses, log_losses = _run_scalarization_case(
        tmp_path,
        scalarize_loss_every_step=False,
    )

    log_steps = {step for step, _loss in log_losses}
    non_log_losses = []
    for step, loss in step_end_losses:
        if step in log_steps:
            assert isinstance(loss, float)
        else:
            non_log_losses.append(loss)
            assert isinstance(loss, torch.Tensor)
            assert not loss.requires_grad
    assert non_log_losses
    assert log_losses
    assert all(isinstance(loss, float) for _step, loss in log_losses)
