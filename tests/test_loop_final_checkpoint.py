from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

from koochak.loop import training_loop


def _run(
    tmp_path: Path,
    *,
    max_steps: int,
    checkpoint: dict[str, Any] | None = None,
    ckpt_every: int = 100,
) -> tuple[dict[str, Any], torch.nn.Module, list[int]]:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    steps: list[int] = []

    def step_fn(
        module: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        _ctx: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        return {"loss": torch.nn.functional.mse_loss(module(batch["x"]), batch["y"])}

    def record_step(_logs: Mapping[str, object], ctx: Mapping[str, object]) -> None:
        steps.append(int(ctx["step"]))

    data = [{"x": torch.ones(1, 1), "y": torch.zeros(1, 1)} for _ in range(max_steps + 1)]
    result = training_loop(
        model=model,
        dataset=data,
        step_fn=step_fn,
        optimizer=optimizer,
        train_cfg={
            "ddp": False,
            "device": "cpu",
            "max_steps": max_steps,
            "ckpt_every": ckpt_every,
            "log_every": 100,
            "grad_clip_norm": None,
            "out_dir": str(tmp_path),
        },
        checkpoint_dict=checkpoint,
        hooks={"on_step_end": [record_step]},
    )
    return result, model, steps


def test_terminal_checkpoint_resumes_at_the_next_unexecuted_step(tmp_path: Path) -> None:
    first, _first_model, first_steps = _run(tmp_path / "first", max_steps=1)
    assert first_steps == [0]
    assert first["step"] == 1
    assert first["next_step"] == 1

    resumed, resumed_model, resumed_steps = _run(
        tmp_path / "resumed",
        max_steps=2,
        checkpoint=first,
    )

    assert resumed_steps == [1]
    assert resumed["step"] == 2
    assert resumed["next_step"] == 2
    torch.testing.assert_close(resumed["model"]["weight"], resumed_model.state_dict()["weight"])
    assert not torch.equal(resumed["model"]["weight"], first["model"]["weight"])


def test_legacy_terminal_checkpoint_infers_completed_step_count(tmp_path: Path) -> None:
    first, _model, _steps = _run(tmp_path / "first", max_steps=1)
    legacy = deepcopy(first)
    legacy.pop("next_step")

    resumed, _resumed_model, resumed_steps = _run(
        tmp_path / "resumed",
        max_steps=2,
        checkpoint=legacy,
    )

    assert resumed_steps == [1]
    assert resumed["next_step"] == 2


def test_terminal_checkpoint_supersedes_earlier_periodic_state(tmp_path: Path) -> None:
    result, model, steps = _run(tmp_path, max_steps=4, ckpt_every=2)

    assert steps == [0, 1, 2, 3]
    assert result["step"] == 4
    assert result["next_step"] == 4
    torch.testing.assert_close(result["model"]["weight"], model.state_dict()["weight"])
