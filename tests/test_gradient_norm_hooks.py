from __future__ import annotations

import torch

from koochak.loop import training_loop


def test_pre_clip_hook_reuses_clipping_norm(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    seen: list[float] = []

    def step_fn(model, batch, _ctx):
        return {"loss": model(batch).square().mean()}

    def on_pre_clip(payload, _ctx):
        seen.append(float(payload["global_norm"]))

    training_loop(
        model=model,
        dataset=[torch.ones(4, 2), torch.ones(4, 2)],
        step_fn=step_fn,
        optimizer=optimizer,
        train_cfg={
            "max_steps": 2,
            "device": "cpu",
            "amp": "fp32",
            "grad_clip_norm": 0.1,
            "log_every": 100,
            "out_dir": str(tmp_path / "run"),
        },
        hooks={"on_pre_clip": [on_pre_clip]},
    )

    assert len(seen) == 2
    assert all(value > 0 for value in seen)
