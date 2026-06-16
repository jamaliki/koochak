from __future__ import annotations


def test_eval_at_step_zero_false_skips_initial_eval(tmp_path):
    import torch

    from koochak.loop import training_loop

    class _InfiniteOnes:
        def __iter__(self):
            while True:
                yield torch.ones(1, 1)

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    eval_steps: list[int] = []

    def step_fn(model, batch, ctx):
        del ctx
        return {"loss": model(batch).square().sum()}

    def eval_fn(model, dataset, ctx):
        del model, dataset
        eval_steps.append(int(ctx["step"]))
        return {"val_loss": 0.0}

    training_loop(
        model=model,
        dataset=_InfiniteOnes(),
        step_fn=step_fn,
        optimizer=optimizer,
        train_cfg={
            "device": "cpu",
            "max_steps": 2,
            "log_every": 1000,
            "eval_every": 1,
            "eval_at_step_zero": False,
            "ckpt_every": 1000,
            "amp": "fp32",
            "out_dir": str(tmp_path),
        },
        eval_dataset=[torch.ones(1, 1)],
        eval_fn=eval_fn,
    )

    assert eval_steps == [1]
