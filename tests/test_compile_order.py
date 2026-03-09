from __future__ import annotations

from pathlib import Path

import torch

from koochak.loop import training_loop


class _DummyDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, **_kwargs):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_training_loop_compiles_before_ddp(monkeypatch, tmp_path: Path) -> None:
    events: list[tuple[str, str]] = []

    def fake_compile(module, **_kwargs):
        events.append(("compile", type(module).__name__))
        return module

    def fake_ddp(module, **_kwargs):
        events.append(("ddp", type(module).__name__))
        return _DummyDDP(module)

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", fake_ddp)

    from koochak import loop as loop_mod

    monkeypatch.setattr(loop_mod.dist_lib, "world_size", lambda: 2)
    monkeypatch.setattr(loop_mod.dist_lib, "rank", lambda: 0)
    monkeypatch.setattr(loop_mod.dist_lib, "rank0", lambda: True)
    monkeypatch.setattr(loop_mod.dist_lib, "is_initialized", lambda: True)

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = training_loop(
        model=model,
        dataset=[],
        step_fn=lambda *_args, **_kwargs: {"loss": torch.tensor(0.0, requires_grad=True)},
        optimizer=optimizer,
        train_cfg={
            "ddp": True,
            "compile": {"enabled": True},
            "max_steps": 0,
            "device": "cpu",
            "out_dir": str(tmp_path / "run"),
        },
    )

    assert events == [("compile", "Linear"), ("ddp", "Linear")]
    assert result["step"] == 0
