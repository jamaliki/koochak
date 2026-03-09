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


def test_training_loop_unwraps_compile_wrap_methods_before_compile(monkeypatch, tmp_path: Path) -> None:
    from koochak.utils import flags

    class WrappedChild(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        @flags.compile_wrap
        def forward(self, x):
            return x + self.weight

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = WrappedChild()

        def forward(self, x):
            return self.child(x)

    events: list[tuple[str, object]] = []

    def fake_compile(module, **_kwargs):
        events.append(("compile", getattr(type(module.child), "forward")))
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)

    model = Parent()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = training_loop(
        model=model,
        dataset=[],
        step_fn=lambda *_args, **_kwargs: {"loss": torch.tensor(0.0, requires_grad=True)},
        optimizer=optimizer,
        train_cfg={
            "ddp": False,
            "compile": {"enabled": True},
            "max_steps": 0,
            "device": "cpu",
            "out_dir": str(tmp_path / "run"),
        },
    )

    compiled_forward = events[0][1]
    assert callable(compiled_forward)
    assert not isinstance(compiled_forward, flags.compile_wrap)
    assert result["step"] == 0
