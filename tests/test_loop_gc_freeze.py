from __future__ import annotations

import gc
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import torch

from koochak.loop import training_loop
from koochak.utils import collector


def _run(tmp_path: Path, max_steps: int, on_step_end) -> None:
    model = torch.nn.Linear(1, 1, bias=False)

    def step_fn(module: torch.nn.Module, batch: Mapping[str, torch.Tensor], _ctx) -> dict[str, torch.Tensor]:
        return {"loss": torch.nn.functional.mse_loss(module(batch["x"]), batch["y"])}

    training_loop(
        model=model,
        dataset=[{"x": torch.ones(1, 1), "y": torch.zeros(1, 1)} for _ in range(max_steps)],
        step_fn=step_fn,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_cfg={"ddp": False, "device": "cpu", "max_steps": max_steps, "ckpt_every": 100,
                   "log_every": 100, "grad_clip_norm": None, "out_dir": str(tmp_path)},
        hooks={"on_step_end": [on_step_end]},
    )


class _RecordingGc:
    def __init__(self, events: list) -> None:
        self.events = events

    def collect(self) -> None:
        self.events.append("collect")

    def freeze(self) -> None:
        self.events.append("freeze")


def test_loop_freezes_after_warm_up_and_after_each_update_that_compiled(tmp_path, monkeypatch) -> None:
    events: list = []
    graphs = iter([5, 5, 6, 6])
    monkeypatch.setattr(collector, "gc", _RecordingGc(events))
    monkeypatch.setattr(collector, "compiled_graphs", lambda: next(graphs))
    _run(tmp_path, 5, lambda _log, ctx: events.append(int(ctx["step"])))
    assert events == [0, 1, "collect", "freeze", 2, 3, "collect", "freeze", 4]


def test_loop_freezes_the_real_collector_after_two_updates(tmp_path) -> None:
    before = gc.get_freeze_count()
    counts: list[int] = []
    _run(tmp_path, 3, lambda _log, _ctx: counts.append(gc.get_freeze_count()))
    assert counts[:2] == [before, before] and counts[2] > before


def test_compiled_graphs_reads_dynamo_counters_only_once_imported(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "torch._dynamo.utils", raising=False)
    assert collector.compiled_graphs() == 0
    counters = {"stats": {"unique_graphs": 7}}
    monkeypatch.setitem(sys.modules, "torch._dynamo.utils", SimpleNamespace(counters=counters))
    assert collector.compiled_graphs() == 7
