from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from koochak.loop import (
    _StartupProgressWatchdog,
    _TrainSettings,
    _ThroughputGuard,
    training_loop,
)
from koochak.storage import checkpoint as checkpoint_lib


def _cfg(**overrides):
    cfg = {
        "throughput_guard_enabled": True,
        "throughput_guard_warmup_steps": 2,
        "throughput_guard_window_steps": 2,
        "throughput_guard_max_median_step_time_s": 1.0,
        "throughput_guard_max_median_batch_wait_s": 0.5,
        "throughput_guard_bad_windows": 1,
    }
    cfg.update(overrides)
    return cfg


def test_guard_is_launch_relative_even_for_a_resumed_global_step():
    guard = _ThroughputGuard(_TrainSettings.from_cfg(_cfg()), rank=0, world_size=1)

    assert guard.observe(step=100_000, step_time_s=10.0, batch_wait_s=1.0) is None
    assert guard.observe(step=100_001, step_time_s=10.0, batch_wait_s=1.0) is None
    failure = guard.observe(step=100_002, step_time_s=10.0, batch_wait_s=1.0)
    assert failure is None
    failure = guard.observe(step=100_003, step_time_s=10.0, batch_wait_s=1.0)

    assert failure is not None
    assert failure["launch_steps_seen"] == 4
    assert failure["global_step"] == 100_003


def test_guard_is_disabled_by_default():
    settings = _TrainSettings.from_cfg({})
    guard = _ThroughputGuard(settings, rank=0, world_size=1)

    assert settings.throughput_guard_enabled is False
    assert guard.observe(step=0, step_time_s=10.0, batch_wait_s=10.0) is None


def test_guard_settings_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        _TrainSettings.from_cfg(_cfg(throughput_guard_window_steps=0))


def test_startup_watchdog_timeout_callback_is_injectable_and_race_safe(tmp_path: Path):
    failures = []
    watchdog = _StartupProgressWatchdog(
        timeout_s=300.0,
        out_dir=str(tmp_path),
        rank=2,
        world_size=4,
        timeout_callback=failures.append,
    )
    watchdog.arm(120_000)
    watchdog.trigger_timeout_for_test()
    watchdog.disarm()

    assert failures == [
        {
            "event": "startup_progress_timeout",
            "rank": 2,
            "world_size": 4,
            "timeout_s": 300.0,
            "starting_step": 120_000,
            "last_step": None,
            "pid": failures[0]["pid"],
            "timestamp": failures[0]["timestamp"],
        }
    ]

    completed = []
    watchdog = _StartupProgressWatchdog(
        timeout_s=300.0,
        out_dir=str(tmp_path),
        rank=0,
        world_size=1,
        timeout_callback=completed.append,
    )
    watchdog.arm(10)
    watchdog.mark_first_step_complete(10)
    watchdog.trigger_timeout_for_test()
    watchdog.disarm()
    assert completed == []


def test_startup_watchdog_timeout_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        _TrainSettings.from_cfg({"startup_progress_timeout_s": -1})


def test_startup_watchdog_exits_when_diagnostic_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    watchdog = _StartupProgressWatchdog(
        timeout_s=300.0,
        out_dir=str(tmp_path),
        rank=0,
        world_size=4,
    )
    exits = []

    def fail_to_create_summary(*_args, **_kwargs):
        raise OSError("filesystem unavailable")

    loop_module = sys.modules[_StartupProgressWatchdog.__module__]
    monkeypatch.setattr(loop_module.faulthandler, "dump_traceback", lambda **_kwargs: None)
    monkeypatch.setattr(loop_module.os, "makedirs", fail_to_create_summary)
    monkeypatch.setattr(loop_module.os, "_exit", exits.append)

    watchdog._default_timeout_callback({"event": "startup_progress_timeout"})

    assert exits == [watchdog.exit_code]


def test_training_loop_writes_resumable_throughput_failure_artifacts(tmp_path: Path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    def step_fn(module, batch, _ctx):
        return {"loss": module(batch).square().sum()}

    with pytest.raises(SystemExit) as exc:
        training_loop(
            model=model,
            dataset=[torch.ones(1, 1) for _ in range(8)],
            step_fn=step_fn,
            optimizer=optimizer,
            train_cfg={
                "device": "cpu",
                "max_steps": 8,
                "log_every": 1000,
                "eval_every": 1000,
                "ckpt_every": 1000,
                "out_dir": str(tmp_path),
                "keep_last_k": 3,
                **_cfg(
                    throughput_guard_warmup_steps=1,
                    throughput_guard_window_steps=2,
                    throughput_guard_max_median_step_time_s=1e-12,
                ),
            },
        )

    assert exc.value.code == 86
    checkpoint_path = tmp_path / "step000000002.pt"
    summary_path = tmp_path / "throughput_guard" / "throughput_guard_failure_step000000002.json"
    assert checkpoint_path.exists()
    assert summary_path.exists()
    assert (tmp_path / "latest.pt").exists()

    checkpoint = checkpoint_lib.load(str(checkpoint_path))
    assert checkpoint["step"] == 2
    assert checkpoint["metrics"]["throughput_guard"]["global_step"] == 2
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["checkpoint_path"] == str(checkpoint_path)
    assert len(summary["rank_metrics"]) == 1
    assert summary["rank_metrics"][0]["rank"] == 0
    assert summary["rank_metrics"][0]["median_batch_wait_s"] >= 0.0
