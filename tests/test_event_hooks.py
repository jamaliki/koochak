from __future__ import annotations

import builtins
import sys
import types
from unittest import mock

import pytest
import torch

from koochak.cli.train import _maybe_add_scruffy_hooks
from koochak.core import hooks as hooks_lib
from koochak.logging.events import make_event_hooks, make_scruffy_hooks


def _call(hooks, name, *args):
    assert len(hooks[name]) == 1
    hooks[name][0](*args)


def test_event_hooks_publish_bounded_lifecycle_records_without_large_inputs() -> None:
    published = []
    times = iter((0.0, 5.0, 31.0, 32.0))
    hooks = make_event_hooks(
        lambda kind, data: published.append((kind, data)),
        progress_interval_s=30,
        clock=lambda: next(times),
    )
    ctx = {
        "device": torch.device("cpu"),
        "rank": 0,
        "world_size": 4,
        "train_cfg": {"max_steps": 10},
        "config_json": {"secret": "must not be published"},
    }

    _call(hooks, "on_train_start", ctx)
    metrics = {
        **{f"metric_{index:02d}": float(index) for index in range(40)},
        "step": 0,
        "loss": torch.tensor(2.5),
        "lr": 0.001,
        "not_scalar": torch.ones(2),
        "text": "not a metric",
        "nan": float("nan"),
    }
    _call(hooks, "on_step_end", metrics, {**ctx, "step": 0})
    _call(hooks, "on_step_end", {"loss": 2.0}, {**ctx, "step": 1})
    _call(hooks, "on_step_end", {"loss": 1.5}, {**ctx, "step": 2})
    _call(hooks, "on_eval_end", {"val_loss": torch.tensor(1.25)}, {**ctx, "step": 3})

    class UnreadableCheckpoint(dict):
        def __iter__(self):  # pragma: no cover - fails the test if contents are inspected
            raise AssertionError("checkpoint contents were inspected")

    _call(
        hooks,
        "on_checkpoint",
        "/tmp/checkpoint.pt",
        UnreadableCheckpoint(secret="large state"),
        {**ctx, "step": 3},
    )
    _call(hooks, "on_step_end", {"loss": 1.0}, {**ctx, "step": 9})
    _call(hooks, "on_train_end", ctx)

    assert [kind for kind, _ in published] == [
        "workload.phase",
        "workload.progress",
        "workload.progress",
        "workload.milestone",
        "workload.artifact",
        "workload.phase",
    ]
    assert all("config" not in data and "config_json" not in data for _, data in published)
    first_progress = published[1][1]
    assert first_progress["step"] == 0
    assert first_progress["completed"] == 1
    assert first_progress["total"] == 10
    assert first_progress["unit"] == "steps"
    assert first_progress["metrics"]["loss"] == 2.5
    assert first_progress["metrics"]["lr"] == 0.001
    assert len(first_progress["metrics"]) == 32
    assert "not_scalar" not in first_progress["metrics"]
    assert "text" not in first_progress["metrics"]
    assert "nan" not in first_progress["metrics"]
    assert published[-2][1] == {
        "artifact_type": "checkpoint",
        "location": "/tmp/checkpoint.pt",
        "step": 3,
    }
    assert published[-1][1]["step"] == 9
    assert "on_log" not in hooks


def test_exception_is_bounded_and_publisher_failures_never_escape_hooks() -> None:
    attempts = []

    def fail(kind, data):
        attempts.append((kind, data))
        raise OSError("coordinator unavailable")

    hooks = make_event_hooks(fail, progress_interval_s=0, clock=lambda: 1.0)
    ctx = {"rank": 0, "world_size": 1, "step": 7}

    with pytest.warns(RuntimeWarning, match="training will continue") as warnings_seen:
        hooks_lib.emit(hooks, "on_train_start", ctx, suppress_exceptions=False)
        hooks_lib.emit(hooks, "on_step_end", {"loss": 1.0}, ctx)
        hooks_lib.emit(hooks, "on_eval_end", {"val_loss": 1.0}, ctx)
        hooks_lib.emit(hooks, "on_checkpoint", "/tmp/x.pt", {"model": object()}, ctx)
        hooks_lib.emit(hooks, "on_train_end", ctx)
        hooks_lib.emit(hooks, "on_exception", RuntimeError("x" * 1000), ctx)

    assert len(warnings_seen) == 1
    assert len(attempts) == 6
    failed = attempts[-1]
    assert failed[0] == "workload.phase"
    assert failed[1]["status"] == "failed"
    assert failed[1]["error_type"] == "RuntimeError"
    assert len(failed[1]["message"]) == 512

    huge_exception = type("E" * 1000, (Exception,), {})
    working = make_event_hooks(
        lambda kind, data: attempts.append((kind, data)),
        clock=lambda: 1.0,
    )
    _call(working, "on_exception", huge_exception("bad"), ctx)
    assert len(attempts[-1][1]["error_type"]) == 128


def test_hooks_are_rank_zero_only() -> None:
    published = []
    hooks = make_event_hooks(lambda kind, data: published.append((kind, data)))

    with mock.patch("koochak.core.dist.rank0", return_value=False):
        _call(hooks, "on_train_start", {"rank": 1, "world_size": 2})

    assert published == []


def test_scruffy_adapter_imports_lazily_and_supplies_worker_identity(monkeypatch) -> None:
    monkeypatch.setenv("SCRUFFY_ROOT", "/shared/scruffy")
    monkeypatch.setenv("SCRUFFY_JOB_ID", "job-123")
    monkeypatch.setenv("SCRUFFY_NODE", "gpu-4")
    real_import = builtins.__import__

    def reject_scruffy(name, *args, **kwargs):
        if name == "scruffy":
            raise AssertionError("Scruffy imported while hooks were only constructed")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=reject_scruffy):
        hooks = make_scruffy_hooks()

    publish_event = mock.Mock(return_value={"event_id": "evt-1"})
    module = types.ModuleType("scruffy")
    module.publish_event = publish_event
    monkeypatch.setitem(sys.modules, "scruffy", module)
    _call(hooks, "on_train_start", {"rank": 0, "world_size": 1})

    publish_event.assert_called_once_with(
        mock.ANY,
        job_id="job-123",
        kind="workload.phase",
        data={"phase": "training", "status": "started", "world_size": 1},
        source={"name": "koochak", "node": "gpu-4"},
    )
    assert str(publish_event.call_args.args[0]) == "/shared/scruffy"


def test_generic_cli_adds_scruffy_hooks_only_with_complete_identity(monkeypatch) -> None:
    base = {"on_step_end": [lambda *_args: None]}
    extra = {
        "on_step_end": [lambda *_args: None],
        "on_train_end": [lambda *_args: None],
    }

    with mock.patch("koochak.cli.train.make_scruffy_hooks", return_value=extra) as factory:
        monkeypatch.delenv("SCRUFFY_ROOT", raising=False)
        monkeypatch.delenv("SCRUFFY_JOB_ID", raising=False)
        assert _maybe_add_scruffy_hooks(base) is base
        monkeypatch.setenv("SCRUFFY_ROOT", "/shared/scruffy")
        assert _maybe_add_scruffy_hooks(base) is base
        monkeypatch.setenv("SCRUFFY_JOB_ID", "job-123")
        merged = _maybe_add_scruffy_hooks(base)

    factory.assert_called_once_with()
    assert len(merged["on_step_end"]) == 2
    assert len(merged["on_train_end"]) == 1


def test_scruffy_adapter_omits_an_unset_optional_node(monkeypatch) -> None:
    monkeypatch.setenv("SCRUFFY_ROOT", "/shared/scruffy")
    monkeypatch.setenv("SCRUFFY_JOB_ID", "job-123")
    monkeypatch.delenv("SCRUFFY_NODE", raising=False)
    publish_event = mock.Mock(return_value={"event_id": "evt-1"})
    module = types.ModuleType("scruffy")
    module.publish_event = publish_event
    monkeypatch.setitem(sys.modules, "scruffy", module)

    _call(make_scruffy_hooks(), "on_train_start", {"rank": 0, "world_size": 1})

    assert publish_event.call_args.kwargs["source"] == {"name": "koochak"}


def test_checkpoint_publication_uses_a_stable_scruffy_event_id(
    monkeypatch, tmp_path
) -> None:
    checkpoint_path = tmp_path / "step000000007.pt"
    ckpt = {"step": 7, "model": {}}
    from koochak.storage import checkpoint as checkpoint_lib

    checkpoint_lib.save(ckpt, str(checkpoint_path))
    monkeypatch.setenv("SCRUFFY_ROOT", "/shared/scruffy")
    monkeypatch.setenv("SCRUFFY_JOB_ID", "job-123")
    publish_event = mock.Mock(return_value={"event_id": "evt-1"})
    module = types.ModuleType("scruffy")
    module.publish_event = publish_event
    monkeypatch.setitem(sys.modules, "scruffy", module)

    hooks = make_scruffy_hooks()
    ctx = {"rank": 0, "world_size": 1, "step": 7}
    _call(hooks, "on_checkpoint", str(checkpoint_path), ckpt, ctx)
    first = publish_event.call_args.kwargs
    _call(hooks, "on_checkpoint", str(checkpoint_path), ckpt, ctx)
    second = publish_event.call_args.kwargs

    assert first["event_id"] == second["event_id"]
    assert first["event_id"].startswith("koochak-checkpoint-")
    assert "wait" not in first
    assert "timeout" not in first
    assert first["data"]["publication"] == checkpoint_lib.publication(
        str(checkpoint_path)
    )


def test_scruffy_checkpoint_ack_wait_is_opt_in_and_checkpoint_only(
    monkeypatch, tmp_path
) -> None:
    from koochak.storage import checkpoint as checkpoint_lib

    checkpoint_path = tmp_path / "step000000007.pt"
    checkpoint = {"step": 7, "model": {}}
    checkpoint_lib.save(checkpoint, str(checkpoint_path))
    monkeypatch.setenv("SCRUFFY_ROOT", "/shared/scruffy")
    monkeypatch.setenv("SCRUFFY_JOB_ID", "job-123")
    calls = []

    def publish_event(root, **values):
        calls.append((root, values))
        return {
            "state": "accepted" if values.get("wait") else "spooled",
            "acknowledged": bool(values.get("wait")),
        }

    module = types.ModuleType("scruffy")
    module.publish_event = publish_event
    monkeypatch.setitem(sys.modules, "scruffy", module)

    hooks = make_scruffy_hooks(progress_interval_s=0, artifact_ack_timeout_s=4.5)
    _call(hooks, "on_train_start", {"rank": 0, "world_size": 1})
    _call(
        hooks,
        "on_checkpoint",
        str(checkpoint_path),
        checkpoint,
        {"rank": 0, "world_size": 1, "step": 7},
    )
    _call(
        hooks,
        "on_evacuation",
        str(checkpoint_path),
        checkpoint,
        {"rank": 0, "world_size": 1, "step": 7},
    )

    assert calls[0][1].get("wait") is None
    assert calls[0][1].get("timeout") is None
    assert calls[1][1]["wait"] is True
    assert calls[1][1]["timeout"] == 4.5
    assert calls[2][1].get("wait") is None
    assert calls[2][1].get("timeout") is None
    assert calls[1][1]["event_id"].startswith("koochak-checkpoint-")
    assert calls[1][1]["data"]["publication"] == checkpoint_lib.publication(
        str(checkpoint_path)
    )


def test_scruffy_checkpoint_ack_timeout_uses_environment_and_explicit_wins(
    monkeypatch, tmp_path
) -> None:
    from koochak.storage import checkpoint as checkpoint_lib

    checkpoint_path = tmp_path / "step000000001.pt"
    checkpoint = {"step": 1, "model": {}}
    checkpoint_lib.save(checkpoint, str(checkpoint_path))
    monkeypatch.setenv("SCRUFFY_ROOT", "/shared/scruffy")
    monkeypatch.setenv("SCRUFFY_JOB_ID", "job-123")
    monkeypatch.setenv("KOOCHAK_SCRUFFY_ARTIFACT_ACK_TIMEOUT_SECONDS", "30")
    calls = []

    def publish_event(_root, **values):
        calls.append(values)
        return {"state": "accepted", "acknowledged": True}

    module = types.ModuleType("scruffy")
    module.publish_event = publish_event
    monkeypatch.setitem(sys.modules, "scruffy", module)

    hooks = make_scruffy_hooks()
    _call(hooks, "on_checkpoint", str(checkpoint_path), checkpoint, {"step": 1})
    assert calls[-1]["timeout"] == 30.0

    calls.clear()
    hooks = make_scruffy_hooks(artifact_ack_timeout_s=2)
    _call(hooks, "on_checkpoint", str(checkpoint_path), checkpoint, {"step": 1})
    assert calls[-1]["timeout"] == 2.0


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "not-a-timeout"])
def test_scruffy_checkpoint_ack_timeout_requires_finite_nonnegative_value(
    monkeypatch, value
) -> None:
    monkeypatch.setenv("SCRUFFY_ROOT", "/shared/scruffy")
    monkeypatch.setenv("SCRUFFY_JOB_ID", "job-123")
    with pytest.raises(ValueError, match="artifact_ack_timeout_s"):
        make_scruffy_hooks(artifact_ack_timeout_s=value)


def test_scruffy_checkpoint_ack_timeout_rejects_malformed_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SCRUFFY_ROOT", "/shared/scruffy")
    monkeypatch.setenv("SCRUFFY_JOB_ID", "job-123")
    monkeypatch.setenv("KOOCHAK_SCRUFFY_ARTIFACT_ACK_TIMEOUT_SECONDS", "bad")
    with pytest.raises(ValueError, match="artifact_ack_timeout_s"):
        make_scruffy_hooks()


def test_scruffy_checkpoint_ack_rejection_fails_closed(monkeypatch, tmp_path) -> None:
    from koochak.storage import checkpoint as checkpoint_lib

    checkpoint_path = tmp_path / "step000000001.pt"
    checkpoint = {"step": 1, "model": {}}
    checkpoint_lib.save(checkpoint, str(checkpoint_path))
    monkeypatch.setenv("SCRUFFY_ROOT", "/shared/scruffy")
    monkeypatch.setenv("SCRUFFY_JOB_ID", "job-123")

    def reject(_root, **_values):
        return {"state": "rejected", "acknowledged": False}

    module = types.ModuleType("scruffy")
    module.publish_event = reject
    monkeypatch.setitem(sys.modules, "scruffy", module)
    hooks = make_scruffy_hooks(artifact_ack_timeout_s=1)

    with pytest.raises(RuntimeError, match="did not acknowledge"):
        _call(hooks, "on_checkpoint", str(checkpoint_path), checkpoint, {"step": 1})
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        hooks_lib.emit(
            hooks,
            "on_checkpoint",
            str(checkpoint_path),
            checkpoint,
            {"step": 1},
        )


def test_setup_failure_publishes_failed_phase(monkeypatch, tmp_path) -> None:
    from koochak import loop as loop_mod
    from koochak.loop import training_loop

    published = []

    def fail_compile(_loop):
        raise RuntimeError("compile setup failed")

    monkeypatch.setattr(loop_mod._TrainLoop, "_apply_compile", fail_compile)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="compile setup failed"):
        training_loop(
            model=model,
            dataset=[],
            step_fn=lambda *_args: {"loss": torch.tensor(0.0, requires_grad=True)},
            optimizer=optimizer,
            train_cfg={
                "device": "cpu",
                "max_steps": 0,
                "out_dir": str(tmp_path),
            },
            hooks=make_event_hooks(
                lambda kind, data: published.append((kind, data))
            ),
        )

    assert [data["status"] for kind, data in published if kind == "workload.phase"] == [
        "started",
        "failed",
    ]
    assert published[-1][1]["error_type"] == "RuntimeError"
