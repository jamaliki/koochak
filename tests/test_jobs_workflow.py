from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from koochak.jobs import (
    EnvironmentProfile,
    PreparedTask,
    PreparedWorkflow,
    prepare_run,
    submit_scruffy_workflow,
)
from koochak.jobs import backends


def _run(tmp_path: Path, name: str):
    profile = EnvironmentProfile(
        profile_id="workflow-test",
        python=sys.executable,
        variables={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"},
    )
    return prepare_run(
        name=name,
        profile=profile,
        python_args=["-c", "print('ok')"],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / name),
    )


def _resources() -> dict[str, int]:
    return {
        "nodes": 1,
        "gpus_per_node": 0,
        "cpus_per_node": 1,
        "memory_gb_per_node": 1,
    }


def test_workflow_stages_all_runs_before_one_scruffy_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _run(tmp_path, "first")
    second = _run(tmp_path, "second")
    workflow = PreparedWorkflow(
        request_id="campaign/train/attempt-1",
        workflow_id="campaign-1",
        project_id="project",
        tasks=(
            PreparedTask("train", first, _resources()),
            PreparedTask(
                "consume",
                second,
                _resources(),
                wait_for=({"kind": "artifact", "task_id": "train", "artifact_id": "checkpoint/step1.pt"},),
            ),
        ),
    )
    calls: list[object] = []

    def fake_submit(root, **kwargs):
        calls.append((root, kwargs, Path(first.manifest_path).is_file(), Path(second.manifest_path).is_file()))
        return {"state": "submitted", "deduplicated": False}

    module = types.ModuleType("scruffy")
    module.submit_workflow = fake_submit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scruffy", module)
    result = submit_scruffy_workflow(workflow, root=tmp_path / "queue")

    assert result["state"] == "submitted"
    assert len(calls) == 1
    _, kwargs, first_staged, second_staged = calls[0]
    assert first_staged and second_staged
    assert kwargs["request_id"] == workflow.request_id
    assert kwargs["tasks"][1]["needs"] == []
    assert kwargs["tasks"][1]["wait_for"] == [
        {"kind": "artifact", "task_id": "train", "artifact_id": "checkpoint/step1.pt"}
    ]


def test_workflow_request_id_is_forwarded_for_client_deduplication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = PreparedWorkflow(
        request_id="stable/request-1",
        workflow_id="workflow-1",
        project_id="project",
        tasks=(PreparedTask("task", _run(tmp_path, "task"), _resources()),),
    )
    seen: list[str] = []

    def fake_submit(_root, **kwargs):
        seen.append(kwargs["request_id"])
        return {"state": "submitted", "deduplicated": len(seen) == 2}

    module = types.ModuleType("scruffy")
    module.submit_workflow = fake_submit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scruffy", module)
    first = submit_scruffy_workflow(workflow, root=tmp_path / "queue")
    second = submit_scruffy_workflow(workflow, root=tmp_path / "queue")
    assert first["deduplicated"] is False
    assert second["deduplicated"] is True
    assert seen == ["stable/request-1", "stable/request-1"]


def test_workflow_staging_failure_admits_no_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = PreparedWorkflow(
        request_id="request/1",
        workflow_id="workflow-1",
        project_id="project",
        tasks=(PreparedTask("task", _run(tmp_path, "task"), _resources()),),
    )
    called = False

    def fail_stage(_run):
        raise OSError("stage failed")

    def fake_submit(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(backends, "stage_run", fail_stage)
    module = types.ModuleType("scruffy")
    module.submit_workflow = fake_submit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scruffy", module)
    with pytest.raises(OSError, match="stage failed"):
        submit_scruffy_workflow(workflow, root=tmp_path / "queue")
    assert not called


def test_workflow_is_immutable_and_recovery_is_preserved_for_strict_client() -> None:
    run = _run(Path("/tmp"), "workflow-immutable")
    task = PreparedTask(
        "task",
        run,
        _resources(),
        recovery={
            "max_attempts": 2,
            "retry_on": ["evacuated"],
            "evacuation": {"signal": "SIGUSR1", "grace_seconds": 10},
        },
    )
    workflow = PreparedWorkflow("request/1", "workflow-1", "project", (task,))
    assert workflow.tasks[0].to_scruffy_spec(
        request_id=workflow.request_id,
        workflow_id=workflow.workflow_id,
        project_id=workflow.project_id,
    )["recovery"]["evacuation"]["signal"] == "USR1"
    with pytest.raises(AttributeError):
        workflow.workflow_id = "other"  # type: ignore[misc]
