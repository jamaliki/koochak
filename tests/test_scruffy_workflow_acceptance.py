from __future__ import annotations

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import sysconfig
import time
import traceback
from pathlib import Path
from typing import Any

import pytest

from koochak.jobs import (
    DeclaredOutput,
    EnvironmentProfile,
    PreparedTask,
    PreparedWorkflow,
    prepare_run,
    submit_scruffy_workflow,
)
from koochak.storage import checkpoint

SCRUFFY_COMMIT = "d2b7dc2f98794eaf585077f67b9fd3644bb565ab"
WORKFLOW_ID = "koochak-scruffy-acceptance"
PROJECT_ID = "koochak-acceptance"
TRAIN_ARTIFACT_ID = "checkpoint/step000000003.pt"
MAX_STEPS = 6
TIMEOUT = 30.0

TRAINER = r'''
import os
import sys
import time
from pathlib import Path

import torch

from koochak.logging.events import make_scruffy_hooks
from koochak.loop import training_loop

out_dir = Path(sys.argv[1])
pause_marker = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(1729)
model = torch.nn.Linear(1, 1, bias=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
dataset = [{"x": torch.ones(1, 1), "y": torch.zeros(1, 1)} for _ in range(16)]

def step_fn(module, batch, _ctx):
    return {"loss": torch.nn.functional.mse_loss(module(batch["x"]), batch["y"])}

hooks = make_scruffy_hooks(progress_interval_s=0)
original_checkpoint = hooks["on_checkpoint"][0]

def hold_after_first_published_checkpoint(path, saved, context):
    original_checkpoint(path, saved, context)
    if (
        os.environ.get("SCRUFFY_ATTEMPT") == "1"
        and Path(path).name == "step000000003.pt"
    ):
        pause_marker.write_text("published\n")
        time.sleep(1.5)

hooks["on_checkpoint"][0] = hold_after_first_published_checkpoint
training_loop(
    model=model,
    dataset=dataset,
    step_fn=step_fn,
    optimizer=optimizer,
    train_cfg={
        "ddp": False,
        "device": "cpu",
        "max_steps": 6,
        "ckpt_every": 3,
        "log_every": 1000,
        "out_dir": str(out_dir),
        "evacuation_enabled": True,
    },
    resume="auto",
    hooks=hooks,
)
'''

STAGE = r'''
import sys
from pathlib import Path

from koochak.storage.artifact import DeclaredOutput, publish_artifact

artifact_id, stage, output_path, provenance_json, text = sys.argv[1:]
output = DeclaredOutput(
    artifact_id,
    output_path,
    stage=stage,
    provenance=__import__("json").loads(provenance_json),
)
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
Path(output_path).write_text(text)
publish_artifact(output)
'''


def _controller_worker(root: str, scruffy_source: str) -> None:
    sys.path.insert(0, scruffy_source)
    try:
        from scruffy.controller import run_controller
        from scruffy.models import NodeInventory

        run_controller(
            root=Path(root),
            inventory=(NodeInventory("local-node", (0,), 1, 1),),
            launcher="local",
            allocation_id="acceptance-allocation",
            poll_interval=0.01,
            cancel_grace=0.5,
            gpu_health_mode="off",
        )
    except Exception:
        traceback.print_exc()
        raise


def _same_state(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_state(first, second) for first, second in zip(left, right)
        )
    if hasattr(left, "equal"):
        return bool(left.equal(right))
    return left == right


def _wait_until(
    predicate, *, description: str, controller: multiprocessing.Process
) -> Any:
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        if not controller.is_alive():
            raise AssertionError(f"controller exited while waiting for {description}")
        try:
            result = predicate()
        except (FileNotFoundError, KeyError):
            result = None
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {description}")


def _profile() -> EnvironmentProfile:
    return EnvironmentProfile(
        profile_id="scruffy-acceptance",
        python=sys.executable,
        variables={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"},
    )


def _resources() -> dict[str, int]:
    return {
        "nodes": 1,
        "gpus_per_node": 0,
        "cpus_per_node": 1,
        "memory_gb_per_node": 1,
    }


def _provenance(task_id: str) -> dict[str, str]:
    return {
        "project_id": PROJECT_ID,
        "workflow_id": WORKFLOW_ID,
        "task_id": task_id,
        "code_commit": "koochak-acceptance-fixture",
    }


def _artifact_task(
    workspace: Path, task_id: str, predecessor: tuple[str, str] | None = None
) -> PreparedTask:
    artifact_id = f"{task_id}/result.txt"
    output_path = workspace / "outputs" / f"{task_id}.txt"
    script = workspace / "fixtures" / f"{task_id}.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(STAGE)
    provenance = _provenance(task_id)
    output = DeclaredOutput(
        artifact_id,
        str(output_path),
        stage=task_id,
        provenance=provenance,
    )
    run = prepare_run(
        name=task_id,
        profile=_profile(),
        python_args=[
            str(script),
            artifact_id,
            task_id,
            str(output_path),
            json.dumps(provenance, sort_keys=True),
            f"{task_id}\n",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        run_dir=str(workspace / "runs" / task_id),
        declared_outputs=[output],
    )
    wait_for = ()
    if predecessor is not None:
        producer, producer_artifact = predecessor
        wait_for = ({"kind": "artifact", "task_id": producer, "artifact_id": producer_artifact},)
    return PreparedTask(task_id, run, _resources(), wait_for=wait_for)


@pytest.fixture
def exact_scruffy(monkeypatch: pytest.MonkeyPatch):
    source_value = os.environ.get("KOOCHAK_SCRUFFY_SOURCE")
    if not source_value:
        pytest.skip("set KOOCHAK_SCRUFFY_SOURCE to the exact Scruffy checkout")
    source = Path(source_value).resolve()
    package_root = source / "src"
    if not (package_root / "scruffy" / "controller.py").is_file():
        pytest.fail(f"Scruffy source is not a checkout with src/scruffy: {source}")
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert revision.returncode == 0, revision.stderr
    assert revision.stdout.strip() == SCRUFFY_COMMIT, (
        f"acceptance test requires Scruffy {SCRUFFY_COMMIT}, got {revision.stdout.strip()}"
    )

    sys.path.insert(0, str(package_root))
    monkeypatch.setenv("PYTHONPATH", str(package_root))

    # The managed runner deliberately uses ``python -I``.  Make the exact
    # checkout visible to those fresh interpreters without changing either
    # repository or relying on a globally installed Scruffy package.
    purelib = Path(sysconfig.get_paths()["purelib"])
    bridge = purelib / "koochak_acceptance_scruffy.pth"
    prior = bridge.read_bytes() if bridge.exists() else None
    bridge.write_text(str(package_root) + "\n")
    try:
        yield source, package_root
    finally:
        if prior is None:
            bridge.unlink(missing_ok=True)
        else:
            bridge.write_bytes(prior)
        sys.path.remove(str(package_root))


def test_atomic_workflow_survives_local_evacuation_and_matches_reference(
    tmp_path: Path, exact_scruffy
) -> None:
    _, package_root = exact_scruffy
    from scruffy.client import request_evacuation, status, wait_for_evacuation
    from scruffy.storage import read_events

    root = tmp_path / "queue"
    workspace = tmp_path / "workspace"
    train_dir = workspace / "train"
    pause_marker = workspace / "pause.marker"
    trainer_profile = _profile()
    trainer = prepare_run(
        name="train",
        profile=trainer_profile,
        python_args=[str(workspace / "trainer.py"), str(train_dir), str(pause_marker)],
        cwd=str(Path(__file__).resolve().parents[1]),
        run_dir=str(workspace / "runs" / "train"),
    )
    (workspace / "trainer.py").parent.mkdir(parents=True, exist_ok=True)
    (workspace / "trainer.py").write_text(TRAINER)

    sample = _artifact_task(workspace, "sample", ("train", TRAIN_ARTIFACT_ID))
    fold = _artifact_task(workspace, "fold", ("sample", "sample/result.txt"))
    analysis = _artifact_task(workspace, "analysis", ("fold", "fold/result.txt"))
    train = PreparedTask(
        "train",
        trainer,
        _resources(),
        recovery={
            "max_attempts": 2,
            "retry_on": ["evacuated"],
            "evacuation": {"signal": "USR1", "grace_seconds": 5},
        },
    )
    workflow = PreparedWorkflow(
        request_id="koochak-acceptance-request",
        workflow_id=WORKFLOW_ID,
        project_id=PROJECT_ID,
        tasks=(train, sample, fold, analysis),
    )

    context = multiprocessing.get_context("spawn")
    controller = context.Process(
        target=_controller_worker,
        args=(str(root), str(package_root)),
    )
    controller.start()
    try:
        _wait_until(
            lambda: (
                snapshot
                if (snapshot := status(root)).get("allocation")
                and snapshot["allocation"].get("state") == "running"
                else None
            ),
            description="controller startup",
            controller=controller,
        )
        submitted = submit_scruffy_workflow(workflow, root=root)
        assert submitted["state"] == "submitted"
        task_ids = {item["task_id"]: item["job_id"] for item in submitted["tasks"]}
        train_id = task_ids["train"]

        def checkpoint_seen() -> dict[str, Any] | None:
            job = status(root, train_id)
            if any(
                item.get("publication", {}).get("artifact_id") == TRAIN_ARTIFACT_ID
                for item in job.get("artifact_evidence", [])
                if isinstance(item, dict)
            ):
                return job
            return None

        _wait_until(
            checkpoint_seen,
            description="numbered checkpoint publication",
            controller=controller,
        )
        assert pause_marker.read_text() == "published\n"
        evacuation = request_evacuation(
            root,
            workflow_id=WORKFLOW_ID,
            project_id=PROJECT_ID,
            request_id="acceptance-evacuation",
            resume_after=True,
        )
        result = wait_for_evacuation(root, evacuation["request_id"], timeout=TIMEOUT)
        assert result["state"] == "complete"
        assert [target["outcome"] for target in result["targets"].values()] == ["retry_queued"]

        terminal_jobs = {
            task_id: _wait_until(
                lambda task_id=task_id, job_id=job_id: (
                    job
                    if (job := status(root, job_id))["state"] in {"succeeded", "failed"}
                    else None
                ),
                description=f"{task_id} terminal state",
                controller=controller,
            )
            for task_id, job_id in task_ids.items()
            if task_id != "train"
        }
        trainer_attempt1 = status(root, train_id)
        assert trainer_attempt1["state"] == "failed"
        assert trainer_attempt1["reason"] == "evacuated"
        successor_id = trainer_attempt1["successor_job_id"]
        trainer_attempt2 = _wait_until(
            lambda: (
                job
                if (job := status(root, successor_id))["state"] in {"succeeded", "failed"}
                else None
            ),
            description="trainer retry success",
            controller=controller,
        )
        assert trainer_attempt2["state"] == "succeeded"
        assert trainer_attempt2["attempt"] == 2
        assert trainer_attempt2["predecessor_job_id"] == train_id
        assert trainer_attempt2["retry_reason"] == "evacuated"
        train_jobs = [
            job
            for job in status(root).get("jobs", {}).values()
            if job.get("workflow_id") == WORKFLOW_ID and job.get("task_id") == "train"
        ]
        assert len(train_jobs) == 2
        assert sorted(job["attempt"] for job in train_jobs) == [1, 2]

        resumed = checkpoint.load(str(train_dir / "step000000006.pt"))
        assert resumed["next_step"] == MAX_STEPS
        interruption_cursor = checkpoint.load(str(train_dir / "step000000005.pt"))["next_step"]
        assert interruption_cursor == 5
        assert resumed["next_step"] > interruption_cursor

        for task_id, job in terminal_jobs.items():
            assert job["state"] == "succeeded", (task_id, job)
            assert len(
                [candidate for candidate in status(root).get("jobs", {}).values() if candidate.get("task_id") == task_id]
            ) == 1
            output_path = workspace / "outputs" / f"{task_id}.txt"
            assert output_path.read_text() == f"{task_id}\n"
            assert output_path.with_name(output_path.name + ".ready.json").is_file()

        events = read_events(root)
        assert sum(
            event.get("data", {}).get("publication", {}).get("artifact_id") == TRAIN_ARTIFACT_ID
            for event in events
            if event.get("kind") == "workload.artifact"
        ) == 1
        for artifact_id in ("sample/result.txt", "fold/result.txt", "analysis/result.txt"):
            assert sum(
                event.get("data", {}).get("publication", {}).get("artifact_id") == artifact_id
                for event in events
                if event.get("kind") == "workload.artifact"
            ) == 1
        sample_job = terminal_jobs["sample"]
        assert [
            (item["task_id"], item["artifact_id"])
            for item in sample_job["condition_satisfactions"]
        ] == [("train", TRAIN_ARTIFACT_ID)]

        # Run the same immutable trainer command without Scruffy interruption
        # and compare the complete deterministic model/optimizer state.
        reference_dir = workspace / "reference"
        torch_script = workspace / "reference.py"
        torch_script.write_text(
            TRAINER.replace(
                'hooks = make_scruffy_hooks(progress_interval_s=0)',
                'hooks = {}',
            ).replace(
                'original_checkpoint = hooks["on_checkpoint"][0]',
                'original_checkpoint = lambda *args: None',
            ).replace(
                'hooks["on_checkpoint"][0] = hold_after_first_published_checkpoint',
                '',
            ).replace(
                'resume="auto",',
                'resume=None,',
            )
        )
        reference = subprocess.run(
            [sys.executable, str(torch_script), str(reference_dir), str(workspace / "unused.marker")],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "SCRUFFY_ATTEMPT": "reference"},
        )
        assert reference.returncode == 0, reference.stderr
        expected = checkpoint.load(str(reference_dir / "step000000006.pt"))
        assert _same_state(resumed["model"], expected["model"])
        assert _same_state(resumed["optimizer"], expected["optimizer"])
    finally:
        if controller.is_alive():
            os.kill(controller.pid, signal.SIGTERM)
        controller.join(TIMEOUT)
        if controller.is_alive():
            controller.kill()
            controller.join()
        assert controller.exitcode in (0, -signal.SIGTERM)
