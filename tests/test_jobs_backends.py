from __future__ import annotations

import asyncio
import builtins
import hashlib
import io
import json
import posixpath
import shlex
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from koochak.jobs import (
    EnvironmentProfile,
    prepare_run,
    runner,
    submit_pazuzu,
    submit_scruffy,
)


def _prepared(tmp_path: Path):
    profile = EnvironmentProfile(
        profile_id="backend-test",
        python=sys.executable,
        variables={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"},
    )
    return prepare_run(
        name="backend-test",
        profile=profile,
        python_args=["-m", "project.train"],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
    )


def test_pazuzu_adapter_stages_over_stdin_and_submits_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)

    class FakeSlurmJob:
        def __init__(self, **values):
            self.__dict__.update(values)

    module = types.ModuleType("pazuzu")
    module.SlurmJob = FakeSlurmJob  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pazuzu", module)

    class Client:
        def __init__(self) -> None:
            self.staged = []
            self.job = None

        async def run(self, command, *, stdin, timeout):
            self.staged.append((command, stdin, timeout))
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        async def submit_slurm(self, job):
            self.job = job
            return "handle"

    client = Client()
    resources = object()
    result = asyncio.run(
        submit_pazuzu(
            client,
            prepared,
            resources=resources,
            log_dir=str(tmp_path / "logs"),
        )
    )

    runtime_path = posixpath.join(prepared.run_dir, "koochak-runtime.zip")
    runtime, manifest = prepared.artifacts
    assert (runtime.path, manifest.path) == (runtime_path, prepared.manifest_path)
    assert manifest.sha256 == prepared.manifest_sha256

    # The runtime archive must land before the manifest that pins it.
    assert result == "handle"
    assert len(client.staged) == 2
    (runtime_command, runtime_stdin, runtime_timeout), (
        manifest_command,
        manifest_stdin,
        manifest_timeout,
    ) = client.staged
    runtime_argv = shlex.split(runtime_command)
    manifest_argv = shlex.split(manifest_command)
    assert runtime_argv[:3] == [prepared.python, "-I", "-c"]
    assert runtime_argv[4:] == [runtime.path, runtime.sha256]
    assert manifest_argv[:3] == [prepared.python, "-I", "-c"]
    assert manifest_argv[4:] == [manifest.path, manifest.sha256]
    assert runtime_argv[3] == manifest_argv[3]
    assert manifest.content.decode() not in manifest_command
    assert (runtime_timeout, manifest_timeout) == (60, 60)

    assert runtime_stdin == runtime.content
    assert hashlib.sha256(runtime_stdin).hexdigest() == runtime.sha256
    with zipfile.ZipFile(io.BytesIO(runtime_stdin)) as archive:
        assert archive.read("koochak/jobs/runner.py") == Path(
            runner.__file__
        ).read_bytes()

    assert manifest_stdin == manifest.content
    assert hashlib.sha256(manifest_stdin).hexdigest() == prepared.manifest_sha256
    assert json.loads(manifest_stdin)["runner_runtime"] == {
        "path": runtime.path,
        "sha256": runtime.sha256,
    }

    assert client.job.argv == prepared.runner_argv()
    assert client.job.argv[4:] == [
        runtime.path,
        runtime.sha256,
        manifest.path,
        manifest.sha256,
    ]
    assert client.job.environment == {}
    assert client.job.resources is resources


def test_pazuzu_staging_failure_does_not_submit(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    module = types.ModuleType("pazuzu")
    module.SlurmJob = lambda **values: SimpleNamespace(**values)
    monkeypatch.setitem(sys.modules, "pazuzu", module)

    class Client:
        async def run(self, command, *, stdin, timeout):
            return SimpleNamespace(exit_code=1, stdout="", stderr="staging failed")

        async def submit_slurm(self, job):
            pytest.fail("A failed staging command must prevent submission")

    with pytest.raises(RuntimeError, match="staging failed"):
        asyncio.run(submit_pazuzu(Client(), prepared, resources=object(), log_dir="/logs"))


def test_scruffy_adapter_stages_locally_and_uses_only_the_python_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    seen = {}

    def fake_submit(root, **values):
        seen.update(root=root, **values)
        return {"job_id": "job-1"}

    module = types.ModuleType("scruffy")
    module.submit_job = fake_submit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scruffy", module)
    resources = object()

    result = submit_scruffy(
        prepared,
        root=tmp_path / "queue",
        resources=resources,
        request_id="campaign/train/attempt-1",
        project_id="project",
        workflow_id="campaign",
        task_id="infer",
        wait_for=[
            {
                "kind": "artifact",
                "task_id": "train",
                "artifact_id": "checkpoint/step000000007.pt",
            }
        ],
    )

    assert result == {"job_id": "job-1"}
    assert Path(prepared.manifest_path).is_file()
    assert seen["argv"] == prepared.runner_argv()
    assert seen["environment"] == {}
    assert seen["request"] is resources
    assert seen["wait_for"] == [
        {
            "kind": "artifact",
            "task_id": "train",
            "artifact_id": "checkpoint/step000000007.pt",
        }
    ]


def test_scruffy_adapter_explains_an_incompatible_installed_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    original_import = builtins.__import__

    def incompatible_import(name, *args, **kwargs):
        if name == "scruffy":
            raise ImportError("cannot import name 'UTC' from 'datetime'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", incompatible_import)

    with pytest.raises(RuntimeError, match=r"upgrade.*koochak\[scruffy\]"):
        submit_scruffy(
            prepared,
            root=tmp_path / "queue",
            resources=object(),
            request_id="campaign/train/attempt-1",
            project_id="project",
        )


def test_scruffy_adapter_never_drops_unsupported_artifact_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    called = False

    def old_submit(
        root,
        *,
        argv,
        name,
        cwd,
        environment,
        request,
        request_id,
        project_id,
        workflow_id=None,
        task_id=None,
        needs=None,
    ):
        nonlocal called
        called = True

    module = types.ModuleType("scruffy")
    module.submit_job = old_submit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scruffy", module)

    with pytest.raises(RuntimeError, match=r"artifact conditions.*upgrade"):
        submit_scruffy(
            prepared,
            root=tmp_path / "queue",
            resources=object(),
            request_id="campaign/infer/attempt-1",
            project_id="project",
            workflow_id="campaign",
            task_id="infer",
            needs=[{"task_id": "train", "condition": "succeeded"}],
            wait_for=[
                {
                    "kind": "artifact",
                    "task_id": "train",
                    "artifact_id": "checkpoint/step000000007.pt",
                }
            ],
        )

    assert not called
    assert not Path(prepared.manifest_path).exists()
