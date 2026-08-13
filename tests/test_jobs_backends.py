from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from koochak.jobs import (
    EnvironmentProfile,
    prepare_run,
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
            return SimpleNamespace(returncode=0, stdout="", stderr="")

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

    assert result == "handle"
    assert len(client.staged) == 1
    assert client.staged[0][1] == prepared.artifacts[0].content
    assert prepared.artifacts[0].content.decode() not in client.staged[0][0]
    assert client.job.argv == prepared.runner_argv()
    assert client.job.environment == {}
    assert client.job.resources is resources


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
    )

    assert result == {"job_id": "job-1"}
    assert Path(prepared.manifest_path).is_file()
    assert seen["argv"] == prepared.runner_argv()
    assert seen["environment"] == {}
    assert seen["request"] is resources
