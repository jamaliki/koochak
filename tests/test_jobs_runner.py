from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from koochak.jobs import EnvironmentProfile, prepare_run, runner, stage_run


def _profile(*, secrets: tuple[str, ...] = ()) -> EnvironmentProfile:
    return EnvironmentProfile(
        profile_id="runner-test",
        python=sys.executable,
        variables={
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "EXPLICIT": "profile",
        },
        secrets=secrets,
        create_directories=("{run_dir}/cache",),
    )


def test_clean_environment_discards_ambient_values_and_preserves_runtime_identity() -> None:
    environment = _profile(secrets=("TOKEN",)).resolve(
        cwd="/work", run_dir="/runs/a"
    )
    source = {
        "PATH": "/ambient/bin",
        "PYTHONPATH": "/ambient/python",
        "CC": "/ambient/compiler",
        "NCCL_DEBUG": "ambient",
        "SLURM_JOB_ID": "123",
        "CUDA_VISIBLE_DEVICES": "2",
        "TOKEN": "secret",
    }

    clean = runner.build_environment(environment.to_dict(), source)

    assert clean["PATH"] != source["PATH"]
    assert clean["EXPLICIT"] == "profile"
    assert clean["SLURM_JOB_ID"] == "123"
    assert clean["CUDA_VISIBLE_DEVICES"] == "2"
    assert clean["TOKEN"] == "secret"
    assert "PYTHONPATH" not in clean
    assert "CC" not in clean
    assert "NCCL_DEBUG" not in clean


def test_runner_records_preflight_then_execs_exact_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare_run(
        name="runner",
        profile=_profile(),
        python_args=["-c", "print('ok')"],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
    )
    stage_run(prepared)
    seen: dict[str, object] = {}

    class ExecCalled(Exception):
        pass

    def fake_exec(executable, argv, environment):
        seen.update(executable=executable, argv=argv, environment=environment)
        raise ExecCalled

    monkeypatch.setattr(runner.os, "environ", {"HOME": str(tmp_path), "NOISE": "drop"})
    monkeypatch.setattr(runner.os, "chdir", lambda _: None)
    monkeypatch.setattr(runner.os, "execve", fake_exec)

    with pytest.raises(ExecCalled):
        runner.execute_manifest(prepared.manifest_path, prepared.manifest_sha256)

    result = json.loads((Path(prepared.run_dir) / "preflight.json").read_text())
    assert result["state"] == "passed"
    assert result["profile_sha256"] == _profile().sha256
    assert seen["argv"] == [sys.executable, "-c", "print('ok')"]
    assert seen["environment"]["EXPLICIT"] == "profile"  # type: ignore[index]
    assert "NOISE" not in seen["environment"]  # type: ignore[operator]


def test_runner_rejects_a_tampered_materialized_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("train:\n  out_dir: old\n")
    prepared = prepare_run(
        name="runner",
        profile=_profile(),
        python_args=["train.py", "--config", "{config}"],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
        base_config=base,
    )
    stage_run(prepared)
    config_path = Path(prepared.run_dir) / "config.yaml"
    config_path.chmod(0o644)
    config_path.write_text("tampered: true\n")
    monkeypatch.setattr(runner.os, "environ", {"HOME": str(tmp_path)})

    with pytest.raises(runner.PreflightError, match="config digest differs"):
        runner.execute_manifest(prepared.manifest_path, prepared.manifest_sha256)

    result = json.loads((Path(prepared.run_dir) / "preflight.json").read_text())
    assert result["state"] == "failed"
    assert "config digest differs" in result["error"]


def test_isolated_runner_executes_the_declared_python(tmp_path: Path) -> None:
    output = tmp_path / "run" / "executed.txt"
    prepared = prepare_run(
        name="subprocess-runner",
        profile=_profile(),
        python_args=[
            "-c",
            f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
        ],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
    )
    stage_run(prepared)
    environment = {"HOME": str(tmp_path), "NOISE": "must-not-survive"}

    result = subprocess.run(
        prepared.runner_argv(),
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text() == "ok"
    preflight = json.loads((Path(prepared.run_dir) / "preflight.json").read_text())
    assert preflight["state"] == "passed"
    assert preflight["environment_sha256"]
