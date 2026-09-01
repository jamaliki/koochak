from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from koochak.jobs import DeclaredOutput, EnvironmentProfile, prepare_run, runner, stage_run
from koochak.storage.artifact import publish_artifact


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


def test_runner_records_preflight_then_runs_exact_workload(
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

    def fake_child(argv, *, cwd, environment):
        seen.update(executable=argv[0], argv=argv, environment=environment)
        assert cwd == prepared.cwd
        return 0

    monkeypatch.setattr(runner.os, "environ", {"HOME": str(tmp_path), "NOISE": "drop"})
    monkeypatch.setattr(runner, "_run_managed_child", fake_child)

    assert runner.execute_manifest(prepared.manifest_path, prepared.manifest_sha256) == 0

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


def test_managed_runner_publishes_declared_output_after_child_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "run" / "result.txt"
    output = DeclaredOutput("result", str(output_path), stage="metrics")
    prepared = prepare_run(
        name="managed-runner",
        profile=_profile(),
        python_args=["-c", f"from pathlib import Path; Path({str(output_path)!r}).write_text('ok')"],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
        declared_outputs=[output],
    )
    stage_run(prepared)
    output_path.write_text("ok")
    publish_artifact(output)
    published: list[dict[str, object]] = []

    def fake_publish(root, **values):
        published.append({"root": root, **values})
        return {"state": "spooled"}

    import types

    module = types.ModuleType("scruffy")
    module.publish_event = fake_publish  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scruffy", module)
    monkeypatch.setattr(runner.os, "environ", {
        "HOME": str(tmp_path),
        "SCRUFFY_ROOT": str(tmp_path / "queue"),
        "SCRUFFY_JOB_ID": "job-1",
    })

    assert runner.execute_manifest(prepared.manifest_path, prepared.manifest_sha256) == 0
    assert len(published) == 1
    assert published[0]["kind"] == "workload.artifact"
    assert published[0]["data"]["publication"]["artifact_id"] == "result"  # type: ignore[index]


def test_managed_runner_preserves_nonzero_child_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = DeclaredOutput("result", str(tmp_path / "run" / "result.txt"))
    prepared = prepare_run(
        name="managed-failure",
        profile=_profile(),
        python_args=["-c", "raise SystemExit(7)"],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
        declared_outputs=[output],
    )
    stage_run(prepared)
    monkeypatch.setattr(runner.os, "environ", {"HOME": str(tmp_path)})
    assert runner.execute_manifest(prepared.manifest_path, prepared.manifest_sha256) == 7


def test_managed_runner_forwards_usr1_to_child_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(runner.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    timer = threading.Timer(
        0.05, lambda: runner.os.kill(runner.os.getpid(), signal.SIGUSR1)
    )
    timer.start()
    try:
        result = runner._run_managed_child(  # noqa: SLF001 - contract-level test
            [sys.executable, "-c", "import time; time.sleep(.2)"],
            cwd="/tmp",
            environment={"PATH": str(Path(sys.executable).parent)},
        )
    finally:
        timer.join()
    assert result == 0
    assert sent and sent[0][1] == signal.SIGUSR1
    assert sent[0][0] > 0
