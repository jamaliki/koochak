from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from koochak.jobs import (
    DeclaredOutput,
    EnvironmentProfile,
    prepare_run,
    runner,
    stage_run,
)
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

    def fake_child(
        argv,
        *,
        cwd,
        environment,
        import_paths,
        runtime_path,
        runtime_sha256,
    ):
        seen.update(executable=argv[0], argv=argv, environment=environment)
        assert cwd == prepared.cwd
        assert import_paths == ()
        assert runtime_path == str(Path(prepared.run_dir) / "koochak-runtime.zip")
        assert runtime_sha256
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


def test_runner_preflight_honors_profile_pythonpath_for_scruffy_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "client"
    source.mkdir()
    (source / "scruffy.py").write_text(
        "__version__ = 'test'\n"
        "def publish_event(*_args, **_kwargs): return {'state': 'spooled'}\n"
    )
    profile = EnvironmentProfile(
        profile_id="scruffy-runner",
        python=sys.executable,
        variables={
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "PYTHONPATH": str(source),
        },
        packages={"scruffy": "*"},
    )
    prepared = prepare_run(
        name="scruffy-preflight",
        profile=profile,
        python_args=["-c", "print('ok')"],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
    )
    stage_run(prepared)
    monkeypatch.setattr(runner.os, "environ", {"HOME": str(tmp_path)})
    monkeypatch.delitem(sys.modules, "scruffy", raising=False)
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        runner,
        "_run_managed_child",
        lambda argv, *, cwd, environment, import_paths, runtime_path, runtime_sha256: seen.update(
            argv=argv, cwd=cwd, environment=environment
        ) or 0,
    )

    assert runner.execute_manifest(prepared.manifest_path, prepared.manifest_sha256) == 0
    assert seen["environment"]["PYTHONPATH"] == str(source)  # type: ignore[index]
    result = json.loads((Path(prepared.run_dir) / "preflight.json").read_text())
    assert result["state"] == "passed"
    assert result["observed"]["package:scruffy"] == "test"


def test_runner_rejects_missing_scruffy_before_managed_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = EnvironmentProfile(
        profile_id="scruffy-required",
        python=sys.executable,
        variables={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"},
        packages={"scruffy": "*"},
    )
    prepared = prepare_run(
        name="missing-scruffy",
        profile=profile,
        python_args=["-c", "raise SystemExit('child must not run')"],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
    )
    stage_run(prepared)
    monkeypatch.setattr(runner.os, "environ", {"HOME": str(tmp_path)})
    monkeypatch.delitem(sys.modules, "scruffy", raising=False)
    real_import_module = runner.importlib.import_module

    def reject_scruffy(name: str):
        if name == "scruffy":
            raise ModuleNotFoundError("No module named 'scruffy'", name="scruffy")
        return real_import_module(name)

    monkeypatch.setattr(runner.importlib, "import_module", reject_scruffy)
    monkeypatch.setattr(
        runner,
        "_run_managed_child",
        lambda *_args, **_kwargs: pytest.fail("preflight must reject before child launch"),
    )

    with pytest.raises(runner.PreflightError, match="required package is not importable"):
        runner.execute_manifest(prepared.manifest_path, prepared.manifest_sha256)


def test_isolated_runner_and_isolated_child_use_declared_import_root(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "stale-installed-koochak"
    stale_jobs = stale / "koochak" / "jobs"
    stale_jobs.mkdir(parents=True)
    (stale / "koochak" / "__init__.py").write_text("")
    (stale_jobs / "__init__.py").write_text("")
    (stale_jobs / "runner.py").write_text(
        "raise SystemExit('stale installed Koochak runner was used')\n"
    )
    source = tmp_path / "isolated-client"
    package = source / "scruffy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "__version__ = 'synthetic'\n"
        "def publish_event(*_args, **_kwargs): return {'state': 'spooled'}\n"
    )
    marker = tmp_path / "child-import.txt"
    profile = EnvironmentProfile(
        profile_id="isolated-scruffy",
        python=sys.executable,
        variables={
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "PYTHONPATH": str(source),
        },
        packages={"scruffy": "synthetic"},
    )
    prepared = prepare_run(
        name="isolated-scruffy",
        profile=profile,
        python_args=[
            "-I",
            "-c",
            (
                "from pathlib import Path; import scruffy; "
                f"Path({str(marker)!r}).write_text(scruffy.__version__)"
            )
        ],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
    )
    stage_run(prepared)
    result = subprocess.run(
        prepared.runner_argv(),
        capture_output=True,
        text=True,
        env={
            "HOME": str(tmp_path),
            "PATH": str(Path(sys.executable).parent),
            "PYTHONPATH": str(stale),
        },
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "synthetic"
    preflight = json.loads((Path(prepared.run_dir) / "preflight.json").read_text())
    assert preflight["state"] == "passed"
    assert preflight["observed"]["package:scruffy"] == "synthetic"


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
    real_killpg = runner.os.killpg

    def record_signal(process_group: int, requested_signal: int) -> None:
        if requested_signal == 0:
            real_killpg(process_group, requested_signal)
        else:
            sent.append((process_group, requested_signal))

    monkeypatch.setattr(runner.os, "killpg", record_signal)
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


def test_matching_staging_is_hardened_and_symlink_target_is_rejected(
    tmp_path: Path,
) -> None:
    prepared = prepare_run(
        name="staging",
        profile=_profile(),
        python_args=["-c", "print('ok')"],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
    )
    stage_run(prepared)
    manifest = Path(prepared.manifest_path)
    manifest.chmod(0o666)
    stage_run(prepared)
    assert manifest.stat().st_mode & 0o222 == 0

    target = tmp_path / "symlink-run" / "launch.json"
    target.parent.mkdir()
    backing = tmp_path / "backing.json"
    backing.write_bytes(prepared.artifacts[-1].content)
    target.symlink_to(backing)
    symlinked = prepare_run(
        name="symlink-staging",
        profile=_profile(),
        python_args=["-c", "print('ok')"],
        cwd=str(tmp_path),
        run_dir=str(target.parent),
    )
    with pytest.raises(FileExistsError, match="non-regular"):
        stage_run(symlinked)
    assert target.is_symlink()


def test_concurrent_staging_publishes_one_identical_read_only_run(tmp_path: Path) -> None:
    prepared = prepare_run(
        name="concurrent-staging",
        profile=_profile(),
        python_args=["-c", "print('ok')"],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
    )
    barrier = threading.Barrier(8)

    def stage() -> None:
        barrier.wait()
        stage_run(prepared)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _index: stage(), range(8)))
    for artifact in prepared.artifacts:
        target = Path(artifact.path)
        assert target.read_bytes() == artifact.content
        assert target.stat().st_mode & 0o222 == 0


def test_pending_signal_before_launch_exits_without_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_signal = runner.signal.signal
    installed = False

    def deliver_on_install(signum, handler):
        nonlocal installed
        previous = real_signal(signum, handler)
        if callable(handler) and not installed:
            installed = True
            handler(signum, None)
        return previous

    monkeypatch.setattr(runner.signal, "signal", deliver_on_install)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("pending evacuation must prevent launch"),
    )
    assert (
        runner._run_managed_child(
            [sys.executable, "-c", "pass"],
            cwd="/tmp",
            environment={"PATH": str(Path(sys.executable).parent)},
        )
        == 75
    )


def test_signal_just_after_spawn_is_forwarded_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Child:
        pid = 12345

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def spawn(*_args, **_kwargs):
        os.kill(os.getpid(), signal.SIGUSR1)
        return Child()

    forwarded: list[int] = []
    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        runner,
        "_forward_to_owned_group",
        lambda _child, requested_signal: forwarded.append(requested_signal) or True,
    )
    monkeypatch.setattr(runner, "_require_quiescent_group", lambda _group: None)
    assert (
        runner._run_managed_child(
            [sys.executable, "-c", "pass"], cwd="/tmp", environment={}
        )
        == 0
    )
    assert forwarded == [signal.SIGUSR1]


def test_stale_or_completed_process_group_is_never_signaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Child:
        pid = 12345

        def __init__(self, returncode):
            self.returncode = returncode

        def poll(self):
            return self.returncode

    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(runner.os, "killpg", lambda *values: sent.append(values))
    assert runner._forward_to_owned_group(Child(0), signal.SIGUSR1) is False
    monkeypatch.setattr(runner.os, "getpgid", lambda _pid: 999)
    with pytest.raises(runner.PreflightError, match="no longer owns"):
        runner._forward_to_owned_group(Child(None), signal.SIGUSR1)
    assert sent == []


def test_background_descendant_is_killed_and_run_fails_closed(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-started"
    code = f"""
import subprocess, sys, time
from pathlib import Path
subprocess.Popen([
    sys.executable,
    "-c",
    "from pathlib import Path; Path({str(marker)!r}).write_text('yes'); import time; time.sleep(30)",
])
deadline = time.monotonic() + 2
while not Path({str(marker)!r}).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
raise SystemExit(0)
"""
    with pytest.raises(runner.PreflightError, match="background descendants"):
        runner._run_managed_child(
            [sys.executable, "-c", code],
            cwd=str(tmp_path),
            environment={"PATH": str(Path(sys.executable).parent)},
        )
    deadline = time.monotonic() + 1
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.exists()


def test_output_is_revalidated_immediately_before_each_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = []
    for name in ("first", "second"):
        filename = tmp_path / f"{name}.txt"
        filename.write_text(name)
        output = DeclaredOutput(name, str(filename))
        publish_artifact(output)
        outputs.append(output)
    calls: list[str] = []

    def mutate_on_first(_root, **values):
        calls.append(values["data"]["publication"]["artifact_id"])
        if len(calls) == 1:
            Path(outputs[1].path).write_text("changed")

    import types

    module = types.ModuleType("scruffy")
    module.publish_event = mutate_on_first  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scruffy", module)
    with pytest.raises(ValueError, match="bytes differ"):
        runner._publish_outputs(
            tuple(outputs),
            {"SCRUFFY_ROOT": str(tmp_path / "queue"), "SCRUFFY_JOB_ID": "job-1"},
        )
    assert calls == ["first"]


def test_partial_output_event_publication_replays_with_stable_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = []
    for name in ("first", "second"):
        filename = tmp_path / f"{name}.txt"
        filename.write_text(name)
        output = DeclaredOutput(name, str(filename))
        publish_artifact(output)
        outputs.append(output)
    attempts: list[list[str]] = [[], []]
    current = 0

    def flaky(_root, **values):
        event_id = values["event_id"]
        attempts[current].append(event_id)
        if current == 0 and len(attempts[current]) == 2:
            raise OSError("second spool failed")

    import types

    module = types.ModuleType("scruffy")
    module.publish_event = flaky  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scruffy", module)
    environment = {
        "SCRUFFY_ROOT": str(tmp_path / "queue"),
        "SCRUFFY_JOB_ID": "job-1",
    }
    with pytest.raises(OSError, match="second spool"):
        runner._publish_outputs(tuple(outputs), environment)
    current = 1
    runner._publish_outputs(tuple(outputs), environment)
    assert attempts[0] == attempts[1]
