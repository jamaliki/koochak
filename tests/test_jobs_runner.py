from __future__ import annotations

import builtins
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
        relay,
        preemptible,
        import_paths,
        runtime_path,
        runtime_sha256,
    ):
        seen.update(executable=argv[0], argv=argv, environment=environment)
        assert isinstance(relay, runner._EvacuationRelay)  # noqa: SLF001
        assert preemptible is False
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
        lambda argv, *, cwd, environment, relay, preemptible, import_paths, runtime_path, runtime_sha256: seen.update(
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


def test_standalone_runner_completes_two_published_outputs_without_scruffy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    marker = tmp_path / "scruffy-imported"
    (source / "scruffy.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).touch(); raise AssertionError('unexpected Scruffy import')"
    )
    profile = EnvironmentProfile(profile_id="standalone-outputs", python=sys.executable,
        variables={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
                   "PYTHONPATH": f"{source}:{Path(__file__).resolve().parents[1]}"})
    folder = tmp_path / "run"
    outputs = [DeclaredOutput(name, str(folder / f"{name}.txt")) for name in ("first", "second")]
    code = "\n".join([
        "from pathlib import Path", "from koochak.storage.artifact import DeclaredOutput, publish_artifact",
        *[f"Path({output.path!r}).write_text({output.artifact_id!r}); "
          f"publish_artifact(DeclaredOutput({output.artifact_id!r}, {output.path!r}))" for output in outputs],
    ])
    prepared = prepare_run(name="standalone-outputs", profile=profile,
        python_args=["-I", "-c", code], cwd=str(tmp_path), run_dir=str(folder), declared_outputs=outputs)
    stage_run(prepared)
    result = subprocess.run(prepared.runner_argv(), capture_output=True, text=True,
        env={"HOME": str(tmp_path), "SLURM_JOB_ID": "standalone-123"}, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    preflight = json.loads((folder / "preflight.json").read_text())
    assert preflight["state"] == "completed" and preflight["outputs"] == ["first", "second"]


def test_standalone_outputs_keep_two_validation_passes_without_importing_scruffy(tmp_path, monkeypatch):
    outputs = []
    for name in ("first", "second"):
        file = tmp_path / (name + ".txt")
        file.write_text(name)
        output = DeclaredOutput(name, str(file))
        publish_artifact(output)
        outputs.append(output)
    imported, validated = builtins.__import__, runner.validate_artifact
    calls = []
    def reject_scruffy(name, *args, **kwargs):
        assert name != "scruffy", "Standalone outputs must not import Scruffy"
        return imported(name, *args, **kwargs)
    def validate(output):
        calls.append(output.artifact_id)
        return validated(output)
    monkeypatch.setattr(builtins, "__import__", reject_scruffy)
    monkeypatch.setattr(runner, "validate_artifact", validate)
    runner._publish_outputs(tuple(outputs), {})
    assert calls == ["first", "second", "first", "second"]


def test_standalone_outputs_reject_changes_between_validation_passes(tmp_path, monkeypatch):
    outputs = []
    for name in ("first", "second"):
        file = tmp_path / (name + ".txt")
        file.write_text(name)
        output = DeclaredOutput(name, str(file))
        publish_artifact(output)
        outputs.append(output)
    validated = runner.validate_artifact
    calls = []
    def mutate(output):
        manifest = validated(output)
        calls.append(output.artifact_id)
        if len(calls) == 2:
            Path(outputs[0].path).write_text("changed")
        return manifest
    monkeypatch.setattr(runner, "validate_artifact", mutate)
    with pytest.raises(ValueError, match="bytes differ"):
        runner._publish_outputs(tuple(outputs), {})
    assert calls == ["first", "second"]


def test_standalone_output_requires_the_workload_ready_manifest(tmp_path):
    file = tmp_path / "unpublished.txt"
    file.write_text("complete payload without a ready manifest")
    with pytest.raises(ValueError, match="ready manifest is unavailable"):
        runner._publish_outputs((DeclaredOutput("unpublished", str(file)),), {})


@pytest.mark.parametrize("environment", [
    {"SCRUFFY_ROOT": "/queue"}, {"SCRUFFY_JOB_ID": "job"},
    {"SCRUFFY_ROOT": "", "SCRUFFY_JOB_ID": ""}, {"SCRUFFY_ROOT": "/queue", "SCRUFFY_JOB_ID": ""},
])
def test_declared_outputs_reject_partial_or_empty_scruffy_identity(tmp_path, environment):
    file = tmp_path / "result.txt"
    file.write_text("result")
    output = DeclaredOutput("result", str(file))
    publish_artifact(output)
    with pytest.raises(runner.PreflightError, match="require both SCRUFFY_ROOT and SCRUFFY_JOB_ID"):
        runner._publish_outputs((output,), environment)


def test_complete_scruffy_identity_still_requires_the_client(tmp_path, monkeypatch):
    file = tmp_path / "result.txt"
    file.write_text("result")
    output = DeclaredOutput("result", str(file))
    publish_artifact(output)
    imported = builtins.__import__
    def reject_scruffy(name, *args, **kwargs):
        if name == "scruffy":
            raise ModuleNotFoundError("No module named 'scruffy'", name="scruffy")
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_scruffy)
    with pytest.raises(runner.PreflightError, match="require the Scruffy client"):
        runner._publish_outputs((output,), {"SCRUFFY_ROOT": "/queue", "SCRUFFY_JOB_ID": "job"})


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


_HANDLER_CHILD = """
import os, signal, sys, time
from pathlib import Path
requested = False
def handle(_signum, _frame):
    global requested
    requested = True
time.sleep(0.3)
signal.signal(signal.SIGUSR1, handle)
(Path(os.environ["KOOCHAK_EVACUATION_READY_DIR"]) / str(os.getpid())).touch()
deadline = time.monotonic() + 10
while not requested and time.monotonic() < deadline:
    time.sleep(0.01)
sys.exit(75 if requested else 3)
"""


def test_preemptible_runner_waits_for_the_workload_handler_before_forwarding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[int, int]] = []
    real_kill = os.kill

    def record_kill(pid: int, requested_signal: int) -> None:
        sent.append((pid, requested_signal))
        real_kill(pid, requested_signal)

    with runner._EvacuationRelay() as relay:  # noqa: SLF001 - contract-level test
        monkeypatch.setattr(runner.os, "kill", record_kill)
        # A real SIGUSR1 arrives after launch but before the child's handler.
        timer = threading.Timer(0.05, real_kill, (os.getpid(), signal.SIGUSR1))
        timer.start()
        try:
            result = runner._run_managed_child(  # noqa: SLF001
                [sys.executable, "-c", _HANDLER_CHILD],
                cwd="/tmp",
                environment={"PATH": str(Path(sys.executable).parent)},
                relay=relay,
                preemptible=True,
            )
        finally:
            timer.join()
    # The request predates the child's handler; it stayed pending instead of
    # killing the child with the default action, then reached it exactly once.
    assert result == 75
    assert len(sent) == 1 and sent[0][1] == signal.SIGUSR1


def test_runner_ignores_evacuation_for_a_run_that_is_not_preemptible(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        runner,
        "_forward_to_leader",
        lambda *_args: pytest.fail("non-preemptible runs must never be signalled"),
    )
    with runner._EvacuationRelay() as relay:  # noqa: SLF001
        relay.requested = True
        result = runner._run_managed_child(  # noqa: SLF001
            [sys.executable, "-c", "import time; time.sleep(.3)"],
            cwd="/tmp",
            environment={"PATH": str(Path(sys.executable).parent)},
            relay=relay,
            preemptible=False,
        )
    assert result == 0
    assert '"reason": "run_not_preemptible"' in capsys.readouterr().err


@pytest.mark.parametrize("preemptible", [True, False])
def test_only_preemptible_children_receive_the_ready_directory(
    tmp_path: Path, preemptible: bool
) -> None:
    seen = tmp_path / "seen"
    code = (
        "import os, pathlib, sys; "
        "pathlib.Path(sys.argv[1]).write_text(os.environ.get('KOOCHAK_EVACUATION_READY_DIR', ''))"
    )
    with runner._EvacuationRelay() as relay:  # noqa: SLF001
        assert (
            runner._run_managed_child(  # noqa: SLF001
                [sys.executable, "-c", code, str(seen)],
                cwd="/tmp",
                environment={"PATH": str(Path(sys.executable).parent)},
                relay=relay,
                preemptible=preemptible,
            )
            == 0
        )
    ready_dir = seen.read_text()
    assert bool(ready_dir) is preemptible
    if preemptible:
        assert not Path(ready_dir).exists(), "the ready directory is private to one launch"


def test_preemptible_run_records_evacuated_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("train:\n  out_dir: ./runs/x\n")
    prepared = prepare_run(
        name="preemptible-exit",
        profile=_profile(),
        python_args=["-c", "raise SystemExit(75)"],
        cwd=str(tmp_path),
        run_dir=str(tmp_path / "run"),
        base_config=base,
        preemptible=True,
    )
    stage_run(prepared)
    monkeypatch.setattr(runner.os, "environ", {"HOME": str(tmp_path)})
    assert runner.execute_manifest(prepared.manifest_path, prepared.manifest_sha256) == 75
    result = json.loads((tmp_path / "run" / "preflight.json").read_text())
    assert result["state"] == "evacuated"
    assert result["child_returncode"] == 75


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
    with runner._EvacuationRelay() as relay:  # noqa: SLF001
        assert (
            runner._run_managed_child(  # noqa: SLF001
                [sys.executable, "-c", "pass"],
                cwd="/tmp",
                environment={"PATH": str(Path(sys.executable).parent)},
                relay=relay,
                preemptible=True,
            )
            == 75
        )


def test_pending_signal_before_launch_is_ignored_for_a_run_that_is_not_preemptible() -> None:
    with runner._EvacuationRelay() as relay:  # noqa: SLF001
        relay.requested = True
        assert (
            runner._run_managed_child(  # noqa: SLF001
                [sys.executable, "-c", "pass"],
                cwd="/tmp",
                environment={"PATH": str(Path(sys.executable).parent)},
                relay=relay,
                preemptible=False,
            )
            == 0
        )


def test_signal_just_after_spawn_is_forwarded_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Child:
        pid = 12345
        waits = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits < 4:
                raise subprocess.TimeoutExpired("child", timeout)
            return 0

    def spawn(*_args, **_kwargs):
        os.kill(os.getpid(), signal.SIGUSR1)
        return Child()

    forwarded: list[int] = []
    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        runner,
        "_forward_to_leader",
        lambda _child, requested_signal, _ready: forwarded.append(requested_signal) or True,
    )
    monkeypatch.setattr(runner, "_require_quiescent_group", lambda _group: None)
    with runner._EvacuationRelay() as relay:  # noqa: SLF001
        assert (
            runner._run_managed_child(  # noqa: SLF001
                [sys.executable, "-c", "pass"],
                cwd="/tmp",
                environment={},
                relay=relay,
                preemptible=True,
            )
            == 0
        )
    assert forwarded == [signal.SIGUSR1]


def test_stale_or_completed_process_group_is_never_signaled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Child:
        pid = 12345

        def __init__(self, returncode):
            self.returncode = returncode

        def poll(self):
            return self.returncode

    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    (ready_dir / "12345").touch()
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(runner.os, "kill", lambda *values: sent.append(values))
    assert runner._forward_to_leader(Child(0), signal.SIGUSR1, ready_dir) is False  # noqa: SLF001
    monkeypatch.setattr(runner.os, "getpgid", lambda _pid: 999)
    with pytest.raises(runner.PreflightError, match="no longer owns"):
        runner._forward_to_leader(Child(None), signal.SIGUSR1, ready_dir)  # noqa: SLF001
    assert sent == []


def test_leader_without_a_ready_handler_is_never_signaled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Child:
        pid = 12345

        def poll(self):
            return None

    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(runner.os, "kill", lambda *values: sent.append(values))
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: pid)
    (tmp_path / "999").touch()  # another process in the group is not the leader
    assert runner._forward_to_leader(Child(), signal.SIGUSR1, tmp_path) is False  # noqa: SLF001
    (tmp_path / "12345").touch()
    assert runner._forward_to_leader(Child(), signal.SIGUSR1, tmp_path) is True  # noqa: SLF001
    assert sent == [(12345, signal.SIGUSR1)]


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
    with runner._EvacuationRelay() as relay, pytest.raises(  # noqa: SLF001
        runner.PreflightError, match="background descendants"
    ):
        runner._run_managed_child(  # noqa: SLF001
            [sys.executable, "-c", code],
            cwd=str(tmp_path),
            environment={"PATH": str(Path(sys.executable).parent)},
            relay=relay,
            preemptible=False,
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


def test_required_files_wait_out_a_dropped_mount_but_not_forever(tmp_path, monkeypatch):
    import errno
    from pathlib import Path

    target = tmp_path / "config.yaml"
    target.write_text("x")
    real_is_file = Path.is_file
    failures = {"left": 2}

    def flaky(self):
        if self == target and failures["left"]:
            failures["left"] -= 1
            raise ConnectionAbortedError(errno.ECONNABORTED, "Software caused connection abort")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", flaky)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    assert runner._is_file(str(target)) is True
    assert runner._is_file(str(tmp_path / "missing")) is False

    failures["left"] = 10**6
    clock = iter(range(0, 10**6, 100))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    with pytest.raises(ConnectionAbortedError):
        runner._is_file(str(target), wait_seconds=500)
    monkeypatch.setattr(Path, "is_file", lambda self: (_ for _ in ()).throw(PermissionError(str(self))))
    with pytest.raises(PermissionError):
        runner._is_file(str(target))
