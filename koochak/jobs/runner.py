"""Verify a prepared launch, preflight its environment, and exec the workload."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..interruption import (
    EVACUATED_FILE,
    EVACUATION_DIR_ENV,
    EVACUATION_EXIT_CODE,
    EVACUATION_REQUEST_FILE,
)
from ..storage.artifact import (
    DeclaredOutput,
    artifact_publication,
    validate_artifact,
)


class PreflightError(RuntimeError):
    """The declared execution environment did not satisfy its contract."""


_ISOLATED_CHILD_BOOTSTRAP = """\
import hashlib
import json
import runpy
import sys
from pathlib import Path

runtime_path, runtime_sha256, import_paths = sys.argv[1:4]
runtime_content = Path(runtime_path).read_bytes()
if hashlib.sha256(runtime_content).hexdigest() != runtime_sha256:
    raise SystemExit("Koochak runner runtime digest differs")
sys.path.insert(0, runtime_path)
sys.argv = ["koochak.jobs.isolated_exec", import_paths, "--", *sys.argv[5:]]
runpy.run_module("koochak.jobs.isolated_exec", run_name="__main__")
"""


def _sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _matches(name: str, pattern: str) -> bool:
    return name.startswith(pattern[:-1]) if pattern.endswith("*") else name == pattern


def build_environment(
    environment: Mapping[str, Any], source: Mapping[str, str]
) -> dict[str, str]:
    """Build the workload environment from declared and scheduler-owned values."""

    settings = environment["environment"]
    preserve = tuple(settings["preserve"])
    secrets = tuple(settings["secrets"])
    inherited = {
        name: value
        for name, value in source.items()
        if any(_matches(name, pattern) for pattern in preserve)
    }
    result = {**settings["set"], **inherited}
    missing_secrets = [name for name in secrets if name not in source]
    if missing_secrets:
        raise PreflightError(f"missing secret environment names: {missing_secrets!r}")
    result.update({name: source[name] for name in secrets})
    return result


def _load_manifest(manifest_path: str, expected_sha256: str) -> dict[str, Any]:
    content = Path(manifest_path).read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise PreflightError(
            f"launch manifest digest differs: expected {expected_sha256}, got {actual_sha256}"
        )
    document = json.loads(content)
    if not isinstance(document, dict) or document.get("v") != 1:
        raise PreflightError("launch manifest must be a version-1 object")
    return document


# How long a required file on an object-storage mount may keep failing with a
# transient error (see ``storage.store.transient_read_error``) before preflight fails.
TRANSIENT_FILE_WAIT_SECONDS = 900.0


def _is_file(filename: str, *, wait_seconds: float = TRANSIENT_FILE_WAIT_SECONDS) -> bool:
    from ..storage.store import transient_read_error

    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return Path(filename).is_file()
        except OSError as error:
            if not transient_read_error(error) or time.monotonic() >= deadline:
                raise
        time.sleep(min(10.0, max(0.0, deadline - time.monotonic())))


def _require_executable(executable: str) -> None:
    if not Path(executable).is_file() or not os.access(executable, os.X_OK):
        raise PreflightError(f"required executable is unavailable: {executable}")


def _requirements(environment: Mapping[str, Any]) -> dict[str, str]:
    requirements = environment["requirements"]
    observed: dict[str, str] = {}
    _require_executable(environment["python"])
    if Path(sys.executable).resolve() != Path(environment["python"]).resolve():
        raise PreflightError(
            f"runner interpreter differs: expected {environment['python']}, got {sys.executable}"
        )
    for executable in requirements["executables"]:
        _require_executable(executable)
        observed[f"executable:{executable}"] = "available"
    for filename in requirements["files"]:
        if not _is_file(filename):
            raise PreflightError(f"required file is unavailable: {filename}")
        observed[f"file:{filename}"] = "available"
    import_paths = _profile_import_paths(environment)
    # The runner itself is intentionally launched with ``python -I``. Add the
    # profile's explicit import roots for package checks so preflight observes
    # the same source handoff that the managed child will receive.
    for import_path in reversed(import_paths):
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
    for module_name, expected_version in requirements["packages"].items():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or exc.name == module_name.split(".", 1)[0]:
                raise PreflightError(
                    f"required package is not importable: {module_name}"
                ) from exc
            raise PreflightError(
                f"required package is missing a dependency: {module_name}: {exc.name}"
            ) from exc
        except ImportError as exc:
            raise PreflightError(
                f"required package is incompatible: {module_name}: {exc}"
            ) from exc
        actual_version = str(getattr(module, "__version__", "unknown"))
        if expected_version != "*" and actual_version != expected_version:
            raise PreflightError(
                f"{module_name} version differs: expected {expected_version}, got {actual_version}"
            )
        observed[f"package:{module_name}"] = actual_version
    return observed


def _profile_import_paths(environment: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the explicit profile import roots in deterministic order."""

    settings = environment["environment"]
    values = settings["set"]
    pythonpath = values.get("PYTHONPATH", "")
    if not isinstance(pythonpath, str):
        raise PreflightError("environment.set.PYTHONPATH must be a string")
    return tuple(item for item in pythonpath.split(os.pathsep) if item)


def _isolated_python_argv(
    argv: list[str],
    import_paths: tuple[str, ...],
    *,
    runtime_path: str | None = None,
    runtime_sha256: str | None = None,
) -> list[str]:
    """Route isolated Python children through the explicit-path bootstrap."""

    if (
        not import_paths
        or runtime_path is None
        or runtime_sha256 is None
        or not Path(argv[0]).name.startswith("python")
    ):
        return argv
    if "-I" not in argv[1:]:
        return argv
    original = [item for item in argv[1:] if item != "-I"]
    return [
        argv[0],
        "-I",
        "-c",
        _ISOLATED_CHILD_BOOTSTRAP,
        runtime_path,
        runtime_sha256,
        json.dumps(list(import_paths), separators=(",", ":")),
        "--",
        *original,
    ]


def _verify_config(document: Mapping[str, Any]) -> None:
    config = document.get("config")
    if config is None:
        return
    content = Path(config["path"]).read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != config["sha256"]:
        raise PreflightError(
            f"config digest differs: expected {config['sha256']}, got {actual_sha256}"
        )


def _run_checked(command: list[str], *, timeout: float = 180) -> None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreflightError(f"preflight command could not run: {command[0]}: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout)[-4000:].strip()
        raise PreflightError(
            f"preflight command failed ({result.returncode}): {command[0]}: {detail}"
        )


def _c_compiler() -> None:
    compiler = os.environ.get("CC", "")
    _require_executable(compiler)
    with tempfile.TemporaryDirectory(prefix="koochak-cc-") as directory:
        source = Path(directory) / "probe.c"
        output = Path(directory) / "probe"
        source.write_text("int main(void) { return 0; }\n")
        _run_checked([compiler, str(source), "-o", str(output)], timeout=30)


def _cuda() -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise PreflightError("CUDA is not available to the workload")
    value = torch.zeros(1, device="cuda")
    if float(value.item()) != 0.0:
        raise PreflightError("CUDA allocation probe returned an unexpected value")
    torch.cuda.synchronize()


def _torch_compile() -> None:
    source = """
import torch

@torch.compile(fullgraph=True)
def probe(value):
    return value + 1

result = probe(torch.zeros(1, device="cuda"))
torch.cuda.synchronize()
assert result.item() == 1
"""
    _run_checked([sys.executable, "-I", "-c", source])


_PREFLIGHTS = {
    "c_compiler": _c_compiler,
    "cuda": _cuda,
    "torch_compile": _torch_compile,
}


def _write_result(filename: str, document: Mapping[str, Any]) -> None:
    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o444)
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _declared_outputs(document: Mapping[str, Any]) -> tuple[DeclaredOutput, ...]:
    raw = document.get("outputs", [])
    if not isinstance(raw, list):
        raise PreflightError("launch manifest outputs must be a JSON array")
    outputs = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise PreflightError("launch manifest output must be an object")
        try:
            outputs.append(DeclaredOutput.from_dict(value))
        except (TypeError, ValueError) as exc:
            raise PreflightError(f"invalid declared output: {exc}") from exc
    artifact_ids = [output.artifact_id for output in outputs]
    if len(set(artifact_ids)) != len(artifact_ids):
        raise PreflightError("launch manifest outputs must have unique artifact IDs")
    return tuple(outputs)


# A rank that crossed the final barrier marks the run evacuated within
# milliseconds; a SIGTERM racing that mark waits this long for it.
_TERMINATED_MARKER_WAIT_SECONDS = 0.5


class _EvacuationRelay:
    """Record SIGUSR1 for the runner's whole life; never act inside the handler.

    Installing it before preflight keeps an early evacuation request from
    terminating the runner with the signal's default action.

    Once a request reaches a preemptible workload, the relay also answers
    SIGTERM. In a multi-node Slurm step, ``srun --kill-on-bad-exit`` terminates
    every other node's runner as soon as the first exits 75, and srun reports a
    signalled task ahead of any exit code. A runner whose workload has
    evacuated therefore exits 75 instead of dying by the signal.
    """

    def __init__(self) -> None:
        self.requested = False
        self.evacuated = False
        self._previous: Any = signal.SIG_DFL
        self._previous_terminate: Any = None
        self._control_dir: Path | None = None
        self._process_group: int | None = None

    def _record(self, _signum: int, _frame: Any) -> None:
        self.requested = True

    def __enter__(self) -> "_EvacuationRelay":
        self._previous = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._record)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._previous_terminate is not None:
            signal.signal(signal.SIGTERM, self._previous_terminate)
            self._previous_terminate = None
        signal.signal(signal.SIGUSR1, self._previous)

    def watch(self, control_dir: Path, process_group: int) -> None:
        """Answer SIGTERM with the evacuation exit once the workload evacuates."""

        self._control_dir = control_dir
        self._process_group = process_group
        if self._previous_terminate is None:
            previous = signal.getsignal(signal.SIGTERM)
            self._previous_terminate = signal.SIG_DFL if previous is None else previous
            signal.signal(signal.SIGTERM, self._terminate)

    def release(self, *, evacuated: bool) -> None:
        """Forget the workload's group and control directory once it is gone."""

        self.evacuated = self.evacuated or evacuated
        self._control_dir = None
        self._process_group = None

    def _has_evacuated(self) -> bool:
        return self.evacuated or (
            self._control_dir is not None and (self._control_dir / EVACUATED_FILE).exists()
        )

    def _terminate(self, signum: int, frame: Any) -> None:
        deadline = time.monotonic() + _TERMINATED_MARKER_WAIT_SECONDS
        while not self._has_evacuated() and time.monotonic() < deadline:
            time.sleep(0.02)
        if self._has_evacuated():
            # The terminal checkpoint is published and every rank has crossed
            # the final barrier; nothing the workload still does is needed.
            if self._process_group is not None:
                try:
                    os.killpg(self._process_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            os.write(2, b'{"kind": "evacuated_runner_terminated"}\n')
            os._exit(EVACUATION_EXIT_CODE)
        # Not an evacuation: let SIGTERM have the effect it had before.
        previous = self._previous_terminate
        signal.signal(signal.SIGTERM, previous)
        self._previous_terminate = None
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_DFL:
            os.kill(os.getpid(), signum)


def _note(kind: str, **fields: Any) -> None:
    print(json.dumps({"kind": kind, **fields}, sort_keys=True), file=sys.stderr, flush=True)


def _preemptible(document: Mapping[str, Any]) -> bool:
    policy = document.get("preemption")
    if policy is None:
        return False
    if not isinstance(policy, Mapping) or set(policy) != {
        "max_attempts",
        "retry_on",
        "evacuation",
    }:
        raise PreflightError("launch manifest preemption must be a recovery policy object")
    return True


# An evacuated workload's leader normally exits at once. A launcher such as
# torchrun instead reports a failure and may restart the ranks, so a leader that
# outlives the first grace period is stopped, and killed after the second.
_EVACUATED_EXIT_GRACE_SECONDS = 10.0
_EVACUATED_KILL_GRACE_SECONDS = 30.0


def _run_managed_child(
    argv: list[str],
    *,
    cwd: str,
    environment: Mapping[str, str],
    relay: _EvacuationRelay,
    preemptible: bool,
    import_paths: tuple[str, ...] = (),
    runtime_path: str | None = None,
    runtime_sha256: str | None = None,
) -> int:
    """Run one child process group and pass evacuation to a preemptible workload.

    The runner never signals its workload. For a preemptible run it names a
    private directory in ``EVACUATION_DIR_ENV`` and creates ``requested`` there;
    every process that inherits the variable, torchrun ranks included, sees the
    request at its next safe boundary. Once the workload creates ``evacuated``,
    its terminal checkpoint is published: the run reports the evacuation exit
    code whatever its launcher returns, and a leader that lingers is stopped.
    """

    if relay.requested and preemptible:
        return EVACUATION_EXIT_CODE
    with tempfile.TemporaryDirectory(prefix="koochak-evacuation-") as control:
        control_dir = Path(control)
        child_environment = dict(environment)
        if preemptible:
            child_environment[EVACUATION_DIR_ENV] = control
        requested = False
        noted = False
        evacuated_at: float | None = None
        stops: list[tuple[float, signal.Signals]] = []
        try:
            child = subprocess.Popen(
                _isolated_python_argv(
                    argv,
                    import_paths,
                    runtime_path=runtime_path,
                    runtime_sha256=runtime_sha256,
                ),
                cwd=cwd,
                env=child_environment,
                start_new_session=True,
            )
            while True:
                try:
                    returncode = child.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if relay.requested and not requested:
                    if preemptible:
                        relay.watch(control_dir, child.pid)
                        (control_dir / EVACUATION_REQUEST_FILE).touch()
                        _note("evacuation_requested")
                        requested = True
                    elif not noted:
                        _note("evacuation_ignored", reason="run_not_preemptible")
                        noted = True
                if not preemptible:
                    continue
                if evacuated_at is None and (control_dir / EVACUATED_FILE).exists():
                    evacuated_at = time.monotonic()
                    stops = [
                        (_EVACUATED_EXIT_GRACE_SECONDS, signal.SIGTERM),
                        (
                            _EVACUATED_EXIT_GRACE_SECONDS + _EVACUATED_KILL_GRACE_SECONDS,
                            signal.SIGKILL,
                        ),
                    ]
                if stops and time.monotonic() - evacuated_at >= stops[0][0]:
                    _, stop_signal = stops.pop(0)
                    _note("evacuated_leader_stopped", signal=stop_signal.name)
                    try:
                        os.killpg(child.pid, stop_signal)
                    except ProcessLookupError:
                        pass
            evacuated = preemptible and (control_dir / EVACUATED_FILE).exists()
            _require_quiescent_group(child.pid, stopped=evacuated_at is not None and len(stops) < 2)
        except OSError as exc:
            raise PreflightError(f"managed workload could not run: {exc}") from exc
        finally:
            relay.release(evacuated=preemptible and (control_dir / EVACUATED_FILE).exists())
    # subprocess uses negative values for signal termination; shell-compatible
    # status keeps the child's nonzero outcome visible to the scheduler.
    status = 128 + abs(returncode) if returncode < 0 else returncode
    if evacuated and status != EVACUATION_EXIT_CODE:
        _note("evacuation_completed", workload_returncode=status)
        return EVACUATION_EXIT_CODE
    return status


def _group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _require_quiescent_group(
    process_group: int, *, cleanup_seconds: float = 1.0, stopped: bool = False
) -> None:
    """Reject and clean a process group whose leader exited before descendants.

    ``stopped`` means the runner signalled the group itself after the workload
    evacuated; descendants it left behind are then cleaned without failing.
    """

    if not _group_exists(process_group):
        return
    for cleanup_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process_group, cleanup_signal)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + cleanup_seconds
        while _group_exists(process_group) and time.monotonic() < deadline:
            time.sleep(0.01)
        if not _group_exists(process_group):
            break
    if _group_exists(process_group):
        raise PreflightError("managed workload process group could not be quiesced")
    if stopped:
        return
    raise PreflightError("managed workload left background descendants")


def _publish_outputs(outputs: tuple[DeclaredOutput, ...], environment: Mapping[str, str]) -> None:
    """Validate outputs and publish events only for Scruffy workers."""

    manifests = [validate_artifact(output) for output in outputs]
    root = environment.get("SCRUFFY_ROOT")
    job_id = environment.get("SCRUFFY_JOB_ID")
    publish_event = None
    if "SCRUFFY_ROOT" in environment or "SCRUFFY_JOB_ID" in environment:
        if not root or not job_id:
            raise PreflightError(
                "declared outputs require both SCRUFFY_ROOT and SCRUFFY_JOB_ID for Scruffy workers"
            )
        try:
            from scruffy import publish_event
        except ModuleNotFoundError as exc:
            if exc.name == "scruffy":
                raise PreflightError("declared outputs require the Scruffy client") from exc
            raise PreflightError("the installed Scruffy client is missing a dependency") from exc
        except ImportError as exc:
            raise PreflightError("the installed Scruffy client is incompatible") from exc
    for output, initial_manifest in zip(outputs, manifests):
        manifest = validate_artifact(output)
        if manifest != initial_manifest:
            raise PreflightError(
                f"declared output changed during publication: {output.artifact_id}"
            )
        if publish_event is None:
            continue
        identity = hashlib.sha256(
            f"{job_id}\0{output.artifact_id}\0{manifest['sha256']}".encode()
        ).hexdigest()
        publish_event(
            Path(root),
            job_id=job_id,
            kind="workload.artifact",
            event_id=f"koochak-artifact-{identity}",
            source={"name": "koochak-runner"},
            data={
                "artifact_type": output.stage or output.kind,
                "location": output.path,
                "publication": artifact_publication(manifest),
                "stage": output.stage,
                "provenance": dict(output.provenance),
                "counts": dict(manifest["counts"]),
                "metadata": dict(output.metadata),
            },
        )


def execute_manifest(manifest_path: str, expected_sha256: str) -> int:
    """Run checks, then execute and account for the declared workload."""

    with _EvacuationRelay() as relay:
        return _execute_manifest(manifest_path, expected_sha256, relay)


def _execute_manifest(
    manifest_path: str, expected_sha256: str, relay: _EvacuationRelay
) -> int:
    document = _load_manifest(manifest_path, expected_sha256)
    environment = document["environment"]
    secrets = set(environment["environment"]["secrets"])
    result_path = document["preflight_result_path"]
    result: dict[str, Any] = {
        "v": 1,
        "manifest_sha256": expected_sha256,
        "profile_id": environment["id"],
        "profile_sha256": environment["sha256"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        preemptible = _preemptible(document)
        clean_environment = build_environment(environment, os.environ)
        os.environ.clear()
        os.environ.update(clean_environment)
        for directory in environment["environment"]["create_directories"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        _verify_config(document)
        observed = _requirements(environment)
        completed = []
        for check in environment["preflight"]:
            _PREFLIGHTS[check]()
            completed.append(check)
        public_environment = {
            name: value for name, value in clean_environment.items() if name not in secrets
        }
        result.update(
            state="passed",
            checks=completed,
            observed=observed,
            environment_sha256=_sha256(public_environment),
        )
        _write_result(result_path, result)
    except Exception as exc:
        result.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        _write_result(result_path, result)
        raise

    outputs = _declared_outputs(document)
    child_returncode = _run_managed_child(
        document["argv"],
        cwd=document["cwd"],
        environment=clean_environment,
        relay=relay,
        preemptible=preemptible,
        import_paths=_profile_import_paths(environment),
        runtime_path=document["runner_runtime"]["path"],
        runtime_sha256=document["runner_runtime"]["sha256"],
    )
    if child_returncode:
        evacuated = preemptible and child_returncode == EVACUATION_EXIT_CODE
        result.update(
            state="evacuated" if evacuated else "failed",
            child_returncode=child_returncode,
        )
        _write_result(result_path, result)
        return child_returncode
    if outputs:
        try:
            _publish_outputs(outputs, clean_environment)
        except Exception as exc:
            result.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            _write_result(result_path, result)
            raise
    result["state"] = "completed" if outputs else "passed"
    if outputs:
        result["outputs"] = [output.artifact_id for output in outputs]
    _write_result(result_path, result)
    return 0


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m koochak.jobs.runner MANIFEST SHA256", file=sys.stderr)
        return 64
    try:
        return execute_manifest(sys.argv[1], sys.argv[2])
    except Exception as exc:  # noqa: BLE001 - this is the process error boundary.
        print(
            json.dumps(
                {"kind": "environment_preflight_failed", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
