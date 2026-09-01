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
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..storage.artifact import (
    DeclaredOutput,
    artifact_publication,
    validate_artifact,
)


class PreflightError(RuntimeError):
    """The declared execution environment did not satisfy its contract."""


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
        if not Path(filename).is_file():
            raise PreflightError(f"required file is unavailable: {filename}")
        observed[f"file:{filename}"] = "available"
    for module_name, expected_version in requirements["packages"].items():
        module = importlib.import_module(module_name)
        actual_version = str(getattr(module, "__version__", "unknown"))
        if expected_version != "*" and actual_version != expected_version:
            raise PreflightError(
                f"{module_name} version differs: expected {expected_version}, got {actual_version}"
            )
        observed[f"package:{module_name}"] = actual_version
    return observed


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


def _run_managed_child(
    argv: list[str],
    *,
    cwd: str,
    environment: Mapping[str, str],
) -> int:
    """Run one child process group and forward evacuation only to that group."""

    requested = False
    forwarded = False
    previous_handler = signal.getsignal(signal.SIGUSR1)

    def _request(_signum: int, _frame: Any) -> None:
        nonlocal requested
        requested = True

    signal.signal(signal.SIGUSR1, _request)
    try:
        child = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(environment),
            start_new_session=True,
        )
        while True:
            try:
                returncode = child.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if requested and not forwarded:
                    os.killpg(child.pid, signal.SIGUSR1)
                    forwarded = True
    except OSError as exc:
        raise PreflightError(f"managed workload could not run: {exc}") from exc
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)
    # subprocess uses negative values for signal termination; shell-compatible
    # status keeps the child's nonzero outcome visible to the scheduler.
    return 128 + abs(returncode) if returncode < 0 else returncode


def _publish_outputs(outputs: tuple[DeclaredOutput, ...], environment: Mapping[str, str]) -> None:
    """Validate all outputs before publishing any typed artifact event."""

    manifests = [validate_artifact(output) for output in outputs]
    try:
        from scruffy import publish_event
    except ModuleNotFoundError as exc:
        if exc.name == "scruffy":
            raise PreflightError("declared outputs require the Scruffy client") from exc
        raise PreflightError("the installed Scruffy client is missing a dependency") from exc
    except ImportError as exc:
        raise PreflightError("the installed Scruffy client is incompatible") from exc

    root = environment.get("SCRUFFY_ROOT")
    job_id = environment.get("SCRUFFY_JOB_ID")
    if not root or not job_id:
        raise PreflightError(
            "declared outputs require SCRUFFY_ROOT and SCRUFFY_JOB_ID runtime identity"
        )
    for output, manifest in zip(outputs, manifests):
        identity = hashlib.sha256(
            f"{output.artifact_id}\0{manifest['sha256']}".encode()
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
    )
    if child_returncode:
        result.update(state="failed", child_returncode=child_returncode)
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
