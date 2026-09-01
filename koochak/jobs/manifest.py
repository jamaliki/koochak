"""Compile experiment inputs and an environment profile into one launch."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from omegaconf import OmegaConf

from .profile import EnvironmentProfile
from ..storage.artifact import DeclaredOutput
from ..storage.immutable import write_immutable_file

_JOB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _absolute(value: str, name: str) -> str:
    if not isinstance(value, str) or not PurePosixPath(value).is_absolute() or "\x00" in value:
        raise ValueError(f"{name} must be an absolute POSIX path")
    return value


@dataclass(frozen=True, slots=True)
class ConfigPatch:
    """A dotted OmegaConf update applied before a launch is prepared."""

    path: str
    value: Any
    merge: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("patch path must be non-empty")


@dataclass(frozen=True, slots=True)
class Artifact:
    """One immutable file required by a prepared run."""

    path: str
    content: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class PreparedRun:
    """A backend-independent Python launch and its immutable artifacts."""

    name: str
    cwd: str
    run_dir: str
    python: str
    manifest_path: str
    manifest_sha256: str
    artifacts: tuple[Artifact, ...]
    declared_outputs: tuple[DeclaredOutput, ...] = ()

    def runner_argv(self) -> list[str]:
        return [
            self.python,
            "-I",
            "-m",
            "koochak.jobs.runner",
            self.manifest_path,
            self.manifest_sha256,
        ]


def materialize_config(
    base_config: str | Path,
    *,
    run_dir: str,
    patches: Sequence[ConfigPatch] = (),
    out_dir_config_key: str | None = "train.out_dir",
) -> str:
    """Resolve a training YAML document and return its final text."""

    config = OmegaConf.load(str(base_config))
    if out_dir_config_key:
        OmegaConf.update(config, out_dir_config_key, run_dir, merge=True, force_add=True)
    for patch in patches:
        OmegaConf.update(
            config, patch.path, patch.value, merge=patch.merge, force_add=True
        )
    return OmegaConf.to_yaml(config, resolve=True)


def _resolve_python_args(
    arguments: Sequence[str], *, cwd: str, run_dir: str, config_path: str | None
) -> list[str]:
    if isinstance(arguments, (str, bytes)) or not arguments:
        raise ValueError("python_args must contain at least one argument")
    replacements = {"{cwd}": cwd, "{run_dir}": run_dir}
    if config_path is not None:
        replacements["{config}"] = config_path
    resolved = []
    for argument in arguments:
        if not isinstance(argument, str) or not argument or "\x00" in argument:
            raise ValueError("python_args must contain non-empty strings without NUL bytes")
        value = argument
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        if "{config}" in value:
            raise ValueError("{config} requires base_config")
        resolved.append(value)
    return resolved


def prepare_run(
    *,
    name: str,
    profile: EnvironmentProfile,
    python_args: Sequence[str],
    cwd: str,
    run_dir: str,
    base_config: str | Path | None = None,
    patches: Sequence[ConfigPatch] = (),
    out_dir_config_key: str | None = "train.out_dir",
    declared_outputs: Sequence[DeclaredOutput] = (),
) -> PreparedRun:
    """Create a deterministic launch manifest without performing I/O remotely."""

    if not _JOB_NAME.fullmatch(name):
        raise ValueError("name must contain only letters, digits, '.', '_' or '-'")
    cwd = _absolute(cwd, "cwd")
    run_dir = _absolute(run_dir, "run_dir").rstrip("/") or "/"
    manifest_path = posixpath.join(run_dir, "launch.json")
    config_path = posixpath.join(run_dir, "config.yaml") if base_config is not None else None
    config_text = (
        materialize_config(
            base_config,
            run_dir=run_dir,
            patches=patches,
            out_dir_config_key=out_dir_config_key,
        )
        if base_config is not None
        else None
    )
    environment = profile.resolve(cwd=cwd, run_dir=run_dir)
    outputs = tuple(declared_outputs)
    output_ids = [output.artifact_id for output in outputs]
    if len(set(output_ids)) != len(output_ids):
        raise ValueError("declared outputs must have unique artifact IDs")
    argv = [
        environment.python,
        *_resolve_python_args(
            python_args, cwd=cwd, run_dir=run_dir, config_path=config_path
        ),
    ]
    document: dict[str, Any] = {
        "v": 1,
        "name": name,
        "cwd": cwd,
        "run_dir": run_dir,
        "argv": argv,
        "environment": environment.to_dict(),
        "preflight_result_path": posixpath.join(run_dir, "preflight.json"),
        "outputs": [output.to_dict() for output in outputs],
    }
    artifacts = []
    if config_text is not None and config_path is not None:
        config_content = config_text.encode()
        config_sha256 = _sha256(config_content)
        document["config"] = {
            "path": config_path,
            "sha256": config_sha256,
        }
        artifacts.append(Artifact(config_path, config_content, config_sha256))
    manifest_content = (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    manifest_sha256 = _sha256(manifest_content)
    artifacts.append(Artifact(manifest_path, manifest_content, manifest_sha256))
    return PreparedRun(
        name=name,
        cwd=cwd,
        run_dir=run_dir,
        python=environment.python,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        artifacts=tuple(artifacts),
        declared_outputs=outputs,
    )


def _write_immutable(artifact: Artifact) -> None:
    write_immutable_file(artifact.path, artifact.content)


def stage_run(prepared: PreparedRun) -> None:
    """Write a prepared run atomically on a directly mounted filesystem."""

    for artifact in prepared.artifacts:
        _write_immutable(artifact)
