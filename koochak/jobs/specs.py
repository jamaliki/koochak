from __future__ import annotations

import json
import posixpath
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ConfigPatch:
    """A dotted-path OmegaConf update applied to a base config."""

    path: str
    value: Any
    merge: bool = True

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("ConfigPatch.path must be a non-empty dotted path")


@dataclass(frozen=True)
class SlurmResources:
    """Slurm resource request for one job or one array task."""

    partition: str | None = None
    nodes: int = 1
    tasks_per_node: int = 1
    gpus: int = 0
    cpus: int = 1
    mem_gb: int | None = None
    time: str = "01:00:00"
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    array: str | None = None
    dependency: str | None = None
    additional_directives: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError("SlurmResources.nodes must be >= 1")
        if self.tasks_per_node < 1:
            raise ValueError("SlurmResources.tasks_per_node must be >= 1")
        if self.gpus < 0:
            raise ValueError("SlurmResources.gpus must be >= 0")
        if self.cpus < 1:
            raise ValueError("SlurmResources.cpus must be >= 1")
        if self.mem_gb is None or int(self.mem_gb) <= 0:
            raise ValueError("SlurmResources.mem_gb must be set explicitly and be > 0")
        if not str(self.time).strip():
            raise ValueError("SlurmResources.time must be non-empty")


@dataclass(frozen=True)
class RemotePaths:
    """Remote paths used by a Slurm training job."""

    repo: str
    run_root: str
    python: str = "python"
    path_prefixes: Sequence[str] = ()

    def run_dir_for(self, name: str) -> str:
        explicit = _safe_job_name(name)
        return posixpath.join(self.run_root.rstrip("/"), explicit)


@dataclass(frozen=True)
class RuntimeFlags:
    """Common runtime environment flags plus arbitrary project-specific env."""

    env: Mapping[str, Any] = field(default_factory=dict)
    python_unbuffered: bool = True
    omp_num_threads: int | None = None
    wandb_mode: str | None = None
    wandb_disabled: bool | None = None
    torchinductor_cache: bool = True
    triton_cache: bool = True

    def environment(self, run_dir: str) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.python_unbuffered:
            out["PYTHONUNBUFFERED"] = "1"
        if self.omp_num_threads is not None:
            out["OMP_NUM_THREADS"] = str(int(self.omp_num_threads))
        if self.wandb_mode is not None:
            out["WANDB_MODE"] = str(self.wandb_mode)
        if self.wandb_disabled is not None:
            out["WANDB_DISABLED"] = "true" if self.wandb_disabled else "false"
        if self.torchinductor_cache:
            out["TORCHINDUCTOR_CACHE_DIR"] = posixpath.join(run_dir, "torchinductor_cache")
        if self.triton_cache:
            out["TRITON_CACHE_DIR"] = posixpath.join(run_dir, "triton_cache")
        for key, value in self.env.items():
            key = str(key)
            if not _ENV_NAME_RE.match(key):
                raise ValueError(f"Invalid environment variable name: {key!r}")
            out[key] = str(value)
        return out


@dataclass(frozen=True)
class TrainJobSpec:
    """A config-driven Python training command to run under Slurm."""

    name: str
    base_config: str | Path
    command: Sequence[str]
    resources: SlurmResources
    patches: Sequence[ConfigPatch] = ()
    runtime: RuntimeFlags = field(default_factory=RuntimeFlags)
    run_dir: str | None = None
    config_filename: str = "config.yaml"
    sbatch_filename: str = "job.sbatch"
    manifest_filename: str = "manifest.json"
    out_dir_config_key: str | None = "train.out_dir"
    workdir: str | None = None

    def __post_init__(self) -> None:
        _safe_job_name(self.name)
        if not self.command:
            raise ValueError("TrainJobSpec.command must contain Python arguments")
        for filename in (self.config_filename, self.sbatch_filename, self.manifest_filename):
            if "/" in filename or filename in {"", ".", ".."}:
                raise ValueError(f"Job artifact filename must be a plain filename: {filename!r}")


@dataclass(frozen=True)
class RenderedJob:
    name: str
    run_dir: str
    config_path: str
    sbatch_path: str
    manifest_path: str
    stdout_path: str
    stderr_path: str
    config_text: str
    sbatch_text: str
    manifest_text: str

    def write_local(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / self.config_path.rsplit("/", 1)[-1]).write_text(self.config_text)
        (root / self.sbatch_path.rsplit("/", 1)[-1]).write_text(self.sbatch_text)
        (root / self.manifest_path.rsplit("/", 1)[-1]).write_text(self.manifest_text)


@dataclass(frozen=True)
class JobHandle:
    job_id: str
    name: str
    run_dir: str
    config_path: str
    sbatch_path: str
    manifest_path: str
    stdout_path: str
    stderr_path: str


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: str
    source: str
    elapsed: str | None = None
    time_limit: str | None = None
    reason: str | None = None
    exit_code: str | None = None
    job_name: str | None = None


def materialize_config(
    base_config: str | Path,
    *,
    run_dir: str,
    patches: Sequence[ConfigPatch] = (),
    out_dir_config_key: str | None = "train.out_dir",
) -> str:
    """Load a base YAML config, apply patches, and return YAML text."""

    cfg = OmegaConf.load(str(base_config))
    if out_dir_config_key:
        OmegaConf.update(cfg, out_dir_config_key, run_dir, merge=True, force_add=True)
    for patch in patches:
        OmegaConf.update(cfg, patch.path, patch.value, merge=patch.merge, force_add=True)
    return OmegaConf.to_yaml(cfg, resolve=True)


def manifest_for(job: TrainJobSpec, rendered: RenderedJob, remote_paths: RemotePaths) -> str:
    payload = {
        "name": job.name,
        "run_dir": rendered.run_dir,
        "config_path": rendered.config_path,
        "sbatch_path": rendered.sbatch_path,
        "stdout_path": rendered.stdout_path,
        "stderr_path": rendered.stderr_path,
        "base_config": str(job.base_config),
        "command": list(job.command),
        "patches": [asdict(patch) for patch in job.patches],
        "resources": asdict(job.resources),
        "runtime": {
            **asdict(job.runtime),
            "env": dict(job.runtime.env),
        },
        "remote_paths": {
            "repo": remote_paths.repo,
            "run_root": remote_paths.run_root,
            "python": remote_paths.python,
            "path_prefixes": list(remote_paths.path_prefixes),
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _safe_job_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name).strip()).strip("-")
    if not cleaned:
        raise ValueError("Job name must contain at least one safe character")
    return cleaned[:128]
