from __future__ import annotations

from .slurm import SshSlurmBackend
from .specs import (
    ConfigPatch,
    JobHandle,
    JobStatus,
    RemotePaths,
    RenderedJob,
    RuntimeFlags,
    SlurmResources,
    TrainJobSpec,
    materialize_config,
)
from .ssh import CommandResult, SshCommandRunner

__all__ = [
    "CommandResult",
    "ConfigPatch",
    "JobHandle",
    "JobStatus",
    "RemotePaths",
    "RenderedJob",
    "RuntimeFlags",
    "SlurmResources",
    "SshCommandRunner",
    "SshSlurmBackend",
    "TrainJobSpec",
    "materialize_config",
]
