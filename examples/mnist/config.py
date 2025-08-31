from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class WandBConfig:
    enabled: bool = False
    project: str = "koochak"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    mode: Literal["online", "offline"] = "online"
    dir: Optional[str] = None
    resume: Literal["never", "allow", "must"] = "allow"
    id: Optional[str] = None
    log_artifacts: bool = True


@dataclass
class TrainConfig:
    # Loop
    max_steps: int = 1_000
    log_every: int = 50
    eval_every: int = 200
    ckpt_every: int = 200

    # Optim / schedule
    grad_accum: int = 1
    grad_clip_norm: Optional[float] = None
    scheduler_step: Literal["step", "eval", "never"] = "step"

    # Precision
    amp: Literal["fp32", "fp16", "bf16"] = "fp32"

    # Distributed
    ddp: bool = False
    ddp_backend: Literal["nccl", "gloo", "mpi"] = "nccl"
    find_unused_parameters: bool = False

    # Misc
    seed: int = 42
    compile: bool = False
    device: str = "cuda"
    out_dir: str = "./runs/mnist"
    keep_last_k: int = 3

    # Logging
    wandb: Optional[WandBConfig] = None
    stdout_logging: bool = True

