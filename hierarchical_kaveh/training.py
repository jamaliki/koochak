"""The small Hierarchical Kaveh adapter to Koochak's training runtime."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import os
from pathlib import Path
from typing import Any, Mapping
import warnings

import torch
from torch import Tensor, nn

from koochak.core import dist as dist_lib
from koochak.core import hooks as hooks_lib
from koochak.core import launch
from koochak.logging.csv import make_csv_hooks
from koochak.logging.jsonl import make_jsonl_hooks
from koochak.logging.stdout import make_stdout_hooks
from koochak.loop import training_loop
from koochak.optim.build import build_optimizer
from koochak.storage import checkpoint as checkpoint_lib
from koochak.utils.seed import set_all_seeds

from .config import LossConfig, ModelConfig, RunConfig
from .data import build_train_dataloader
from .diffusion import compute_losses
from .model import HierarchicalKaveh
from .model.backend import require_fused
from .types import DenoiserInput, Prediction


def denoiser_input(
    batch: Mapping[str, Tensor],
    previous: Prediction | None = None,
) -> DenoiserInput:
    """Translate one standard-EDM batch into the immutable model contract."""

    inputs = DenoiserInput(
        coordinates=batch["x_t"],
        sigma=batch["t"],
        residue_index=batch["res_idx"],
        chain_index=batch["chain_idx"],
        chain_break=batch["chain_breaks_per_residue"],
        atom_mask=batch["atom14_mask"].to(torch.bool),
        aatype_input=batch["aatype_input"],
    )
    return inputs.with_self_conditioning(previous)


class PallatomTrainingStep:
    """One 100%-self-conditioned Pallatom training step."""

    def __init__(
        self,
        loss_config: LossConfig,
        model_config: ModelConfig,
    ) -> None:
        self.loss_config = loss_config
        self.model_config = model_config

    def __call__(
        self,
        model: nn.Module,
        batch: Mapping[str, Tensor],
        context: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        inputs = denoiser_input(batch)
        autocast = context["autocast"]
        with torch.no_grad(), autocast():
            previous = model(inputs, compute_distogram=False)
        inputs = denoiser_input(batch, previous)

        with autocast():
            output = model(inputs, compute_distogram=True)
            losses = compute_losses(
                output,
                inputs,
                batch,
                self.loss_config,
                self.model_config,
                patch_distogram_implementation="triton" if require_fused() else "auto",
            )

        return {
            "loss": losses["loss"],
            "coordinate_loss": losses["coordinate_loss"].detach(),
            "aatype_loss": losses["aatype_loss"].detach(),
            "smooth_lddt_loss": losses["smooth_lddt_loss"].detach(),
            "distogram_loss": losses["distogram_loss"].detach(),
            "self_conditioned": losses["loss"].new_ones(()),
            "node_count": batch["residue_mask"].sum(),
            "node_slot_count": batch["atom14_mask"].sum(),
        }


def _hooks(config: RunConfig) -> dict[str, list]:
    hooks = make_stdout_hooks()
    if config.logging.csv_path:
        hooks = hooks_lib.merge(hooks, make_csv_hooks(config.logging.csv_path))
    if config.logging.jsonl_path:
        hooks = hooks_lib.merge(hooks, make_jsonl_hooks(config.logging.jsonl_path))
    if config.wandb.enabled:
        from koochak.logging.wandb_logger import make_wandb_hooks

        hooks = hooks_lib.merge(hooks, make_wandb_hooks(asdict(config.wandb)))
    return hooks


def _resume_checkpoint(resume: str | Path | None, out_dir: str) -> dict[str, Any] | None:
    if resume is None:
        return None
    checkpoint_file = checkpoint_lib.latest(out_dir) if str(resume) == "latest" else str(resume)
    if checkpoint_file is None:
        raise FileNotFoundError(f"no checkpoint found in {out_dir}")
    return checkpoint_lib.load(checkpoint_file)


@contextmanager
def _performance_policy(config: RunConfig):
    """Make configured performance fallbacks explicit and process-local."""

    variable = "HIERARCHICAL_KAVEH_REQUIRE_FUSED"
    previous = os.environ.get(variable)
    if config.train.require_fused:
        os.environ[variable] = "1"
    try:
        with warnings.catch_warnings():
            if config.train.require_compile:
                warnings.filterwarnings(
                    "error",
                    message=r"torch\.compile failed; falling back to eager:.*",
                    category=RuntimeWarning,
                )
            yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def run_training(config: RunConfig, *, resume: str | Path | None = None) -> dict[str, Any]:
    """Train with Koochak and return the final checkpoint dictionary."""

    with _performance_policy(config):
        return _run_training(config, resume=resume)


def _run_training(config: RunConfig, *, resume: str | Path | None) -> dict[str, Any]:
    """Implementation kept separate so performance policy spans the full runtime."""

    plain_config = config.to_dict()
    launch_context = launch.initialize(plain_config)
    set_all_seeds(config.train.seed + int(launch_context["rank"]))

    model = HierarchicalKaveh(config.model)
    optimizer = build_optimizer(model, asdict(config.optimizer))
    checkpoint = _resume_checkpoint(resume, config.train.out_dir)
    global_step = 0 if checkpoint is None else int(
        checkpoint.get("next_step", int(checkpoint.get("step", 0)) + 1)
    )
    loader = build_train_dataloader(
        config.data,
        config.diffusion,
        sigma_data=config.model.sigma_data,
        global_step=global_step,
    )
    step = PallatomTrainingStep(config.loss, config.model)
    hooks = _hooks(config)
    final_checkpoint = training_loop(
        model=model,
        dataset=loader,
        step_fn=step,
        optimizer=optimizer,
        scheduler=None,
        train_cfg=plain_config["train"],
        config_json=plain_config,
        checkpoint_dict=checkpoint,
        hooks=hooks,
    )

    if config.train.save_final and dist_lib.rank0():
        step_number = int(final_checkpoint.get("next_step", final_checkpoint.get("step", 0)))
        checkpoint_file = Path(config.train.out_dir) / f"step{step_number:07d}.pt"
        checkpoint_lib.save(
            final_checkpoint,
            str(checkpoint_file),
            keep_last_k=config.train.keep_last_k,
        )
    dist_lib.barrier()
    return final_checkpoint


__all__ = ["PallatomTrainingStep", "denoiser_input", "run_training"]
