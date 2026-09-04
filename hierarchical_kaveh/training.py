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
from koochak.logging.events import make_scruffy_hooks
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


_BATCH_TELEMETRY = (
    "data_worker_id",
    "data_owned_shard_count",
    "data_cached_shard_count",
    "data_cache_hit_count",
    "data_cache_miss_count",
    "data_cache_bytes",
    "data_mixture_strict_count",
    "data_mixture_broader_count",
    "data_mixture_strict_cumulative_count",
    "data_mixture_broader_cumulative_count",
    "data_mixture_strict_pool_count",
    "data_mixture_broader_pool_count",
    "data_mixture_strict_worker_fallback",
    "data_mixture_broader_worker_fallback",
    "koochak_prefetch_cpu_fetch_time_s",
    "koochak_prefetch_get_wait_s",
    "koochak_prefetch_event_ready",
    "koochak_prefetch_queue_depth",
    "koochak_prefetch_cpu_queue_depth",
    "koochak_prefetch_age_s",
    "koochak_prefetch_prepare_submit_s",
)


def denoiser_input(
    batch: Mapping[str, Any],
    previous: Prediction | None = None,
) -> DenoiserInput:
    """Translate one standard-EDM batch into the immutable model contract."""

    inputs = DenoiserInput(
        coordinates=batch["x_t"],
        sigma=batch["t"],
        residue_index=batch["res_idx"],
        chain_index=batch["chain_idx"],
        chain_break=batch["chain_breaks_per_residue"],
        atom_mask=batch["model_atom_mask"].to(torch.bool),
        aatype_input=batch["aatype_input"],
        secondary_structure_input=batch.get("secondary_structure_input"),
        patch_capacity=batch.get("patch_capacity"),
    )
    return inputs.with_self_conditioning(previous)


def _self_conditioning_model(model: nn.Module) -> nn.Module:
    """Use eager inference for the optional auxiliary denoiser pass."""

    unwrapped = getattr(model, "module", model)
    return getattr(unwrapped, "_orig_mod", unwrapped)


class PallatomTrainingStep:
    """One 100%-self-conditioned Pallatom training step."""

    def __init__(
        self,
        loss_config: LossConfig,
        model_config: ModelConfig,
        *,
        self_conditioning_probability: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.loss_config = loss_config
        self.model_config = model_config
        self.self_conditioning_probability = float(self_conditioning_probability)
        self.seed = int(seed)

    def _use_self_conditioning(self, step: int) -> bool:
        if self.self_conditioning_probability in {0.0, 1.0}:
            return bool(self.self_conditioning_probability)
        generator = torch.Generator().manual_seed(self.seed + int(step))
        return bool(torch.rand((), generator=generator) < self.self_conditioning_probability)

    def __call__(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        inputs = denoiser_input(batch)
        autocast = context["autocast"]
        use_self_conditioning = self._use_self_conditioning(int(context.get("step", 0)))
        previous = None
        if use_self_conditioning:
            with torch.no_grad(), autocast():
                previous = _self_conditioning_model(model)(
                    inputs, compute_distogram=False
                )
        inputs = denoiser_input(batch, previous)

        with autocast():
            model_kwargs = {"compute_distogram": self.loss_config.distogram_weight > 0.0}
            if (
                self.loss_config.intermediate_distogram_weight > 0.0
                or self.model_config.intermediate_distogram_feedback
            ):
                model_kwargs["compute_intermediate_distograms"] = True
            output = model(inputs, **model_kwargs)
            losses = compute_losses(
                output,
                inputs,
                batch,
                self.loss_config,
                self.model_config,
                patch_distogram_implementation="triton" if require_fused() else "auto",
            )

        metrics: dict[str, Tensor | float] = {
            "loss": losses["loss"],
            "coordinate_loss": losses["coordinate_loss"].detach(),
            "aatype_loss": losses["aatype_loss"].detach(),
            "aatype_marginal_js_loss": losses.get(
                "aatype_marginal_js_loss", losses["loss"].new_zeros(())
            ).detach(),
            "aatype_active_fraction": losses["aatype_active_fraction"].detach(),
            "aatype_sigma_weight_mean": losses.get(
                "aatype_sigma_weight_mean", losses["aatype_active_fraction"]
            ).detach(),
            "smooth_lddt_loss": losses["smooth_lddt_loss"].detach(),
            "distogram_loss": losses["distogram_loss"].detach(),
            "secondary_structure_loss": losses.get(
                "secondary_structure_loss", losses["loss"].new_zeros(())
            ).detach(),
            "self_conditioned": losses["loss"].new_tensor(float(use_self_conditioning)),
            "node_count": batch["residue_mask"].sum(),
            "node_slot_count": batch["model_atom_mask"].sum(),
            "supervised_slot_count": batch["coordinate_mask"].sum(),
        }
        if "intermediate_distogram_loss" in losses:
            metrics["intermediate_distogram_loss"] = losses["intermediate_distogram_loss"].detach()
        metrics.update({key: batch[key] for key in _BATCH_TELEMETRY if key in batch})
        return metrics


def _hooks(config: RunConfig) -> dict[str, list]:
    hooks = make_stdout_hooks()
    if config.logging.csv_path:
        hooks = hooks_lib.merge(hooks, make_csv_hooks(config.logging.csv_path))
    if config.logging.jsonl_path:
        hooks = hooks_lib.merge(hooks, make_jsonl_hooks(config.logging.jsonl_path))
    if config.wandb.enabled:
        from koochak.logging.wandb_logger import make_wandb_hooks

        hooks = hooks_lib.merge(hooks, make_wandb_hooks(asdict(config.wandb)))
    scruffy_root = os.environ.get("SCRUFFY_ROOT")
    scruffy_job_id = os.environ.get("SCRUFFY_JOB_ID")
    if bool(scruffy_root) != bool(scruffy_job_id):
        raise RuntimeError(
            "Scruffy worker identity requires both SCRUFFY_ROOT and SCRUFFY_JOB_ID"
        )
    if scruffy_root and scruffy_job_id:
        # Append after local logging hooks. Koochak emits on_checkpoint only
        # after the immutable checkpoint and ready publication are complete.
        hooks = hooks_lib.merge(hooks, make_scruffy_hooks())
    return hooks


def _resume_checkpoint(resume: str | Path | None, out_dir: str) -> dict[str, Any] | None:
    # Koochak owns validated auto-resume selection.  Do not load a candidate
    # here: doing so would bypass its publication checks and retry event.
    if resume is None or str(resume).lower() == "auto":
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
    launch_context = (
        launch.initialize(plain_config)
        if config.train.ddp
        else {"rank": 0, "world_size": 1, "is_rank0": True, "device": None}
    )
    set_all_seeds(config.train.seed + int(launch_context["rank"]))

    model = HierarchicalKaveh(config.model)
    optimizer = build_optimizer(model, asdict(config.optimizer))
    resume_mode = None if resume is None else str(resume).lower()
    auto_resume = (
        checkpoint_lib.resolve_auto_resume(config.train.out_dir)
        if resume_mode == "auto"
        else None
    )
    checkpoint = auto_resume[1] if auto_resume is not None else _resume_checkpoint(
        resume, config.train.out_dir
    )
    global_step = 0 if checkpoint is None else int(
        checkpoint.get("next_step", int(checkpoint.get("step", 0)) + 1)
    )
    loader = build_train_dataloader(
        config.data,
        config.diffusion,
        sigma_data=config.model.sigma_data,
        global_step=global_step,
    )
    step = PallatomTrainingStep(
        config.loss,
        config.model,
        self_conditioning_probability=config.train.self_conditioning_probability,
        seed=config.train.seed,
    )
    hooks = _hooks(config)
    loop_kwargs: dict[str, Any] = {
        "model": model,
        "dataset": loader,
        "step_fn": step,
        "optimizer": optimizer,
        "scheduler": None,
        "train_cfg": plain_config["train"],
        "config_json": plain_config,
        "checkpoint_dict": checkpoint,
        "hooks": hooks,
    }
    # The strict project config is the source of truth for this opt-in.  Do
    # not pass an override for the default-disabled case: Koochak still reads
    # the config field, while ordinary launches retain their legacy call
    # shape and signal behavior.
    if config.train.evacuation_enabled:
        loop_kwargs["evacuation"] = True
    if resume_mode == "auto":
        loop_kwargs["resume"] = "auto"
        if auto_resume is not None:
            loop_kwargs["auto_resume_path"] = auto_resume[0]
    final_checkpoint = training_loop(
        **loop_kwargs,
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
