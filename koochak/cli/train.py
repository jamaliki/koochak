from __future__ import annotations

import argparse
import importlib
import os
from typing import Any

from ..loop import training_loop
from ..core import hooks as hooks_lib
from ..logging.stdout import make_stdout_hooks
from ..logging.csv import make_csv_hooks
from ..logging.jsonl import make_jsonl_hooks
from ..logging.wandb_logger import make_wandb_hooks
from ..optim.build import build_optimizer, build_scheduler
from ..utils.seed import set_all_seeds
from .. import config as config_lib
from ..storage import checkpoint as checkpoint_lib
from ..integrations.morbo import create_integration as create_morbo_integration


def _import_obj(path: str):
    if ":" in path:
        mod, name = path.split(":", 1)
    elif "." in path:
        mod, name = path.rsplit(".", 1)
    else:
        raise ValueError(f"Invalid import path: {path}")
    m = importlib.import_module(mod)
    return getattr(m, name)


def main():
    parser = argparse.ArgumentParser(description="Koochak generic training CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg_all = config_lib.load_config(args.config)

    # Sections
    cfg_train = config_lib.get_section(cfg_all, "train")
    cfg_data = config_lib.get_section(cfg_all, "data")
    cfg_optim = config_lib.get_section(cfg_all, "optim")
    cfg_logging = config_lib.get_section(cfg_all, "logging")
    cfg_wandb = config_lib.get_section(cfg_all, "wandb")
    cfg_morbo = config_lib.get_section(cfg_all, "morbo", required=False)
    cfg_entry = config_lib.get_section(cfg_all, "entry")

    # Strict config summary and validation (default strict)
    config_lib.summarize(
        cfg_all,
        strict=bool(getattr(cfg_train, "strict_config", True)),
        warn_unknown=bool(getattr(cfg_train, "config_warn_unknown", True)),
    )

    # Seed
    set_all_seeds(int(getattr(cfg_train, "seed", 42)))

    # Build model
    if not getattr(cfg_entry, "model", None):
        raise ValueError("entry.model must be provided as 'module:callable' or 'module.Class'")
    model_fn = _import_obj(cfg_entry.model)
    model = model_fn()  # no-arg call

    # Build optimizer + scheduler from YAML optim section
    opt = build_optimizer(model.parameters(), cfg_optim.optimizer)
    sched = build_scheduler(opt, cfg_optim.scheduler, cfg_train)

    # Dataset and step functions
    if not getattr(cfg_entry, "dataset", None) or not getattr(cfg_entry, "step", None):
        raise ValueError("entry.dataset and entry.step must be provided")
    dataset_fn = _import_obj(cfg_entry.dataset)
    step_fn = _import_obj(cfg_entry.step)
    dataset = dataset_fn()

    # Optional eval dataset + fn
    eval_dataset = None
    eval_fn = None
    if getattr(cfg_entry, "eval_dataset", None):
        eval_dataset = _import_obj(cfg_entry.eval_dataset)()
    if getattr(cfg_entry, "eval_fn", None):
        eval_fn = _import_obj(cfg_entry.eval_fn)

    # Resume
    latest = checkpoint_lib.latest(str(cfg_train.out_dir))
    ckpt = checkpoint_lib.load(latest) if latest and os.path.exists(latest) else None

    config_json = config_lib.as_dict(cfg_all)

    # Hooks (stdout, CSV/JSONL, optional W&B and Morbo)
    hooks = hooks_lib.merge({}, make_stdout_hooks())
    csv_path = cfg_logging.csv_path or os.path.join(str(cfg_train.out_dir), "log.csv")
    jsonl_path = cfg_logging.jsonl_path or os.path.join(str(cfg_train.out_dir), "log.jsonl")
    if csv_path:
        hooks = hooks_lib.merge(hooks, make_csv_hooks(csv_path))
    if jsonl_path:
        hooks = hooks_lib.merge(hooks, make_jsonl_hooks(jsonl_path))
    if cfg_wandb and getattr(cfg_wandb, "enabled", False):
        hooks = hooks_lib.merge(hooks, make_wandb_hooks(cfg_wandb))
    morbo = create_morbo_integration(cfg_morbo, cfg_train, ckpt)
    if morbo is not None:
        hooks = hooks_lib.merge(hooks, morbo.hooks)
        config_json.setdefault("morbo", {}).update(
            {
                "project_id": morbo.identity.project_id,
                "run_id": morbo.identity.run_id,
                "attempt_id": morbo.identity.attempt_id,
                "run_name": morbo.identity.run_name,
                "identity_path": morbo.identity.identity_path,
            }
        )

    # Train
    try:
        training_loop(
            model=model,
            dataset=dataset,
            step_fn=step_fn,
            optimizer=opt,
            scheduler=sched,
            train_cfg=cfg_train,
            config_json=config_json,
            checkpoint_dict=ckpt,
            eval_dataset=eval_dataset,
            eval_fn=eval_fn,
            hooks=hooks,
        )
    finally:
        if morbo is not None:
            morbo.client.close()


if __name__ == "__main__":
    main()
