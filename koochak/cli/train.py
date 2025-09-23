from __future__ import annotations

import argparse
import importlib
import os
from typing import Any, Dict

from kaveh.koochak.loop import training_loop
from kaveh.koochak.core import hooks as hooks_lib
from kaveh.koochak.logging.stdout import make_stdout_hooks
from kaveh.koochak.logging.csv import make_csv_hooks
from kaveh.koochak.logging.jsonl import make_jsonl_hooks
from kaveh.koochak.logging.wandb_logger import make_wandb_hooks
from kaveh.koochak.optim.build import build_optimizer, build_scheduler
from kaveh.koochak.utils.seed import set_all_seeds
from kaveh.koochak.utils import config as config_lib
from kaveh.koochak.storage import checkpoint as checkpoint_lib

try:
    from omegaconf import OmegaConf  # type: ignore
except Exception as _e:
    OmegaConf = None


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

    if yaml is None:
        raise RuntimeError("omegaconf is required. Please `pip install omegaconf`.")

    cfg_all: Dict[str, Any] = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)  # type: ignore

    # Sections
    cfg_train = dict(cfg_all.get("train", {}))
    cfg_data = dict(cfg_all.get("data", {}))
    cfg_optim = dict(cfg_all.get("optim", {}))
    cfg_logging = dict(cfg_all.get("logging", {}))
    cfg_wandb = cfg_all.get("wandb")
    cfg_entry = dict(cfg_all.get("entry", {}))

    # Strict config summary and validation (default strict)
    config_lib.apply_hybrid_validation_pre(cfg_all, cfg_train, schema=config_lib.default_schema())

    # Seed
    set_all_seeds(int(cfg_train.get("seed", 42)))

    # Build model
    if "model" not in cfg_entry:
        raise ValueError("entry.model must be provided as 'module:callable' or 'module.Class'")
    model_fn = _import_obj(cfg_entry["model"])  # type: ignore[index]
    model = model_fn()  # no-arg call

    # Build optimizer + scheduler from YAML optim section
    opt = build_optimizer(model.parameters(), cfg_optim.get("optimizer"))
    sched = build_scheduler(opt, cfg_optim.get("scheduler"), cfg_train)

    # Dataset and step functions
    if "dataset" not in cfg_entry or "step" not in cfg_entry:
        raise ValueError("entry.dataset and entry.step must be provided")
    dataset_fn = _import_obj(cfg_entry["dataset"])  # type: ignore[index]
    step_fn = _import_obj(cfg_entry["step"])  # type: ignore[index]
    dataset = dataset_fn()

    # Optional eval dataset + fn
    eval_dataset = None
    eval_fn = None
    if "eval_dataset" in cfg_entry:
        eval_dataset = _import_obj(cfg_entry["eval_dataset"])()
    if "eval_fn" in cfg_entry:
        eval_fn = _import_obj(cfg_entry["eval_fn"])

    # Hooks (stdout, CSV/JSONL, optional W&B)
    hooks = hooks_lib.merge({}, make_stdout_hooks())
    csv_path = cfg_logging.get("csv_path") or os.path.join(str(cfg_train.get("out_dir", "./runs/exp")), "log.csv")
    jsonl_path = cfg_logging.get("jsonl_path") or os.path.join(str(cfg_train.get("out_dir", "./runs/exp")), "log.jsonl")
    if csv_path:
        hooks = hooks_lib.merge(hooks, make_csv_hooks(csv_path))
    if jsonl_path:
        hooks = hooks_lib.merge(hooks, make_jsonl_hooks(jsonl_path))
    if cfg_wandb and isinstance(cfg_wandb, dict) and cfg_wandb.get("enabled", False):
        hooks = hooks_lib.merge(hooks, make_wandb_hooks(cfg_wandb))

    # Resume
    latest = checkpoint_lib.latest(str(cfg_train.get("out_dir", "./runs/exp")))
    ckpt = checkpoint_lib.load(latest) if latest and os.path.exists(latest) else None

    # Train
    training_loop(
        model=model,
        dataset=dataset,
        step_fn=step_fn,
        optimizer=opt,
        scheduler=sched,
        config=cfg_train,
        checkpoint_dict=ckpt,
        eval_dataset=eval_dataset,
        eval_fn=eval_fn,
        hooks=hooks,
    )

    # Post-run unused reporting
    config_lib.apply_hybrid_validation_post_train(cfg_train)


if __name__ == "__main__":
    main()
