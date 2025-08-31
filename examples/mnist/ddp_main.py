from __future__ import annotations

import argparse
import os
import contextlib
from typing import Any, Dict

import torch

from koochak.core import dist as dist_lib
from .main import (
    SmallCNN,
    step_fn,
    eval_fn,
    build_dataloaders,
)
from koochak.loop import training_loop
from koochak.storage import checkpoint as checkpoint_lib
from koochak.core import hooks as hooks_lib
from koochak.logging.stdout import make_stdout_hooks
from koochak.logging.csv import make_csv_hooks
from koochak.logging.jsonl import make_jsonl_hooks
from koochak.data.iterable import cycle
from koochak.optim.build import build_optimizer, build_scheduler
from koochak.utils.seed import set_all_seeds
from koochak.utils import config as config_lib

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--backend", default="nccl")
    args = parser.parse_args()

    if yaml is None:
        raise RuntimeError("pyyaml is required. Please `pip install pyyaml`.")

    dist_lib.init_process_group(backend=args.backend)

    # Select local device when using CUDA
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())

    with open(args.config, "r") as f:
        cfg_all: Dict[str, Any] = yaml.safe_load(f) or {}

    cfg_train = dict(cfg_all.get("train", {}))
    cfg_data = dict(cfg_all.get("data", {}))
    cfg_optim = dict(cfg_all.get("optim", {}))
    cfg_wandb = cfg_all.get("wandb")
    cfg_logging = dict(cfg_all.get("logging", {}))
    # Schema validation
    schema = config_lib.default_schema()
    unknown = config_lib.schema_validate(cfg_all, schema)
    if unknown:
        msg = "Unknown config keys: " + ", ".join(sorted(unknown))
        if bool(cfg_train.get("strict_config", False)):
            raise ValueError(msg)
        else:
            print("[koochak][config][warn]", msg)

    # Force DDP on and device to current CUDA device (logical)
    cfg_train["ddp"] = True
    if torch.cuda.is_available():
        cfg_train["device"] = "cuda"

    # defaults
    cfg_train.setdefault("max_steps", 1000)
    cfg_train.setdefault("log_every", 50)
    cfg_train.setdefault("eval_every", 200)
    cfg_train.setdefault("ckpt_every", 200)
    cfg_train.setdefault("amp", "fp32")
    cfg_train.setdefault("seed", 42)
    cfg_train.setdefault("out_dir", "./runs/mnist")
    cfg_train.setdefault("keep_last_k", 3)

    cfg_optim.setdefault("optimizer", {"name": "AdamW", "lr": 3e-4})

    set_all_seeds(int(cfg_train["seed"]))

    model = SmallCNN()
    opt = build_optimizer(model.parameters(), cfg_optim.get("optimizer"))
    sched = build_scheduler(opt, cfg_optim.get("scheduler"), cfg_train)

    train_loader, test_loader = build_dataloaders(
        str(cfg_data.get("data_dir", "./data")),
        int(cfg_data.get("batch_size", 128)),
        int(cfg_data.get("num_workers", 4)),
        int(cfg_train["seed"]))

    train_iter = cycle(train_loader)

    hooks = hooks_lib.merge({}, make_stdout_hooks())
    csv_path = cfg_logging.get("csv_path") or os.path.join(str(cfg_train["out_dir"]), "log.csv")
    jsonl_path = cfg_logging.get("jsonl_path") or os.path.join(str(cfg_train["out_dir"]), "log.jsonl")
    if csv_path:
        hooks = hooks_lib.merge(hooks, make_csv_hooks(csv_path))
    if jsonl_path:
        hooks = hooks_lib.merge(hooks, make_jsonl_hooks(jsonl_path))
    if cfg_wandb and isinstance(cfg_wandb, dict) and cfg_wandb.get("enabled", False):
        from koochak.logging.wandb_logger import make_wandb_hooks

        hooks = hooks_lib.merge(hooks, make_wandb_hooks(cfg_wandb))

    latest = checkpoint_lib.latest(str(cfg_train["out_dir"]))
    ckpt = checkpoint_lib.load(latest) if latest and os.path.exists(latest) else None

    try:
        training_loop(
            model=model,
            dataset=train_iter,
            step_fn=step_fn,
            optimizer=opt,
            scheduler=sched,
            config=cfg_train,
            checkpoint_dict=ckpt,
            eval_dataset=test_loader,
            eval_fn=eval_fn,
            hooks=hooks,
        )
    finally:
        # Report unused keys in train section (usage tracking)
        try:
            unused = config_lib.report_unused("train", cfg_train, cfg_train)
            if unused:
                msg = "Unused train.* keys: " + ", ".join(sorted(unused))
                if bool(cfg_train.get("strict_config", False)):
                    raise ValueError(msg)
                elif bool(cfg_train.get("config_warn_unknown", True)):
                    print("[koochak][config][warn]", msg)
        except Exception:
            pass
        if dist_lib.is_initialized():
            import torch.distributed as dist

            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

