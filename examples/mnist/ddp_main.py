from __future__ import annotations

import argparse
import os
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
from koochak import config as config_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--backend", default="nccl")
    args = parser.parse_args()

    dist_lib.init_process_group(backend=args.backend)

    # Select local device when using CUDA
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())

    cfg_all = config_lib.load_config(args.config)

    cfg_train = config_lib.get_section(cfg_all, "train")
    cfg_data = config_lib.get_section(cfg_all, "data")
    cfg_optim = config_lib.get_section(cfg_all, "optim")
    cfg_wandb = config_lib.get_section(cfg_all, "wandb")
    cfg_logging = config_lib.get_section(cfg_all, "logging")
    # Pre-run summary + schema validation (strict by default)
    config_lib.summarize(
        cfg_all,
        strict=bool(getattr(cfg_train, "strict_config", True)),
        warn_unknown=bool(getattr(cfg_train, "config_warn_unknown", True)),
    )

    # Force DDP on and device to current CUDA device (logical)
    cfg_train.ddp = True
    if cfg_train.shard_dataset is False:
        cfg_train.shard_dataset = True
    if cfg_train.shard_dataset_mode is None:
        cfg_train.shard_dataset_mode = "iterable"
    if torch.cuda.is_available():
        cfg_train.device = "cuda"

    # Per-rank deterministic seeding to avoid correlated randomness
    set_all_seeds(int(cfg_train.seed) + dist_lib.rank())

    model = SmallCNN()
    opt = build_optimizer(model.parameters(), cfg_optim.optimizer)
    sched = build_scheduler(opt, cfg_optim.scheduler, cfg_train)

    train_loader, test_loader = build_dataloaders(
        str(cfg_data.data_dir),
        int(cfg_data.batch_size),
        int(cfg_data.num_workers),
        int(cfg_train.seed))

    train_iter = cycle(train_loader)

    hooks = hooks_lib.merge({}, make_stdout_hooks())
    csv_path = cfg_logging.csv_path or os.path.join(str(cfg_train.out_dir), "log.csv")
    jsonl_path = cfg_logging.jsonl_path or os.path.join(str(cfg_train.out_dir), "log.jsonl")
    if csv_path:
        hooks = hooks_lib.merge(hooks, make_csv_hooks(csv_path))
    if jsonl_path:
        hooks = hooks_lib.merge(hooks, make_jsonl_hooks(jsonl_path))
    if cfg_wandb and getattr(cfg_wandb, "enabled", False):
        from koochak.logging.wandb_logger import make_wandb_hooks

        hooks = hooks_lib.merge(hooks, make_wandb_hooks(cfg_wandb))

    latest = checkpoint_lib.latest(str(cfg_train.out_dir))
    ckpt = checkpoint_lib.load(latest) if latest and os.path.exists(latest) else None

    try:
        training_loop(
            model=model,
            dataset=train_iter,
            step_fn=step_fn,
            optimizer=opt,
            scheduler=sched,
            train_cfg=cfg_train,
            config_json=config_lib.as_dict(cfg_all),
            checkpoint_dict=ckpt,
            eval_dataset=test_loader,
            eval_fn=eval_fn,
            hooks=hooks,
        )
    finally:
        if dist_lib.is_initialized():
            import torch.distributed as dist

            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
