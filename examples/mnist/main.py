from __future__ import annotations

import argparse
import contextlib
import os
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import datasets, transforms
except Exception as e:
    datasets = None
    transforms = None

from koochak.loop import training_loop
from koochak.storage import checkpoint as checkpoint_lib
from koochak.core import hooks as hooks_lib
from koochak.logging.stdout import make_stdout_hooks
from koochak.logging.csv import make_csv_hooks
from koochak.logging.jsonl import make_jsonl_hooks
from koochak.logging.wandb_logger import make_wandb_hooks
from koochak.data.iterable import cycle
from koochak.optim.build import build_optimizer, build_scheduler
from koochak.utils.seed import set_all_seeds, make_worker_init_fn
from koochak import config as config_lib


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.head = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        return self.head(x)


def step_fn(model: nn.Module, batch: Mapping[str, Any], ctx: Mapping[str, Any]) -> Dict[str, Any]:
    x = batch["x"]
    y = batch["y"]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    acc = (logits.argmax(dim=1) == y).float().mean()
    return {"loss": loss, "acc": acc}


@torch.no_grad()
def eval_fn(model: nn.Module, iterable: Iterable, ctx: Mapping[str, Any]) -> Dict[str, float]:
    model.eval()
    device = ctx.get("device")
    total, n = 0.0, 0
    ac = ctx.get("autocast")
    cm = ac if ac is not None else contextlib.nullcontext()
    with cm:
        for batch in iterable:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logits = model(x)
            total += F.cross_entropy(logits, y).item()
            n += 1
    return {"val_loss": total / max(1, n)}


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int, seed: int):
    if datasets is None:
        raise RuntimeError("torchvision not available; install torchvision to run MNIST example")
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)

    def collate(batch):
        xs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return {"x": xs, "y": ys}

    # Rank-aware worker seeding (rank=0 in single-process case)
    from koochak.core import dist as dist_lib
    rank = dist_lib.rank()

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        worker_init_fn=make_worker_init_fn(seed, rank=rank),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        worker_init_fn=make_worker_init_fn(seed + 1, rank=rank),
    )
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"), help="YAML config path")
    args = parser.parse_args()

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

    set_all_seeds(int(cfg_train.seed))

    model = SmallCNN()
    opt = build_optimizer(model.parameters(), cfg_optim.optimizer)
    sched = build_scheduler(opt, cfg_optim.scheduler, cfg_train)

    train_loader, test_loader = build_dataloaders(
        str(cfg_data.data_dir), int(cfg_data.batch_size), int(cfg_data.num_workers), int(cfg_train.seed))

    # Dataset can be finite; we cycle it to satisfy max_steps
    train_iter = cycle(train_loader)

    hooks = hooks_lib.merge({}, make_stdout_hooks())
    # CSV/JSONL logging
    csv_path = cfg_logging.csv_path or os.path.join(str(cfg_train.out_dir), "log.csv")
    jsonl_path = cfg_logging.jsonl_path or os.path.join(str(cfg_train.out_dir), "log.jsonl")
    if csv_path:
        hooks = hooks_lib.merge(hooks, make_csv_hooks(csv_path))
    if jsonl_path:
        hooks = hooks_lib.merge(hooks, make_jsonl_hooks(jsonl_path))
    wb = cfg_wandb
    if wb and getattr(wb, "enabled", False):
        hooks = hooks_lib.merge(hooks, make_wandb_hooks(wb))

    # Resume if available
    latest = checkpoint_lib.latest(str(cfg_train.out_dir))
    ckpt = checkpoint_lib.load(latest) if latest and os.path.exists(latest) else None

    result = training_loop(
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


if __name__ == "__main__":
    main()
