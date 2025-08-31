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
from koochak.utils import config as config_lib

try:
    import yaml  # type: ignore
except Exception as _e:
    yaml = None  # will error at runtime with a helpful message


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

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        worker_init_fn=make_worker_init_fn(seed),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        worker_init_fn=make_worker_init_fn(seed + 1),
    )
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"), help="YAML config path")
    args = parser.parse_args()

    if yaml is None:
        raise RuntimeError("pyyaml is required for the MNIST example. Please `pip install pyyaml`. ")

    with open(args.config, "r") as f:
        cfg_all: Dict[str, Any] = yaml.safe_load(f) or {}

    cfg_train = dict(cfg_all.get("train", {}))
    cfg_data = dict(cfg_all.get("data", {}))
    cfg_optim = dict(cfg_all.get("optim", {}))
    cfg_wandb = cfg_all.get("wandb")
    cfg_logging = dict(cfg_all.get("logging", {}))

    # Schema validation (unknown keys)
    schema = config_lib.default_schema()
    unknown = config_lib.schema_validate(cfg_all, schema)
    if unknown:
        msg = "Unknown config keys: " + ", ".join(sorted(unknown))
        if bool(cfg_train.get("strict_config", False)):
            raise ValueError(msg)
        else:
            print("[koochak][config][warn]", msg)

    # Defaults for train
    cfg_train.setdefault("max_steps", 1000)
    cfg_train.setdefault("log_every", 50)
    cfg_train.setdefault("eval_every", 200)
    cfg_train.setdefault("ckpt_every", 200)
    cfg_train.setdefault("amp", "fp32")
    cfg_train.setdefault("seed", 42)
    cfg_train.setdefault("device", "cuda")
    cfg_train.setdefault("out_dir", "./runs/mnist")
    cfg_train.setdefault("keep_last_k", 3)

    # Defaults for data and optim
    cfg_optim.setdefault("lr", 3e-4)
    cfg_data.setdefault("data_dir", "./data")
    cfg_data.setdefault("batch_size", 128)
    cfg_data.setdefault("num_workers", 4)

    set_all_seeds(int(cfg_train["seed"]))

    model = SmallCNN()
    opt = build_optimizer(model.parameters(), cfg_optim.get("optimizer"))
    sched = build_scheduler(opt, cfg_optim.get("scheduler"), cfg_train)

    train_loader, test_loader = build_dataloaders(
        str(cfg_data["data_dir"]), int(cfg_data["batch_size"]), int(cfg_data["num_workers"]), int(cfg_train["seed"]))

    # Dataset can be finite; we cycle it to satisfy max_steps
    train_iter = cycle(train_loader)

    hooks = hooks_lib.merge({}, make_stdout_hooks())
    # CSV/JSONL logging
    csv_path = cfg_logging.get("csv_path") or os.path.join(str(cfg_train["out_dir"]), "log.csv")
    jsonl_path = cfg_logging.get("jsonl_path") or os.path.join(str(cfg_train["out_dir"]), "log.jsonl")
    if csv_path:
        hooks = hooks_lib.merge(hooks, make_csv_hooks(csv_path))
    if jsonl_path:
        hooks = hooks_lib.merge(hooks, make_jsonl_hooks(jsonl_path))
    wb = cfg_wandb
    if wb and isinstance(wb, dict) and wb.get("enabled", False):
        hooks = hooks_lib.merge(hooks, make_wandb_hooks(wb))

    # Resume if available
    latest = checkpoint_lib.latest(str(cfg_train["out_dir"]))
    ckpt = checkpoint_lib.load(latest) if latest and os.path.exists(latest) else None

    result = training_loop(
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

    # Report unused keys in train section (usage-tracking)
    unused = config_lib.report_unused("train", cfg_train, cfg_train)
    if unused:
        msg = "Unused train.* keys: " + ", ".join(sorted(unused))
        if bool(cfg_train.get("strict_config", False)):
            raise ValueError(msg)
        elif bool(cfg_train.get("config_warn_unknown", True)):
            print("[koochak][config][warn]", msg)


if __name__ == "__main__":
    main()
