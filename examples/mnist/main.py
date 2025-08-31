from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torchvision import datasets, transforms
except Exception as e:
    datasets = None
    transforms = None

from code.loop import training_loop
from storage import checkpoint as checkpoint_lib
from koochak.core import hooks as hooks_lib
from koochak.logging.stdout import make_stdout_hooks
from koochak.logging.wandb_logger import make_wandb_hooks
from koochak.data.iterable import cycle
from koochak.utils.seed import set_all_seeds, make_worker_init_fn

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
    total, n = 0.0, 0
    for batch in iterable:
        logits = model(batch["x"])  # to_device handled by loop
        total += F.cross_entropy(logits, batch["y"]).item()
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
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    # Defaults if missing
    cfg.setdefault("max_steps", 1000)
    cfg.setdefault("log_every", 50)
    cfg.setdefault("eval_every", 200)
    cfg.setdefault("ckpt_every", 200)
    cfg.setdefault("amp", "fp32")
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "cuda")
    cfg.setdefault("out_dir", "./runs/mnist")
    cfg.setdefault("keep_last_k", 3)
    cfg.setdefault("lr", 3e-4)
    cfg.setdefault("data_dir", "./data")
    cfg.setdefault("batch_size", 128)
    cfg.setdefault("num_workers", 4)

    set_all_seeds(int(cfg["seed"]))

    model = SmallCNN()
    opt = AdamW(model.parameters(), lr=float(cfg["lr"]))
    sched = CosineAnnealingLR(opt, T_max=max(1, int(cfg["max_steps"])))

    train_loader, test_loader = build_dataloaders(
        str(cfg["data_dir"]), int(cfg["batch_size"]), int(cfg["num_workers"]), int(cfg["seed"]))

    # Dataset can be finite; we cycle it to satisfy max_steps
    train_iter = cycle(train_loader)

    hooks = hooks_lib.merge({}, make_stdout_hooks())
    wb = cfg.get("wandb")
    if wb and isinstance(wb, dict) and wb.get("enabled", False):
        hooks = hooks_lib.merge(hooks, make_wandb_hooks(wb))

    # Resume if available
    latest = checkpoint_lib.latest(str(cfg["out_dir"]))
    ckpt = checkpoint_lib.load(latest) if latest and os.path.exists(latest) else None

    training_loop(
        model=model,
        dataset=train_iter,
        step_fn=step_fn,
        optimizer=opt,
        scheduler=sched,
        config=cfg,
        checkpoint_dict=ckpt,
        eval_dataset=test_loader,
        eval_fn=eval_fn,
        hooks=hooks,
    )


if __name__ == "__main__":
    main()
