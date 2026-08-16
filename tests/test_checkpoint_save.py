from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from koochak.storage import checkpoint as ckpt


def test_checkpoint_save_and_prune(tmp_path: Path = Path("tests/_tmp_ckpt")):
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def save_step(step: int) -> Path:
        c = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": None,
            "scaler": None,
            "config": {"out_dir": str(tmp_path)},
            "rng": {},
            "wall_time": 0.0,
            "metrics": {"val_loss": float(1000 - step)},
        }
        path = tmp_path / f"step{step:09d}.pt"
        assert ckpt.save(c, str(path), keep_last_k=2) == str(path)
        publication = ckpt.publication(str(path))
        assert publication == {
            "v": 1,
            "artifact_id": f"checkpoint/{path.name}",
            "path": str(path.absolute()),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "manifest_path": ckpt.publication_path(str(path)),
        }
        return path

    save_step(1)
    save_step(2)
    save_step(3)
    files = sorted([p.name for p in tmp_path.glob("step*.pt")])
    assert files == ["step000000002.pt", "step000000003.pt"]
    manifests = sorted([p.name for p in tmp_path.glob("step*.pt.ready.json")])
    assert manifests == [
        "step000000002.pt.ready.json",
        "step000000003.pt.ready.json",
    ]
