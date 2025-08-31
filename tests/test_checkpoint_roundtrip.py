from __future__ import annotations

from pathlib import Path
import shutil

import torch
import torch.nn as nn

from koochak.storage import checkpoint as ckpt


def _copy_state(sd):
    return {k: v.detach().clone() for k, v in sd.items()}


def test_save_load_roundtrip(tmp_path: Path = Path("tests/_tmp_roundtrip")):
    # fresh dir
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    # create a model and optimizer, then save
    m = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    orig = _copy_state(m.state_dict())

    ck = {
        "step": 1,
        "model": orig,
        "optimizer": opt.state_dict(),
        "scheduler": None,
        "scaler": None,
        "config": {"out_dir": str(tmp_path)},
        "rng": {},
        "wall_time": 0.0,
        "metrics": {},
    }
    path = tmp_path / "step000000001.pt"
    ckpt.save(ck, str(path), keep_last_k=2)

    # load into a new model instance
    m2 = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    loaded = ckpt.load(str(path))
    state = ckpt.match_state_dict_to_model(m2, loaded["model"])
    m2.load_state_dict(state)

    for (k, v), (k2, v2) in zip(orig.items(), m2.state_dict().items()):
        assert k == k2
        assert torch.allclose(v, v2)

