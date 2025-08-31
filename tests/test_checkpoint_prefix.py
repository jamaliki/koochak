from __future__ import annotations

import torch.nn as nn
from koochak.storage import checkpoint as ckpt


class Wrapped:
    def __init__(self, module: nn.Module):
        self.module = module

    def state_dict(self):
        return {f"module.{k}": v for k, v in self.module.state_dict().items()}


def test_prefix_helpers_roundtrip():
    m = nn.Linear(3, 2)
    sd = m.state_dict()
    wrapped_sd = {f"module.{k}": v for k, v in sd.items()}
    assert ckpt.strip_module_prefix(wrapped_sd) == sd
    assert ckpt.add_module_prefix(sd) == wrapped_sd


def test_match_state_dict_to_model_matches_target():
    m = nn.Linear(4, 5)
    w = Wrapped(m)
    sd_plain = m.state_dict()
    sd_ddp = w.state_dict()
    adj1 = ckpt.match_state_dict_to_model(w, sd_plain)
    assert set(adj1.keys()) == set(sd_ddp.keys())
    adj2 = ckpt.match_state_dict_to_model(m, sd_ddp)
    assert set(adj2.keys()) == set(sd_plain.keys())

