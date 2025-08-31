from __future__ import annotations

import torch
from koochak.core.precision import Scaler, autocast_context


def test_scaler_scale_roundtrip():
    loss = torch.tensor(1.2345, requires_grad=True)
    s = Scaler("fp16")
    scaled = s.scale(loss)
    assert isinstance(scaled, torch.Tensor)
    assert scaled.shape == loss.shape


def test_autocast_context_enters_and_exits():
    # Should be a context manager for all modes
    for mode in ["fp32", "fp16", "bf16"]:
        cm = autocast_context(mode, torch.device("cpu"))
        # Just ensure it can enter/exit without error on CPU
        with cm:
            x = torch.randn(2, 2)
            y = x + 1
            assert y.shape == x.shape

