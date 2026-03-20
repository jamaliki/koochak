from __future__ import annotations

import torch
import pytest

from koochak.core import precision as precision_mod
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


def test_prepare_compile_backend_patches_dynamo_tf32_state_key(monkeypatch):
    if not hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        pytest.skip("torch lacks backend fp32_precision API")

    try:
        from torch._dynamo import graph_region_tracker
    except Exception as exc:  # pragma: no cover - version dependent
        pytest.skip(f"torch._dynamo.graph_region_tracker unavailable: {exc}")

    original = graph_region_tracker.get_global_state_key

    def _legacy_getter():
        raise AssertionError("legacy TF32 getter should not be called")

    monkeypatch.setattr(torch._C, "_get_cublas_allow_tf32", _legacy_getter)
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")

    try:
        precision_mod.prepare_compile_backend()
        key = graph_region_tracker.get_global_state_key()
        assert key[7] is False

        monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "tf32")
        key = graph_region_tracker.get_global_state_key()
        assert key[7] is True
    finally:
        graph_region_tracker.get_global_state_key = original
