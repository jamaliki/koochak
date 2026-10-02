from __future__ import annotations

import math

import pytest
import torch

from koochak.loop import training_loop
from koochak.optim.grad_clip import clip_grad_norm_


def parameter(gradient):
    param = torch.nn.Parameter(torch.zeros_like(gradient))
    param.grad = gradient.clone()
    return param


@pytest.mark.parametrize("maximum", [0.0, 0.5, 1000.0])
def test_normal_gradients_match_torch(maximum):
    generator = torch.Generator().manual_seed(71)
    params = [parameter(torch.randn(size, generator=generator)) for size in (17, 1024, 32)]
    reference = [parameter(param.grad) for param in params]
    expected = torch.nn.utils.clip_grad_norm_(reference, maximum, foreach=True)
    result = clip_grad_norm_(iter(params), maximum)
    assert result.norm == pytest.approx(float(expected), rel=1e-6)
    assert not result.used_scaled_norm
    for actual, wanted in zip(params, reference):
        torch.testing.assert_close(actual.grad, wanted.grad)


@pytest.mark.parametrize("magnitude", [3.7167e18, 1e32, 3e38])
@pytest.mark.parametrize("maximum", [0.0, 1.0, 1e100])
def test_finite_large_gradients_match_float64(magnitude, maximum):
    values = torch.linspace(-1.0, 1.0, 4096) * magnitude
    params = [parameter(values), parameter(values[::2])]
    originals = [param.grad.clone() for param in params]
    norm64 = torch.cat([grad.double() for grad in originals]).norm()
    coefficient = min(1.0, maximum / (float(norm64) + 1e-6))
    result = clip_grad_norm_(params, maximum)
    assert result.norm == pytest.approx(float(norm64), rel=2e-6)
    assert result.coefficient == pytest.approx(coefficient, rel=2e-6, abs=0.0)
    assert result.used_scaled_norm
    for param, original in zip(params, originals):
        assert torch.isfinite(param.grad).all()
        torch.testing.assert_close(param.grad.double(), original.double() * coefficient,
                                   atol=1e-8, rtol=2e-6)
        if coefficient == 1.0:
            assert torch.equal(param.grad, original)
    if 0.0 < coefficient < 1.0:
        assert float(torch.cat([p.grad.double() for p in params]).norm()) == pytest.approx(maximum)


def test_scalar_large_gradient_avoids_subnormal_multiplier():
    param = parameter(torch.tensor([3e38]))
    result = clip_grad_norm_([param], 1.0)
    assert result.used_scaled_norm
    torch.testing.assert_close(param.grad, torch.ones(1))


def test_overflow_fallback_preserves_small_entries_when_maximum_is_large():
    param = parameter(torch.tensor([1e32, 1e-20]))
    result = clip_grad_norm_([param], 1e30)
    assert result.used_scaled_norm
    torch.testing.assert_close(param.grad, torch.tensor([1e30, 1e-22]), rtol=2e-6, atol=0.0)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_entries_raise_without_mutating_any_gradient(invalid):
    params = [parameter(torch.tensor([2.0, 3.0])), parameter(torch.tensor([1e32, invalid]))]
    originals = [p.grad.clone() for p in params]
    with pytest.raises(FloatingPointError, match="optimizer step stopped"):
        clip_grad_norm_(params, 1.0)
    for param, original in zip(params, originals):
        torch.testing.assert_close(param.grad, original, equal_nan=True, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("maximum", [-1.0, float("nan"), float("inf")])
def test_invalid_maximum_fails(maximum):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        clip_grad_norm_([], maximum)


def test_empty_and_zero_gradients():
    assert clip_grad_norm_([], 1.0).norm == 0.0
    params = [torch.nn.Parameter(torch.ones(3)), parameter(torch.zeros(5))]
    result = clip_grad_norm_(params, 1.0)
    assert result.norm == 0.0
    assert result.coefficient == 1.0
    assert params[0].grad is None
    assert torch.count_nonzero(params[1].grad) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_gradient_dtype_is_preserved(dtype):
    param = parameter(torch.arange(100, dtype=dtype))
    result = clip_grad_norm_([param], 1.0)
    assert math.isfinite(result.norm)
    assert param.grad.dtype == dtype
    assert float(param.grad.double().norm()) == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("check_every", [0, 1])
def test_loop_clips_before_adam_and_rejects_nonfinite(tmp_path, invalid, check_every):
    model = torch.nn.Linear(4, 4, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = model.weight.detach().clone()
    rows = []
    scale = float("inf") if invalid else 1e32

    def step(module, batch, ctx):
        return {"loss": module(batch).sum() * scale}

    def run():
        return training_loop(
            model=model, dataset=[torch.ones(2, 4)], step_fn=step, optimizer=optimizer,
            train_cfg=dict(max_steps=1, device="cpu", out_dir=str(tmp_path),
                           log_every=1, grad_clip_norm=1.0,
                           nonfinite_grad_check_every=check_every),
            hooks={"on_log": [lambda row, ctx: rows.append(row.copy())]},
        )

    if invalid:
        with pytest.raises(FloatingPointError, match="optimizer step stopped"):
            run()
        assert torch.equal(model.weight, before)
        assert not optimizer.state
    else:
        checkpoint = run()
        assert not torch.equal(model.weight, before)
        assert checkpoint["next_step"] == 1
        assert rows[0]["grad_norm"] == pytest.approx(8e32, rel=1e-6)
        assert rows[0]["grad_clip_coefficient"] == pytest.approx(1.25e-33, rel=1e-6, abs=0.0)
        assert rows[0]["grad_clip_scaled_norm"] == 1
        assert all(torch.isfinite(value).all() for state in optimizer.state.values()
                   for value in state.values() if isinstance(value, torch.Tensor))


def test_loop_accepts_a_torch_style_clip_replacement(tmp_path, monkeypatch):
    import koochak.loop as loop

    calls = []

    def torch_clip(parameters, max_norm, *args, **kwargs):
        calls.append(max_norm)
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    monkeypatch.setattr(loop, "clip_grad_norm_", torch_clip)
    model = torch.nn.Linear(4, 4, bias=False)
    rows = []
    checkpoint = training_loop(
        model=model, dataset=[torch.ones(2, 4)], step_fn=lambda m, b, c: {"loss": m(b).sum()},
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_cfg=dict(max_steps=1, device="cpu", out_dir=str(tmp_path), log_every=1, grad_clip_norm=1.0),
        hooks={"on_log": [lambda row, ctx: rows.append(row.copy())]},
    )
    assert calls == [1.0] and checkpoint["next_step"] == 1
    assert "grad_clip_coefficient" not in rows[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_large_gradients_match_float64():
    param = parameter(torch.linspace(-1.0, 1.0, 4096, device="cuda") * 3e38)
    original = param.grad.double()
    expected = original / original.norm()
    result = clip_grad_norm_([param], 1.0)
    assert result.used_scaled_norm
    torch.testing.assert_close(param.grad.double(), expected, rtol=2e-6, atol=1e-8)
