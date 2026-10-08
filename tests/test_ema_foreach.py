"""Compare grouped resident EMA updates with the per-parameter operations."""

from unittest.mock import Mock

import pytest
import torch

from koochak.utils.ema import EMA


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("update_every", [1, 3])
def test_grouped_updates_match_two_operations_and_resume(device, dtype, update_every, monkeypatch):
    torch.manual_seed(173)
    model = torch.nn.ParameterList([
        torch.nn.Parameter(torch.randn(7, 11, device=device, dtype=torch.float32)),
        torch.nn.Parameter(torch.randn(4, 9, device=device, dtype=torch.bfloat16).T),
        torch.nn.Parameter(torch.randn((), device=device, dtype=torch.float64)),
        torch.nn.Parameter(torch.empty(0, device=device)),
        torch.nn.Parameter(torch.randn(3, device=device), requires_grad=False),
    ])
    ema = EMA(model, decay=.987, dtype=dtype, update_every=update_every, offload_to_cpu=False)
    expected = {name: tensor.clone() for name, tensor in ema.shadow.items()}
    frozen_before = model[-1].detach().clone()
    multiply, add = Mock(wraps=torch._foreach_mul_), Mock(wraps=torch._foreach_add_)
    monkeypatch.setattr(torch, "_foreach_mul_", multiply)
    monkeypatch.setattr(torch, "_foreach_add_", add)
    for step in range(1, 10):
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(step * .007)
        ema.update(model)
        if step % update_every == 0:
            decay = .987 ** update_every
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    expected[name].mul_(decay).add_(parameter.detach().to(dtype), alpha=1 - decay)
        for name, value in expected.items():
            torch.testing.assert_close(ema.shadow[name], value, atol=0, rtol=0)
        if step == 5:
            saved = ema.state_dict(clone=True)
            ema = EMA(model, decay=.2, dtype=dtype, update_every=update_every, offload_to_cpu=False)
            ema.load_state_dict(saved)
    expected_calls = 0 if dtype == torch.bfloat16 else 9 // update_every
    assert multiply.call_count == add.call_count == expected_calls
    assert all(len(call.args[0]) == 4 for call in multiply.call_args_list)
    assert set(ema.shadow) == {"0", "1", "2", "3"}
    assert ema.num_updates == 9 // update_every
    assert ema._step_counter == ema._last_update_step_counter == 9
    torch.testing.assert_close(model[-1], frozen_before, atol=0, rtol=0)


@pytest.mark.parametrize("frozen", [False, True])
def test_empty_shadow_does_not_call_foreach(frozen, monkeypatch):
    model = torch.nn.ParameterList([
        torch.nn.Parameter(torch.ones(3), requires_grad=False)] if frozen else [])
    ema = EMA(model, decay=.9, offload_to_cpu=False)
    multiply, add = Mock(), Mock()
    monkeypatch.setattr(torch, "_foreach_mul_", multiply)
    monkeypatch.setattr(torch, "_foreach_add_", add)
    ema.update(model)
    assert not ema.shadow
    assert ema.num_updates == ema._step_counter == ema._last_update_step_counter == 1
    multiply.assert_not_called()
    add.assert_not_called()


def test_async_snapshot_does_not_enter_resident_update(monkeypatch):
    model = torch.nn.Linear(2, 3)
    ema = EMA(model, decay=.9, offload_to_cpu=False)
    snapshot, multiply, add = Mock(), Mock(), Mock()
    ema._copy_device, ema._copy_stream = torch.device("cuda:0"), object()
    monkeypatch.setattr(ema, "_launch_async_cuda_snapshot", snapshot)
    monkeypatch.setattr(torch, "_foreach_mul_", multiply)
    monkeypatch.setattr(torch, "_foreach_add_", add)
    ema.update(model)
    snapshot.assert_called_once_with(model, .9)
    assert ema._last_update_step_counter == 1
    multiply.assert_not_called()
    add.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resident_updates_group_multiple_devices(monkeypatch):
    model = torch.nn.ParameterList([
        torch.nn.Parameter(torch.zeros(3, device=device)) for device in ("cpu", "cuda")])
    ema = EMA(model, decay=.9, offload_to_cpu=False)
    multiply = Mock(wraps=torch._foreach_mul_)
    monkeypatch.setattr(torch, "_foreach_mul_", multiply)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(2)
    ema.update(model)
    assert multiply.call_count == 2
    assert {call.args[0][0].device.type for call in multiply.call_args_list} == {"cpu", "cuda"}
    for shadow in ema.shadow.values():
        torch.testing.assert_close(shadow, torch.full_like(shadow, .2), atol=0, rtol=0)
