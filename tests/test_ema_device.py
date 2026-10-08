import io

import pytest
import torch

from koochak.utils.ema import EMA


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("offload", [False, True])
def test_shadow_device_update_resume_and_frozen_exclusion(device, offload):
    model = torch.nn.Linear(4, 2, device=device, dtype=torch.bfloat16)
    model.bias.requires_grad_(False)
    with torch.no_grad():
        model.weight.fill_(1)
        model.bias.fill_(7)
    ema = EMA(model, decay=.5, offload_to_cpu=offload)
    destination = torch.device("cpu") if offload else model.weight.device
    assert set(ema.shadow) == {"weight"}
    assert ema.shadow["weight"].device == destination
    assert ema.shadow["weight"].dtype == torch.float32
    assert bool(ema.staging) == offload
    if not offload:
        assert ema._copy_stream is None and ema._shadow_executor is None
    with torch.no_grad():
        model.weight.fill_(3)
    ema.update(model)
    state = ema.state_dict(clone=True)
    torch.testing.assert_close(state["shadow"]["weight"], torch.full_like(ema.shadow["weight"], 2))
    assert state["shadow"]["weight"].data_ptr() != ema.shadow["weight"].data_ptr()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    assert checkpoint["shadow"]["weight"].device.type == "cpu"
    resumed = EMA(model, decay=.9, offload_to_cpu=offload)
    resumed.load_state_dict(checkpoint)
    assert resumed.shadow["weight"].device == destination
    assert resumed.decay == .5 and resumed.num_updates == 1
    assert set(resumed.shadow) == {"weight"}
    with torch.no_grad():
        model.weight.fill_(6)
    resumed.update(model)
    resumed.store(model)
    resumed.copy_to(model)
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 4))
    torch.testing.assert_close(model.bias, torch.full_like(model.bias, 7))
    resumed.restore(model)
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 6))
    torch.testing.assert_close(model.bias, torch.full_like(model.bias, 7))
    assert resumed.num_updates == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_loading_waits_for_pending_cpu_offload():
    model = torch.nn.Linear(4, 2, device="cuda")
    ema = EMA(model, decay=.5, offload_to_cpu=True)
    state = ema.state_dict(clone=True)
    for value in state["shadow"].values():
        value.fill_(12)
    ema.update(model)
    ema.load_state_dict(state)
    for value in ema.state_dict()["shadow"].values():
        torch.testing.assert_close(value, torch.full_like(value, 12))
