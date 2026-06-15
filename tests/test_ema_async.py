import copy

import pytest
import torch

from koochak.utils.ema import EMA


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _linear_cuda() -> torch.nn.Module:
    model = torch.nn.Linear(8, 8, bias=True).cuda()
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            param.fill_(0.1 * (i + 1))
    return model


def test_async_cuda_offload_matches_synchronous_ema_with_mutation_guard() -> None:
    model_async = _linear_cuda()
    model_sync = copy.deepcopy(model_async)
    ema_async = EMA(model_async, decay=0.7, offload_to_cpu=True, pin_memory=True)
    ema_sync = EMA(model_sync, decay=0.7, offload_to_cpu=False)

    for step in range(1, 6):
        ema_async.wait_before_param_mutation()
        with torch.no_grad():
            for param_async, param_sync in zip(model_async.parameters(), model_sync.parameters()):
                param_async.add_(float(step) * 0.01)
                param_sync.add_(float(step) * 0.01)
        ema_async.update(model_async)
        ema_sync.update(model_sync)

    async_shadow = ema_async.state_dict(clone=True)["shadow"]
    sync_shadow = ema_sync.state_dict(clone=True)["shadow"]
    for name, expected in sync_shadow.items():
        torch.testing.assert_close(async_shadow[name], expected, atol=0.0, rtol=0.0)


def test_wait_before_param_mutation_protects_async_snapshot() -> None:
    model = _linear_cuda()
    ema = EMA(model, decay=0.5, offload_to_cpu=True, pin_memory=True)
    initial = {name: value.clone() for name, value in ema.state_dict(clone=True)["shadow"].items()}

    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1.0)
    ema.update(model)
    ema.wait_before_param_mutation()

    with torch.no_grad():
        for param in model.parameters():
            param.fill_(100.0)

    shadow = ema.state_dict(clone=True)["shadow"]
    for name, before in initial.items():
        expected = before * 0.5 + torch.ones_like(before) * 0.5
        torch.testing.assert_close(shadow[name], expected, atol=0.0, rtol=0.0)
