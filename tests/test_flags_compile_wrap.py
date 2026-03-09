from __future__ import annotations

import torch

from koochak.utils import flags


class _ToyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @flags.compile_wrap
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + 1


def test_compile_wrap_returns_original_bound_method_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SEE_MORE_ALPHA_USE_COMPILE", "1")
    prev_enabled = flags.get_compile_wrap_enabled()
    flags.set_compile_wrap_enabled(False)
    try:
        module = _ToyModule()
        bound = module.forward

        assert bound.__self__ is module
        assert bound.__func__ is _ToyModule.forward.function

        x = torch.tensor([1.0])
        y = bound(x)

        assert torch.equal(y, torch.tensor([2.0]))
        assert module.calls == 1
    finally:
        flags.set_compile_wrap_enabled(prev_enabled)
