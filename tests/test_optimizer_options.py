from __future__ import annotations

import torch

import koochak.optim.build as build_mod


def test_adamw_passes_explicit_fused_and_foreach(monkeypatch) -> None:
    captured = {}

    def fake_adamw(params, **kwargs):
        captured["params"] = list(params)
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(build_mod, "AdamW", fake_adamw)
    model = torch.nn.Linear(2, 1)

    opt = build_mod.build_optimizer(
        model,
        {
            "name": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.01,
            "fused": True,
            "foreach": False,
        },
    )

    assert opt is not None
    assert captured["params"] == list(model.parameters())
    assert captured["kwargs"]["fused"] is True
    assert captured["kwargs"]["foreach"] is False
    assert captured["kwargs"]["lr"] == 1e-3
    assert captured["kwargs"]["weight_decay"] == 0.01


def test_adam_omits_fused_and_foreach_by_default(monkeypatch) -> None:
    captured = {}

    def fake_adam(params, **kwargs):
        captured["params"] = list(params)
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(build_mod, "Adam", fake_adam)
    model = torch.nn.Linear(2, 1)

    build_mod.build_optimizer(model.parameters(), {"name": "adam"})

    assert captured["params"] == list(model.parameters())
    assert "fused" not in captured["kwargs"]
    assert "foreach" not in captured["kwargs"]
