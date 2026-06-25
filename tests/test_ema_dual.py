from __future__ import annotations

import math
from copy import deepcopy
from typing import Mapping

import torch

from koochak.loop import training_loop
from koochak.utils.ema import EMA
from koochak.utils.ema_posthoc import (
    gamma_from_srel,
    ls_coeffs_for_power_profile,
    reconstruct_dual_power_ema_state_dict,
    reconstruct_power_ema_state_dict,
)


def _mlp() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(3, 8),
        torch.nn.SiLU(),
        torch.nn.Linear(8, 2),
    )


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _smooth_param_value(
    parameter: torch.Tensor,
    param_index: int,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    x = float(step) / float(total_steps)
    grid = torch.linspace(
        0.0,
        1.0,
        parameter.numel(),
        dtype=parameter.dtype,
        device=parameter.device,
    ).reshape_as(parameter)
    return (
        torch.sin(grid * float(param_index + 1) + x * math.pi * 0.7)
        + 0.25 * torch.cos(grid * float(param_index + 3) - x * math.pi * 1.3)
        + 0.1 * (param_index + 1) * x
        + 0.05 * x * x
    )


def _set_mlp_params_smooth(module: torch.nn.Module, step: int, total_steps: int) -> None:
    with torch.no_grad():
        for idx, parameter in enumerate(module.parameters()):
            parameter.copy_(_smooth_param_value(parameter, idx, step, total_steps))


def _clone_ema_state(state: Mapping[str, object]) -> dict[str, object]:
    out = dict(state)
    shadow = state.get("shadow", {})
    assert isinstance(shadow, Mapping)
    out["shadow"] = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in shadow.items()
    }
    return out


def _closed_form_power_average(
    snapshots: list[Mapping[str, torch.Tensor]],
    gamma: float,
) -> dict[str, torch.Tensor]:
    t = len(snapshots)
    denom = float(t) ** (float(gamma) + 1.0)
    weights = [
        ((float(i) ** (float(gamma) + 1.0)) - (float(i - 1) ** (float(gamma) + 1.0))) / denom
        for i in range(1, t + 1)
    ]
    out: dict[str, torch.Tensor] = {}
    for key in snapshots[-1].keys():
        acc = torch.zeros_like(snapshots[-1][key], dtype=torch.float32)
        for weight, state in zip(weights, snapshots):
            acc.add_(state[key].to(dtype=torch.float32), alpha=float(weight))
        out[key] = acc
    return out


def _flatten_state(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([state[key].to(torch.float64).reshape(-1) for key in sorted(state.keys())])


def _assert_state_close(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    atol: float = 2e-6,
    rtol: float = 2e-6,
) -> None:
    assert set(actual.keys()) == set(expected.keys())
    for key, expected_value in expected.items():
        assert torch.allclose(actual[key].to(torch.float32), expected_value, atol=atol, rtol=rtol), key


def _single_weight_model(value: float = 0.0) -> torch.nn.Module:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(float(value))
    return model


def _set_single_weight(model: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        next(model.parameters()).fill_(float(value))


def _single_shadow_value(ema: EMA) -> float:
    shadow = ema.state_dict()["shadow"]
    assert isinstance(shadow, Mapping)
    return float(next(iter(shadow.values())).item())


def test_constant_ema_update_every_uses_elapsed_decay() -> None:
    model = _single_weight_model(0.0)
    ema = EMA(model, decay=0.9, update_every=2, offload_to_cpu=False)

    _set_single_weight(model, 1.0)
    ema.update(model)
    assert _single_shadow_value(ema) == 0.0

    _set_single_weight(model, 2.0)
    ema.update(model)
    expected = 0.0 * (0.9 ** 2) + 2.0 * (1.0 - 0.9 ** 2)
    assert math.isclose(_single_shadow_value(ema), expected, rel_tol=1e-6, abs_tol=1e-6)

    _set_single_weight(model, 3.0)
    ema.update(model)
    assert math.isclose(_single_shadow_value(ema), expected, rel_tol=1e-6, abs_tol=1e-6)

    _set_single_weight(model, 4.0)
    ema.update(model)
    expected = expected * (0.9 ** 2) + 4.0 * (1.0 - 0.9 ** 2)
    assert math.isclose(_single_shadow_value(ema), expected, rel_tol=1e-6, abs_tol=1e-6)


def test_constant_ema_explicit_decay_is_compensated_when_updates_are_thinned() -> None:
    model = _single_weight_model(0.0)
    ema = EMA(model, decay=0.999, update_every=2, offload_to_cpu=False)

    _set_single_weight(model, 1.0)
    ema.update(model, decay=0.5)
    assert _single_shadow_value(ema) == 0.0

    _set_single_weight(model, 2.0)
    ema.update(model, decay=0.5)
    expected = 0.0 * (0.5 ** 2) + 2.0 * (1.0 - 0.5 ** 2)
    assert math.isclose(_single_shadow_value(ema), expected, rel_tol=1e-6, abs_tol=1e-6)


def test_power_ema_matches_edm2_closed_form_on_mlp_params() -> None:
    model = _mlp()
    gamma = gamma_from_srel(0.10)
    ema = EMA(model, decay=0.999, profile="power", gamma=gamma, offload_to_cpu=False)
    snapshots: list[dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for step in range(1, 8):
            for idx, parameter in enumerate(model.parameters()):
                values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
                parameter.copy_(values.mul(0.01).add(float(step)).add(float(idx) * 0.1))
            snapshots.append(_clone_state(model))
            ema.update(model)

    expected = _closed_form_power_average(snapshots, gamma)
    _assert_state_close(ema.state_dict()["shadow"], expected)


def test_posthoc_two_basis_coefficients_match_numerical_lstsq() -> None:
    t = 128.0
    gamma1 = gamma_from_srel(0.05)
    gamma2 = gamma_from_srel(0.10)
    gamma_target = gamma_from_srel(0.075)
    analytic = torch.tensor(
        ls_coeffs_for_power_profile(t, gamma1, gamma2, gamma_target),
        dtype=torch.float64,
    )

    tau = torch.linspace(0.0, t, 20000, dtype=torch.float64)

    def kernel(gamma: float) -> torch.Tensor:
        return ((gamma + 1.0) / (t ** (gamma + 1.0))) * tau.pow(gamma)

    basis = torch.stack([kernel(gamma1), kernel(gamma2)], dim=1)
    target = kernel(gamma_target)
    numerical = torch.linalg.lstsq(basis, target).solution

    assert torch.allclose(analytic, numerical, atol=2e-3, rtol=2e-3)


def test_training_loop_tracks_dual_power_emas_for_mlp(tmp_path) -> None:
    torch.manual_seed(123)
    model = _mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    data = [
        {
            "x": torch.randn(6, 3),
            "y": torch.randn(6, 2),
        }
        for _ in range(6)
    ]
    post_step_snapshots: list[dict[str, torch.Tensor]] = []

    def step_fn(module: torch.nn.Module, batch: Mapping[str, torch.Tensor], _ctx: Mapping[str, object]):
        pred = module(batch["x"])
        return {"loss": torch.nn.functional.mse_loss(pred, batch["y"])}

    def record_post_step(_logs: Mapping[str, object], ctx: Mapping[str, object]) -> None:
        post_step_snapshots.append(_clone_state(ctx["model"]))  # type: ignore[arg-type]

    result = training_loop(
        model=model,
        dataset=data,
        step_fn=step_fn,
        optimizer=optimizer,
        train_cfg={
            "max_steps": len(data),
            "log_every": 1000,
            "eval_every": 1000,
            "ckpt_every": 1000,
            "grad_clip_norm": None,
            "device": "cpu",
            "out_dir": str(tmp_path / "run"),
            "ema": {
                "enabled": True,
                "profile": "constant",
                "decay": 0.9,
                "offload_to_cpu": False,
                "dual": {"enabled": True, "srel1": 0.05, "srel2": 0.10},
            },
        },
        hooks={"on_step_end": [record_post_step]},
    )

    assert len(post_step_snapshots) == len(data)
    assert "ema" in result
    assert "ema_dual" in result
    assert len(result["ema_dual"]) == 2

    dual = result["ema_dual"]
    gamma1 = gamma_from_srel(0.05)
    gamma2 = gamma_from_srel(0.10)
    assert dual[0]["profile"] == "power"
    assert dual[1]["profile"] == "power"
    assert math.isclose(float(dual[0]["gamma"]), gamma1, rel_tol=1e-12)
    assert math.isclose(float(dual[1]["gamma"]), gamma2, rel_tol=1e-12)

    _assert_state_close(dual[0]["shadow"], _closed_form_power_average(post_step_snapshots, gamma1))
    _assert_state_close(dual[1]["shadow"], _closed_form_power_average(post_step_snapshots, gamma2))

    reconstructed = reconstruct_dual_power_ema_state_dict(
        deepcopy(dual),
        step=len(data),
        gamma_target=gamma2,
    )
    _assert_state_close(reconstructed, dual[1]["shadow"], atol=2e-5, rtol=2e-5)


def test_posthoc_reconstructs_target_profile_from_mlp_snapshot_history() -> None:
    total_steps = 512
    snapshot_steps = set(range(32, total_steps + 1, 32))
    gamma1 = gamma_from_srel(0.05)
    gamma2 = gamma_from_srel(0.10)
    gamma_target = gamma_from_srel(0.075)

    model = _mlp()
    ema1 = EMA(model, decay=0.999, profile="power", gamma=gamma1, offload_to_cpu=False)
    ema2 = EMA(model, decay=0.999, profile="power", gamma=gamma2, offload_to_cpu=False)
    trajectory: list[dict[str, torch.Tensor]] = []
    ema_snapshots: list[Mapping[str, object]] = []

    for step in range(1, total_steps + 1):
        _set_mlp_params_smooth(model, step, total_steps)
        trajectory.append(_clone_state(model))
        ema1.update(model)
        ema2.update(model)
        if step in snapshot_steps:
            ema_snapshots.append(ema1.state_dict(clone=True))
            ema_snapshots.append(ema2.state_dict(clone=True))

    expected = _closed_form_power_average(trajectory, gamma_target)
    reconstructed = reconstruct_power_ema_state_dict(
        ema_snapshots,
        srel_target=0.075,
    )
    final_dual_only = reconstruct_dual_power_ema_state_dict(
        [deepcopy(ema_snapshots[-2]), deepcopy(ema_snapshots[-1])],
        step=total_steps,
        gamma_target=gamma_target,
    )

    expected_flat = _flatten_state(expected)
    reconstructed_error = torch.linalg.vector_norm(_flatten_state(reconstructed) - expected_flat)
    final_only_error = torch.linalg.vector_norm(_flatten_state(final_dual_only) - expected_flat)
    expected_norm = torch.linalg.vector_norm(expected_flat)

    assert reconstructed_error / expected_norm < 2e-3
    assert reconstructed_error < final_only_error * 0.35


def test_training_loop_checkpoint_history_reconstructs_mlp_target_ema(tmp_path) -> None:
    total_steps = 129
    gamma_target = gamma_from_srel(0.075)
    post_step_snapshots: list[dict[str, torch.Tensor]] = []
    ema_checkpoint_states: list[Mapping[str, object]] = []

    torch.manual_seed(123)
    model = _mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    def step_fn(module: torch.nn.Module, _batch: object, ctx: Mapping[str, object]):
        step = int(ctx["step"]) + 1
        loss = torch.zeros((), dtype=next(module.parameters()).dtype)
        for idx, parameter in enumerate(module.parameters()):
            target = _smooth_param_value(parameter, idx, step, total_steps)
            loss = loss + (parameter - target).square().sum()
        return {"loss": loss}

    def record_post_step(_logs: Mapping[str, object], ctx: Mapping[str, object]) -> None:
        post_step_snapshots.append(_clone_state(ctx["model"]))  # type: ignore[arg-type]

    def record_checkpoint(_path: str, ckpt: Mapping[str, object], _ctx: Mapping[str, object]) -> None:
        dual = ckpt.get("ema_dual")
        assert isinstance(dual, list)
        ema_checkpoint_states.extend(_clone_ema_state(state) for state in dual)

    result = training_loop(
        model=model,
        dataset=[{} for _ in range(total_steps)],
        step_fn=step_fn,
        optimizer=optimizer,
        train_cfg={
            "max_steps": total_steps,
            "log_every": 1000,
            "eval_every": 1000,
            "ckpt_every": 8,
            "keep_last_k": -1,
            "grad_clip_norm": None,
            "device": "cpu",
            "out_dir": str(tmp_path / "run"),
            "ema": {
                "enabled": True,
                "profile": "constant",
                "decay": 0.9,
                "offload_to_cpu": False,
                "dual": {"enabled": True, "srel1": 0.05, "srel2": 0.10},
            },
        },
        hooks={
            "on_step_end": [record_post_step],
            "on_checkpoint": [record_checkpoint],
        },
    )

    assert result["step"] == 128
    assert len(post_step_snapshots) == total_steps
    assert len(ema_checkpoint_states) == 32

    expected = _closed_form_power_average(post_step_snapshots, gamma_target)
    reconstructed = reconstruct_power_ema_state_dict(
        ema_checkpoint_states,
        srel_target=0.075,
    )
    final_only = reconstruct_dual_power_ema_state_dict(
        [deepcopy(result["ema_dual"][0]), deepcopy(result["ema_dual"][1])],
        step=total_steps,
        gamma_target=gamma_target,
    )

    expected_flat = _flatten_state(expected)
    reconstructed_error = torch.linalg.vector_norm(_flatten_state(reconstructed) - expected_flat)
    final_only_error = torch.linalg.vector_norm(_flatten_state(final_only) - expected_flat)
    expected_norm = torch.linalg.vector_norm(expected_flat)

    assert reconstructed_error / expected_norm < 3e-3
    assert reconstructed_error < final_only_error * 0.25
