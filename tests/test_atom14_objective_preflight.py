from __future__ import annotations

import math
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from scripts.atom14_objective_preflight import (
    build_report,
    inverse_c_out_weight,
)


def _config(tmp_path: Path, *, repaired: bool = True) -> Path:
    loss = {"aatype_sigma_max": 0.5}
    if repaired:
        loss.update({
            "smooth_lddt_sigma_max": 3.0,
            "smooth_lddt_resolved_atom_only": True,
            "smooth_lddt_c_out_compensation": True,
        })
    file = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({
        "model": {"sigma_data": 16.0},
        "diffusion": {"p_mean": -1.2, "p_std": 1.5},
        "loss": loss,
    }), file)
    return file


def test_objective_preflight_records_sigma_and_exact_inverse_c_out_stats(tmp_path: Path) -> None:
    report = build_report(
        _config(tmp_path), cell_id="cell-o1r0t0", objective_repair=True,
        seed=7, sample_count=256,
    )
    assert report["schema"] == "hierarchical-kaveh.atom14-objective-preflight.v1"
    assert report["finite"] is True
    assert set(report["sampled_sigma"]) == {"min", "median", "p90", "p99", "max"}
    lddt = report["lddt"]
    assert lddt["weight_formula"] == "1 / c_out"
    assert set(lddt["inverse_c_out_weight"]) == {"min", "median", "p90", "p99", "max"}
    assert set(lddt["effective_lddt_weight"]) == {"min", "median", "p90", "p99", "max"}
    assert 0.0 < lddt["active_fraction"] < 1.0
    assert all(math.isfinite(value) for value in lddt["inverse_c_out_weight"].values())


def test_preflight_uses_one_over_c_out_not_one_over_c_out_squared() -> None:
    weights = inverse_c_out_weight(torch.tensor([1.0]), 16.0)
    expected = math.sqrt(1.0 + 16.0**2) / (1.0 * 16.0)
    assert float(weights[0]) == pytest.approx(expected)


def test_nonobjective_preflight_does_not_report_repair_weights(tmp_path: Path) -> None:
    report = build_report(
        _config(tmp_path, repaired=False), cell_id="cell-o0r0t0", objective_repair=False,
        seed=7, sample_count=32,
    )
    assert report["lddt"]["enabled"] is False
    assert report["lddt"]["active_fraction"] is None
    assert report["lddt"]["inverse_c_out_weight"] is None


def test_objective_preflight_fails_when_repair_config_is_not_exact(tmp_path: Path) -> None:
    file = _config(tmp_path)
    document = OmegaConf.load(file)
    document.loss.smooth_lddt_c_out_compensation = False
    OmegaConf.save(document, file)
    with pytest.raises(ValueError, match="objective repair config mismatch"):
        build_report(file, cell_id="cell-o1r0t0", objective_repair=True, sample_count=32)
