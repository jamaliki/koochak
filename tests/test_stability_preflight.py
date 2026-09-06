from __future__ import annotations

import math

import pytest

from scripts.stability_preflight import GateThresholds, evaluate_rows


def _rows(*, workers: int = 2, steps: int = 8) -> list[dict[str, float]]:
    rows = []
    for step in range(steps):
        worker = step % workers
        rows.append(
            {
                "step": step,
                "step_time_s": 0.25 + 0.01 * (step % 2),
                "loss": 1.0,
                "data_worker_id": worker,
                "data_owned_shard_count": 4,
                "data_cached_shard_count": 4,
                "data_cache_hit_count": step,
                "data_cache_miss_count": 4,
            }
        )
    return rows


def test_preflight_evaluates_finite_warmed_latency_and_resident_cache() -> None:
    report = evaluate_rows(
        _rows(),
        expected_workers=2,
        thresholds=GateThresholds(warmup_steps=2, minimum_timed_rows=4),
    )
    assert report["passed"]
    assert report["finite_loss"]
    assert report["latency"]["step_time_s"]["p90"] == pytest.approx(0.26)
    assert report["cache"]["observed_workers"] == 2


def test_preflight_rejects_nonfinite_loss_and_post_warmup_cache_misses() -> None:
    rows = _rows()
    rows[4]["loss"] = math.nan
    rows[4]["data_cache_miss_count"] = 5
    report = evaluate_rows(
        rows,
        expected_workers=2,
        thresholds=GateThresholds(warmup_steps=2, minimum_timed_rows=4),
    )
    assert not report["passed"]
    assert any("non-finite" in failure for failure in report["failures"])
    assert any("cache miss" in failure for failure in report["failures"])
