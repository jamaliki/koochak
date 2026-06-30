from __future__ import annotations

from types import SimpleNamespace

import torch

from koochak import loop as loop_mod


def test_rank_timing_prefers_loop_profile_totals(monkeypatch, capsys) -> None:
    """Requested profile_loop_* keys should report loop timings, not zero fillers."""

    captured_payloads = []

    def fake_all_gather_object(gathered, payload):
        captured_payloads.append(payload)
        gathered[0] = payload

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    train_loop = object.__new__(loop_mod._TrainLoop)
    train_loop.rank = 0
    train_loop.world_size = 1
    train_loop.is_rank0 = True
    train_loop.settings = SimpleNamespace(
        rank_timing_extra_keys=("profile_loop_step_fn_time_s",)
    )

    stats = loop_mod._MicroStepStats(
        batch_wait_s=0.1,
        extra_timing_totals={"profile_loop_step_fn_time_s": 0.0},
        profile_timing_totals={"profile_loop_step_fn_time_s": 1.25},
    )

    train_loop._emit_rank_timing(step=7, stats=stats, step_time_s=2.0)

    assert captured_payloads[0]["profile_loop_step_fn_time_s"] == 1.25
    assert "profile_loop_step_fn_time_s=1.250" in capsys.readouterr().out
