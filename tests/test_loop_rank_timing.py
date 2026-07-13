from __future__ import annotations

from types import SimpleNamespace

import torch

from koochak.loop import _TrainLoop


def test_loop_profile_metrics_reach_rank_timing_fallback():
    model = torch.nn.Linear(1, 1, bias=False)
    loop = object.__new__(_TrainLoop)
    loop.settings = SimpleNamespace(
        rank_timing_extra_keys=(
            "profile_loop_backward_time_s",
            "profile_render_materialize_time_s",
        ),
        grad_accum=1,
        autocast_in_step_fn=False,
        prefetch_batches=0,
        profile_step_timing=True,
        profile_step_cuda_sync=False,
        scalarize_loss_every_step=False,
    )
    loop.ctx = {}
    loop.device = torch.device("cpu")
    loop.model = model
    loop.scaler = torch.amp.GradScaler("cpu", enabled=False)

    def step_fn(module, batch, _ctx):
        return {
            "loss": module(batch).square().mean(),
            "profile_render_materialize_time_s": 0.125,
        }

    loop.step_fn = step_fn
    stats = loop._run_micro_steps(
        0,
        iter([torch.ones(1, 1)]),
        collect_rank_timing=True,
    )

    assert "profile_loop_backward_time_s" not in stats.extra_timing_totals
    assert stats.profile_timing_totals["profile_loop_backward_time_s"] > 0.0
    assert stats.extra_timing_totals["profile_render_materialize_time_s"] == 0.125
