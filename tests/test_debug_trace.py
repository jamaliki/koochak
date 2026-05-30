from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.pop("koochak", None)

from koochak.loop import training_loop


def test_training_loop_writes_debug_snapshot_and_loss_tensor_trace(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    out_dir = tmp_path / "run"
    snapshot_dir = tmp_path / "snapshots"
    trace_dir = tmp_path / "trace"

    def step_fn(model, batch, _ctx):
        prediction = model(batch["x"].float())
        mse = (prediction - batch["target"].float()).square().mean()
        l2 = 0.1 * model.weight.square().sum()
        loss = mse + l2
        return {
            "loss": loss,
            "_debug_loss_terms": {"total": loss, "mse": mse, "l2": l2},
            "_debug_tensors": {"prediction": prediction},
        }

    training_loop(
        model=model,
        dataset=[{"x": torch.ones(3, 2), "target": torch.zeros(3, 1)}],
        step_fn=step_fn,
        optimizer=optimizer,
        train_cfg={
            "ddp": False,
            "log_every": 1,
            "max_steps": 1,
            "device": "cpu",
            "out_dir": str(out_dir),
            "debug_snapshot_steps": "0",
            "debug_snapshot_dir": str(snapshot_dir),
            "debug_trace_steps": "0",
            "debug_trace_dir": str(trace_dir),
            "debug_trace_loss_terms": "total,mse,l2",
            "debug_trace_top_k": 4,
        },
    )

    snapshot_path = snapshot_dir / "debug_step000000000_micro00.pt"
    assert snapshot_path.exists()
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    assert snapshot["step"] == 0
    assert "model" in snapshot
    assert "optimizer" in snapshot
    assert torch.equal(snapshot["batch"]["x"], torch.ones(3, 2))

    trace_path = trace_dir / "debug_trace_rank0.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    events = {row["event"] for row in rows}
    assert "debug_tensor_values" in events
    assert "loss_grad_trace" in events
    traced_losses = {row.get("loss_name") for row in rows if row["event"] == "loss_grad_trace"}
    assert {"total", "mse", "l2"}.issubset(traced_losses)
    total_trace = next(
        row
        for row in rows
        if row["event"] == "loss_grad_trace" and row["loss_name"] == "total"
    )
    assert total_trace["tensor_grads"]["prediction"]["present"] is True
    assert total_trace["tensor_grads"]["prediction"]["norm"] > 0.0
    assert total_trace["top_parameter_grad_norms"]
