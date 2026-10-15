"""Eight-GPU NCCL checks for real queries and coordinated emergency saves."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from koochak.health import background
from koochak.loop import training_loop
from koochak.storage import checkpoint


def _cuda_worker(rank, init_file, out_dir, queue, mode):
    os.environ.pop("KOOCHAK_GPU_HEALTH_DISABLE", None)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=8,
        timeout=timedelta(seconds=120), device_id=torch.device("cuda", rank),
    )
    original_init = background.BackgroundGpuHealth.__init__
    original_query = background.query_node_samples
    instances = []
    audits = []
    completed = []

    def monitor_init(self, **kwargs):
        original_init(self, **kwargs)
        instances.append(self)

    def query(watchdogs, step, **kwargs):
        assert rank == 0
        samples = original_query(watchdogs, step, **kwargs)
        with (Path(out_dir) / f"query-owner-{rank}.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, "count": len(samples)}) + "\n")
        if step >= 1:
            if mode == "query":
                raise OSError("Injected CUDA monitor query failure")
            if mode == "stale":
                instances[0]._stop.wait(10)
            if mode == "hardware":
                samples[1].gpu_util_pct = 100
                samples[1].hw_power_brake_slowdown = True
        return samples

    background.BackgroundGpuHealth.__init__ = monitor_init
    background.query_node_samples = query
    background.INTERVAL_SECONDS = .05
    background.MAX_SAMPLE_AGE_SECONDS = 2.0
    Path(out_dir).mkdir(exist_ok=True)
    try:
        torch.manual_seed(100 + rank)
        device = torch.device("cuda", rank)
        model = torch.nn.Linear(4, 4, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)

        def step_fn(module, batch, ctx):
            time.sleep(.04)
            return {"loss": module(batch).square().mean()}

        def record_step(logs, ctx):
            completed.append(ctx["step"])

        def sampling_audit(*args):
            audits.append(True)

        exit_code = None
        try:
            training_loop(
                model=model, optimizer=optimizer,
                dataset=[torch.ones(2, 4, device=device)] * 120,
                step_fn=step_fn, train_cfg={
                    "device": str(device), "ddp": True, "warn_unsharded": False,
                    "max_steps": 120, "ckpt_every": 1000, "log_every": 1000,
                    "out_dir": out_dir,
                    "ema": {"enabled": True, "decay": .999, "offload_to_cpu": False},
                }, hooks={"on_step_end": [record_step],
                          "on_checkpoint_all_ranks": [sampling_audit]},
            )
        except SystemExit as exc:
            exit_code = exc.code
        queue.put({
            "rank": rank, "exit": exit_code, "completed": completed,
            "audits": len(audits),
            "thread_alive": bool(instances[0]._thread and instances[0]._thread.is_alive()),
            "uuid": background.physical_gpu_uuid(torch.cuda.get_device_properties(device).uuid),
        })
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 8,
                    reason="Requires one node with eight CUDA GPUs")
@pytest.mark.parametrize("mode", ["healthy", "hardware", "query", "stale"])
def test_eight_gpu_background_monitor_and_checkpoint(tmp_path, mode):
    context = mp.get_context("spawn")
    queue = context.Queue()
    out_dir = tmp_path / "run"
    processes = [context.Process(
        target=_cuda_worker,
        args=(rank, str(tmp_path / "nccl-init"), str(out_dir), queue, mode),
    ) for rank in range(8)]
    for process in processes:
        process.start()
    try:
        deadline = time.monotonic() + 180
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        assert all(process.exitcode == 0 for process in processes)
        results = sorted((queue.get(timeout=5) for _ in processes), key=lambda item: item["rank"])
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
    assert all(item["completed"] == results[0]["completed"] for item in results)
    assert all(not item["thread_alive"] for item in results)
    assert [item["exit"] for item in results] == [None if mode == "healthy" else 42] * 8
    assert [item["audits"] for item in results] == [1 if mode == "healthy" else 0] * 8
    cursor = results[0]["completed"][-1] + 1
    saved = checkpoint.load(str(out_dir / f"step{cursor:09d}.pt"))
    assert saved["step"] == saved["next_step"] == cursor
    assert len(saved["rng"]["per_rank"]) == 8
    assert saved["optimizer"]["state"] and saved["ema"]
    owners = list(out_dir.glob("query-owner-*.jsonl"))
    assert [file.name for file in owners] == ["query-owner-0.jsonl"]
    assert all(json.loads(line)["count"] == 8 for line in owners[0].read_text().splitlines())
    for result in results:
        file = out_dir / "gpu_health" / f"gpu_health_rank{result['rank']}.jsonl"
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        assert rows and {row["gpu_uuid"] for row in rows} == {result["uuid"]}
    assert len({item["uuid"] for item in results}) == 8
    if mode == "healthy":
        assert cursor == 120
        assert "gpu_health" not in saved["metrics"]
    else:
        assert cursor < 120
        expected = {"hardware": "hw_power_brake_slowdown", "query": "monitor_query_failed",
                    "stale": "monitor_stale"}[mode]
        assert saved["metrics"]["gpu_health"]["failures"][0]["reasons"] == [expected]
