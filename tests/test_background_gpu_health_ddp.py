from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from koochak.health import background
from koochak.health.gpu import sample_from_nvidia_smi_row, PRIMARY_QUERY_FIELDS
from koochak.loop import training_loop
from koochak.storage import checkpoint


def _worker(rank, init_file, out_dir, result_queue, fault):
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
    )
    original_init = background.BackgroundGpuHealth.__init__
    instances = []

    def fake_init(self, **kwargs):
        original_init(self, **kwargs)
        self.enabled = True
        self.device = torch.device("cuda", rank)
        self.cuda_device = rank
        instances.append(self)

    def properties(device):
        assert threading.current_thread() is threading.main_thread()
        return SimpleNamespace(uuid=f"GPU-00000000-0000-0000-0000-{rank:012d}")

    def query(watchdogs, step, **kwargs):
        assert rank == 0
        with (Path(out_dir) / f"queries-rank{rank}.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, "ranks": [item.rank for item in watchdogs]}) + "\n")
        if fault == "stale":
            instances[0]._stop.wait(2)
        elif fault == "query" and step >= 1:
            raise OSError("test query failure")
        rows = []
        for watchdog in watchdogs:
            values = {
                "index": str(watchdog.rank), "uuid": watchdog.gpu_query_id, "name": "H100",
                "temperature.gpu": "60", "temperature.memory": "65", "clocks.sm": "1980",
                "clocks.max.sm": "1980", "utilization.gpu": "100",
            }
            if fault == "hardware" and watchdog.rank == 1 and step >= 1:
                values["clocks.sm"] = "100"
            rows.append(sample_from_nvidia_smi_row(
                step=step, rank=watchdog.rank, world_size=2, hostname=watchdog.hostname,
                slurm_node=watchdog.slurm_node, cuda_device=watchdog.cuda_device,
                gpu_query_id=watchdog.gpu_query_id, fields=PRIMARY_QUERY_FIELDS,
                row=[values.get(field, "N/A") for field in PRIMARY_QUERY_FIELDS],
            ))
        return rows

    background.BackgroundGpuHealth.__init__ = fake_init
    torch.cuda.get_device_properties = properties
    background.query_node_samples = query
    background.INTERVAL_SECONDS = .02
    background.MAX_SAMPLE_AGE_SECONDS = .06
    Path(out_dir).mkdir(exist_ok=True)
    completed = []
    normal_collectives = []
    original_reduce = torch.distributed.all_reduce
    original_gather = torch.distributed.all_gather_object

    def reduce(tensor, *args, **kwargs):
        normal_collectives.append(("reduce", tensor.numel()))
        return original_reduce(tensor, *args, **kwargs)

    def gather(*args, **kwargs):
        normal_collectives.append(("gather", None))
        return original_gather(*args, **kwargs)

    torch.distributed.all_reduce = reduce
    torch.distributed.all_gather_object = gather
    try:
        torch.manual_seed(100 + rank)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)

        def step_fn(module, batch, ctx):
            time.sleep(.02)
            return {"loss": module(batch).square().sum()}

        def on_step(logs, ctx):
            completed.append(ctx["step"])

        try:
            training_loop(
                model=model, optimizer=optimizer, dataset=[torch.ones(1, 1)] * 30,
                step_fn=step_fn, train_cfg={
                    "device": "cpu", "ddp": True, "warn_unsharded": False,
                    "max_steps": 30, "ckpt_every": 100, "log_every": 100,
                    "out_dir": out_dir,
                }, hooks={"on_step_end": [on_step]},
            )
        except SystemExit as exc:
            result_queue.put({
                "rank": rank, "exit": exc.code, "completed": completed,
                "collectives": normal_collectives,
                "thread_alive": bool(instances[0]._thread and instances[0]._thread.is_alive()),
            })
        else:
            result_queue.put({"rank": rank, "exit": None})
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("fault", ["hardware", "query", "stale"])
def test_background_fault_agrees_completed_update_and_saves_all_rank_rng(tmp_path, fault):
    context = mp.get_context("spawn")
    queue = context.Queue()
    out_dir = tmp_path / "run"
    processes = [context.Process(
        target=_worker, args=(rank, str(tmp_path / "dist-init"), str(out_dir), queue, fault),
    ) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=30)
        assert all(process.exitcode == 0 for process in processes)
        results = sorted((queue.get(timeout=3) for _ in processes), key=lambda item: item["rank"])
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
    assert [item["exit"] for item in results] == [42, 42]
    assert results[0]["completed"] == results[1]["completed"]
    assert all(not item["thread_alive"] for item in results)
    for item in results:
        assert item["collectives"] == [
            ("gather", None),  # One topology election at startup.
            *[("reduce", 2)] * len(item["completed"]),
            ("gather", None),  # Fault details only after global stop agreement.
            ("gather", None),  # Full per-rank RNG before publication.
        ]
    cursor = results[0]["completed"][-1] + 1
    saved = checkpoint.load(str(out_dir / f"step{cursor:09d}.pt"))
    assert saved["step"] == saved["next_step"] == cursor
    assert len(saved["rng"]["per_rank"]) == 2
    failure = saved["metrics"]["gpu_health"]["failures"][0]
    expected = {"hardware": "sm_clock_low", "query": "monitor_query_failed", "stale": "monitor_stale"}
    assert failure["reasons"] == [expected[fault]]
    assert not (out_dir / "queries-rank1.jsonl").exists()
    query_rows = [json.loads(line) for line in (out_dir / "queries-rank0.jsonl").read_text().splitlines()]
    assert all(row["ranks"] == [0, 1] for row in query_rows)
