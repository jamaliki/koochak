"""Periodic DDP checkpoints retain each rank's random state."""

import multiprocessing as mp
from pathlib import Path

import pytest
import torch

from koochak.loop import training_loop
from koochak.storage import checkpoint


def _worker(rank, init_file, folder, asynchronous, audit=False):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        torch.manual_seed(100 + rank)
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)

        def step(module, batch, context):
            return {"loss": (module(batch) - torch.rand_like(batch)).square().mean()}

        def record(log, context):
            if context["step"] == 1:
                torch.save(torch.get_rng_state(), Path(folder) / f"expected-{rank}.pt")

        events = []

        def checkpoint_hook(file, saved, context):
            assert Path(file).with_suffix(".pt.ready.json").exists()
            assert saved["ema"]["shadow"]
            shadow = saved["ema"]["shadow"]["weight"]
            gathered = [None, None]
            torch.distributed.all_gather_object(gathered, shadow)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
            events.append(context["step"])

        hooks = {"on_step_end": [record]}
        if audit:
            hooks["on_checkpoint_all_ranks"] = [checkpoint_hook]

        training_loop(model=model, dataset=[torch.ones(1, 1)] * 4, step_fn=step,
                      optimizer=optimizer, hooks=hooks,
                      train_cfg=dict(ddp=True, device="cpu", max_steps=3, ckpt_every=1,
                                     checkpoint_async=asynchronous, keep_last_k=10,
                                     ema=dict(enabled=True, decay=.999, offload_to_cpu=False),
                                     log_every=100, out_dir=folder))
        if audit:
            torch.save(events, Path(folder) / f"audit-{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("asynchronous,audit", [(False, False), (True, False), (False, True)])
def test_periodic_checkpoint_gathers_every_rank(tmp_path, asynchronous, audit):
    context = mp.get_context("spawn")
    folder = tmp_path / "run"
    folder.mkdir()
    processes = [context.Process(target=_worker,
                                 args=(rank, str(tmp_path / "init"), str(folder), asynchronous, audit))
                 for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    saved = checkpoint.load(str(folder / "step000000001.pt"))
    assert saved["next_step"] == 2
    states = saved["rng"]["per_rank"]
    assert len(states) == 2
    for rank, state in enumerate(states):
        expected = torch.load(folder / f"expected-{rank}.pt", weights_only=True)
        torch.testing.assert_close(state["torch_cpu"], expected, rtol=0, atol=0)
    assert not torch.equal(states[0]["torch_cpu"], states[1]["torch_cpu"])
    if audit:
        for rank in range(2):
            assert torch.load(folder / f"audit-{rank}.pt", weights_only=True) == [1, 2, 3]


def test_all_rank_hook_rejects_async_publication(tmp_path):
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="synchronous"):
        training_loop(model=model, dataset=[], step_fn=lambda *args: {},
                      optimizer=torch.optim.SGD(model.parameters(), lr=.1),
                      train_cfg=dict(ddp=False, out_dir=str(tmp_path), checkpoint_async=True),
                      hooks={"on_checkpoint_all_ranks": [lambda *args: None]})


def test_all_rank_hook_failure_stops_training(tmp_path):
    model = torch.nn.Linear(1, 1)

    def reject(*args):
        raise ValueError("Invalid sample")

    with pytest.raises(ValueError, match="Invalid sample"):
        training_loop(model=model, dataset=[torch.ones(1, 1)],
                      step_fn=lambda module, batch, context: {"loss": module(batch).square().mean()},
                      optimizer=torch.optim.SGD(model.parameters(), lr=.1),
                      train_cfg=dict(ddp=False, out_dir=str(tmp_path), max_steps=1),
                      hooks={"on_checkpoint_all_ranks": [reject]})


def test_rng_capture_failure_prevents_publication(tmp_path, monkeypatch):
    import koochak.loop as loop

    def reject():
        raise RuntimeError("RNG capture failed")

    monkeypatch.setattr(loop, "get_rng_state", reject)
    model = torch.nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="RNG capture failed"):
        training_loop(model=model, dataset=[torch.ones(1, 1)],
                      step_fn=lambda module, batch, context: {"loss": module(batch).square().mean()},
                      optimizer=torch.optim.SGD(model.parameters(), lr=.1),
                      train_cfg=dict(ddp=False, out_dir=str(tmp_path), max_steps=1))
    assert not list(tmp_path.glob("*.pt*"))
