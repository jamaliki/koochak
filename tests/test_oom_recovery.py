from __future__ import annotations

import json
import multiprocessing as mp
import socket
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import pytest
import torch

from koochak import OOM_RESTART_EXIT_CODE
from koochak import oom as oom_lib
from koochak.config import load_config, validate_unknown_keys
from koochak.core import dist as dist_lib
from koochak.loop import training_loop
from koochak.storage import checkpoint
from koochak.storage.store import open_store


def _batch(index: int) -> dict[str, torch.Tensor]:
    return {"x": torch.full((1, 1), float(index + 1)), "y": torch.zeros(1, 1), "i": torch.tensor(index)}


class _Data:
    """Indexed batches; ``fetch_oom`` raises OOM from inside ``next`` at that index."""

    def __init__(self, count: int, start: int = 0, fetch_oom: Optional[int] = None) -> None:
        self.count = count
        self.start = start
        self.fetch_oom = fetch_oom

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for index in range(self.start, self.count):
            if index == self.fetch_oom:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB (fetch)")
            yield _batch(index)


def _train_cfg(out_dir: Path, **overrides: Any) -> dict[str, Any]:
    oom = {"enabled": True, **overrides.pop("oom", {})}
    return {
        "ddp": False,
        "device": "cpu",
        "max_steps": 4,
        "ckpt_every": 100,
        "log_every": 100,
        "out_dir": str(out_dir),
        "oom_recovery": oom,
        **overrides,
    }


def _run(
    out_dir: Path,
    *,
    oom_steps: set[int] = frozenset(),
    data: Optional[_Data] = None,
    checkpoint_dict: Optional[dict[str, Any]] = None,
    resume: Optional[str] = None,
    **cfg: Any,
) -> dict[str, Any]:
    """Run a tiny loop and report what it saw.

    ``step_fn`` raises OOM on its first call for each step in ``oom_steps``.
    """

    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    seen: list[tuple[int, int]] = []
    events: dict[str, list[Any]] = {"step_end": [], "oom": [], "checkpoint": []}

    def step_fn(module: torch.nn.Module, batch: Mapping[str, torch.Tensor], ctx: Mapping[str, Any]):
        step = int(ctx["step"])
        seen.append((step, int(batch["i"])))
        if step in oom_steps and sum(1 for s, _ in seen if s == step) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return {"loss": torch.nn.functional.mse_loss(module(batch["x"]), batch["y"])}

    hooks = {
        "on_step_end": [lambda logs, ctx: events["step_end"].append(int(ctx["step"]))],
        "on_oom": [lambda record, ctx: events["oom"].append(dict(record))],
        "on_checkpoint": [lambda path, ckpt, ctx: events["checkpoint"].append(path)],
    }
    result: dict[str, Any] = {"seen": seen, "events": events, "model": model, "scheduler": scheduler}
    try:
        result["ckpt"] = training_loop(
            model=model,
            dataset=data if data is not None else _Data(64),
            step_fn=step_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_cfg=_train_cfg(out_dir, **cfg),
            checkpoint_dict=checkpoint_dict,
            hooks=hooks,
            resume=resume,
        )
    except SystemExit as exc:
        result["exit"] = exc.code
    return result


def _records(out_dir: Path) -> list[str]:
    return [key for key in open_store(str(out_dir)).list(oom_lib.RECORD_PREFIX)]


# ----------------------------------------------------------------------- policy


def test_consecutive_run_counts_the_run_ending_at_the_step() -> None:
    assert oom_lib.consecutive_run(set(), 5) == 1
    assert oom_lib.consecutive_run({3, 4}, 5) == 3
    assert oom_lib.consecutive_run({2, 4}, 5) == 2


@pytest.mark.parametrize(
    ("skipped", "distributed", "during_fetch", "action"),
    [
        (set(), False, False, oom_lib.SKIP),
        ({4}, False, False, oom_lib.SKIP),
        ({3, 4}, False, False, oom_lib.RESTART),
        ({2, 3, 4}, False, False, oom_lib.FAIL),
        (set(), False, True, oom_lib.RESTART),
        (set(), True, False, oom_lib.RESTART),
        ({2, 3, 4}, True, False, oom_lib.FAIL),
    ],
)
def test_decide(skipped: set[int], distributed: bool, during_fetch: bool, action: str) -> None:
    settings = oom_lib.OomRecoverySettings(enabled=True, max_consecutive=3)
    decision = oom_lib.decide(settings, skipped, 5, distributed=distributed, during_fetch=during_fetch)
    assert decision.action == action


def test_decide_enforces_total_cap() -> None:
    settings = oom_lib.OomRecoverySettings(enabled=True, max_consecutive=3, max_skipped_steps=2)
    assert oom_lib.decide(settings, {1}, 5, distributed=False, during_fetch=False).action == oom_lib.SKIP
    assert oom_lib.decide(settings, {1, 3}, 5, distributed=False, during_fetch=False).action == oom_lib.FAIL


def test_recorded_steps_ignores_unrelated_keys(tmp_path: Path) -> None:
    store = open_store(str(tmp_path))
    record = {"step": 7, "rank": 3}
    assert oom_lib.write_record(store, record) == "oom/step000000007-rank00003.json"
    oom_lib.write_record(store, record)  # idempotent for the same rank and step
    oom_lib.write_record(store, {"step": 7, "rank": 0})
    store.put("oom/notes.txt", b"x")
    assert oom_lib.recorded_steps(store) == frozenset({7})


def test_config_schema_accepts_section_and_rejects_bad_values(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  oom_recovery:\n    enabled: true\n    max_consecutive: 2\n")
    cfg = load_config(str(path))
    assert not validate_unknown_keys(cfg)
    settings = oom_lib.OomRecoverySettings.from_cfg(cfg.train)
    assert settings == oom_lib.OomRecoverySettings(enabled=True, max_consecutive=2)
    with pytest.raises(ValueError, match="max_consecutive"):
        oom_lib.OomRecoverySettings.from_cfg({"oom_recovery": {"max_consecutive": 0}})


# ------------------------------------------------------------------ single process


def test_disabled_policy_propagates_oom(tmp_path: Path) -> None:
    with pytest.raises(torch.OutOfMemoryError):
        _run(tmp_path, oom_steps={1}, oom={"enabled": False})
    assert _records(tmp_path) == []


def test_oom_step_is_skipped_in_place_and_data_stays_aligned(tmp_path: Path) -> None:
    baseline = _run(tmp_path / "baseline", grad_accum=2)
    result = _run(tmp_path / "oom", oom_steps={1}, grad_accum=2)

    # Step 1 failed on its first micro-batch (index 2); index 3 was drained, unseen.
    assert result["seen"] == [(0, 0), (0, 1), (1, 2), (2, 4), (2, 5), (3, 6), (3, 7)]
    assert result["events"]["step_end"] == [0, 2, 3]
    assert [e["action"] for e in result["events"]["oom"]] == [oom_lib.SKIP]
    assert result["events"]["oom"][0]["phase"] == "step_fn"
    assert _records(tmp_path / "oom") == ["oom/step000000001-rank00000.json"]

    # The skipped step made no update but still advanced the scheduler.
    assert result["scheduler"].last_epoch == baseline["scheduler"].last_epoch == 4
    assert not torch.equal(result["ckpt"]["model"]["weight"], baseline["ckpt"]["model"]["weight"])
    assert result["ckpt"]["oom_skipped_steps"] == [1]
    assert "oom_skipped_steps" not in baseline["ckpt"]


def test_backward_oom_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    original = torch.Tensor.backward

    def backward(self: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise torch.OutOfMemoryError("CUDA out of memory (backward)")
        original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", backward)
    result = _run(tmp_path)
    assert result["events"]["step_end"] == [0, 2, 3]
    assert result["events"]["oom"][0]["phase"] == "backward"


def test_consecutive_ooms_checkpoint_and_exit_then_fail_after_restart(tmp_path: Path) -> None:
    first = _run(tmp_path, oom_steps={1, 2}, oom={"max_consecutive": 2}, max_steps=6)

    assert first["exit"] == OOM_RESTART_EXIT_CODE
    assert [e["action"] for e in first["events"]["oom"]] == [oom_lib.SKIP, oom_lib.RESTART]
    assert first["events"]["checkpoint"] and first["events"]["checkpoint"][-1].endswith("step000000003.pt")
    selection = checkpoint.resolve_auto_resume(str(tmp_path))
    assert selection is not None
    path, saved = selection
    assert path.endswith("step000000003.pt")
    assert saved["next_step"] == 3
    assert saved["oom_skipped_steps"] == [1, 2]
    assert saved["scheduler"]["last_epoch"] == 3

    # A third OOM in a row, right after the restart, fails loudly.
    with pytest.raises(torch.OutOfMemoryError):
        _run(
            tmp_path,
            oom_steps={3},
            oom={"max_consecutive": 2},
            max_steps=6,
            data=_Data(64, start=saved["next_step"]),
            resume="auto",
        )


def test_restart_resumes_cleanly_when_the_oom_does_not_recur(tmp_path: Path) -> None:
    first = _run(tmp_path, oom_steps={1}, oom={"max_consecutive": 1}, max_steps=4)
    assert first["exit"] == OOM_RESTART_EXIT_CODE
    _path, saved = checkpoint.resolve_auto_resume(str(tmp_path))
    second = _run(tmp_path, max_steps=4, data=_Data(64, start=saved["next_step"]), resume="auto")
    assert second["seen"] == [(2, 2), (3, 3)]
    assert second["ckpt"]["oom_skipped_steps"] == [1]


def test_fetch_oom_checkpoints_and_exits_without_draining(tmp_path: Path) -> None:
    result = _run(tmp_path, data=_Data(64, fetch_oom=2))
    assert result["exit"] == OOM_RESTART_EXIT_CODE
    assert result["seen"] == [(0, 0), (1, 1)]
    assert result["events"]["oom"][0]["phase"] == "fetch"
    _path, saved = checkpoint.resolve_auto_resume(str(tmp_path))
    assert saved["next_step"] == 3
    assert saved["oom_skipped_steps"] == [2]


def test_recorded_steps_are_replayed_as_skips_without_running(tmp_path: Path) -> None:
    oom_lib.write_record(open_store(str(tmp_path)), {"step": 2, "rank": 4})
    result = _run(tmp_path, grad_accum=2)
    assert [step for step, _ in result["seen"]] == [0, 0, 1, 1, 3, 3]
    assert [index for _, index in result["seen"]][-2:] == [6, 7]
    assert result["events"]["oom"] == []
    assert result["scheduler"].last_epoch == 4


def test_recorded_skip_counts_toward_consecutive_run(tmp_path: Path) -> None:
    oom_lib.write_record(open_store(str(tmp_path)), {"step": 1, "rank": 0})
    with pytest.raises(torch.OutOfMemoryError):
        _run(tmp_path, oom_steps={2}, oom={"max_consecutive": 1})


# --------------------------------------------------------------------- distributed


def test_distributed_oom_records_and_exits_without_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pretend to be rank 1 of 2 without a process group: the loop must not try
    # to coordinate, because real peers would be blocked in collectives.
    monkeypatch.setattr(dist_lib, "rank", lambda: 1)
    monkeypatch.setattr(dist_lib, "world_size", lambda: 2)
    result = _run(tmp_path, oom_steps={1})

    assert result["exit"] == OOM_RESTART_EXIT_CODE
    assert result["seen"] == [(0, 0), (1, 1)]
    assert _records(tmp_path) == ["oom/step000000001-rank00001.json"]
    assert checkpoint.resolve_auto_resume(str(tmp_path)) is None
    record = json.loads(open_store(str(tmp_path)).get(_records(tmp_path)[0]))
    assert record["action"] == oom_lib.RESTART and record["world_size"] == 2


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ddp_replay_worker(rank: int, init_method: str, out_dir: str, queue: Any) -> None:
    torch.distributed.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        torch.manual_seed(0)
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        seen: list[int] = []

        def step_fn(module: torch.nn.Module, batch: Mapping[str, torch.Tensor], ctx: Mapping[str, Any]):
            seen.append(int(ctx["step"]))
            # Ranks see different data, so their gradients differ until all-reduced.
            x = batch["x"] * (rank + 1)
            return {"loss": torch.nn.functional.mse_loss(module(x), batch["y"])}

        result = training_loop(
            model=model,
            dataset=_Data(64),
            step_fn=step_fn,
            optimizer=optimizer,
            train_cfg={**_train_cfg(Path(out_dir)), "ddp": True, "warn_unsharded": False},
        )
        queue.put((rank, seen, result["model"]["weight"].tolist(), result.get("oom_skipped_steps")))
    finally:
        torch.distributed.destroy_process_group()


def test_ddp_ranks_agree_on_recorded_skips(tmp_path: Path) -> None:
    # Only the failed rank wrote the record; rank 0 reads and broadcasts it.
    oom_lib.write_record(open_store(str(tmp_path / "out")), {"step": 1, "rank": 1})
    context = mp.get_context("spawn")
    queue = context.Queue()
    # A TCP rendezvous: a file:// store on a shared filesystem can take minutes.
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    procs = [
        context.Process(
            target=_ddp_replay_worker,
            args=(rank, init_method, str(tmp_path / "out"), queue),
        )
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    try:
        for proc in procs:
            proc.join(timeout=120)
            assert proc.exitcode == 0
    finally:
        for proc in procs:
            proc.kill()
    results = sorted(queue.get(timeout=5) for _ in procs)
    for _rank, seen, _weight, skipped in results:
        assert seen == [0, 2, 3]
        assert skipped == [1]
    assert results[0][2] == results[1][2]


# ---------------------------------------------------------------------------- CUDA


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_oom_is_skipped_and_memory_released(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = torch.nn.Linear(1024, 1024).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    _free, total = torch.cuda.mem_get_info(device)
    allocated: list[int] = []

    def step_fn(module: torch.nn.Module, batch: Mapping[str, torch.Tensor], ctx: Mapping[str, Any]):
        activations = module(torch.randn(4096, 1024, device=device))
        if int(ctx["step"]) == 1:
            torch.empty(int(total), dtype=torch.uint8, device=device)  # more than the GPU has
        return {"loss": activations.square().mean()}

    def after_step(_logs: Mapping[str, Any], _ctx: Mapping[str, Any]) -> None:
        allocated.append(torch.cuda.memory_allocated(device))

    training_loop(
        model=model,
        dataset=[{"x": torch.zeros(1)} for _ in range(8)],
        step_fn=step_fn,
        optimizer=optimizer,
        train_cfg=_train_cfg(tmp_path, device="cuda"),
        hooks={"on_step_end": [after_step]},
    )
    assert len(allocated) == 3  # steps 0, 2, 3
    assert allocated[1] <= allocated[0]
    record = json.loads(open_store(str(tmp_path)).get(_records(tmp_path)[0]))
    assert record["memory"]["total_bytes"] == total
