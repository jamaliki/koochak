from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import signal
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from koochak.interruption import EVACUATION_EXIT_CODE, EvacuationController
from koochak.loop import training_loop
from koochak.logging.events import make_event_hooks
from koochak.storage import checkpoint


def _train_cfg(out_dir: Path, *, max_steps: int = 3) -> dict[str, Any]:
    return {
        "ddp": False,
        "device": "cpu",
        "max_steps": max_steps,
        "ckpt_every": 100,
        "log_every": 100,
        "out_dir": str(out_dir),
        "evacuation_enabled": True,
    }


def _step_fn(module: torch.nn.Module, batch: Mapping[str, torch.Tensor], _ctx: Mapping[str, Any]):
    return {"loss": torch.nn.functional.mse_loss(module(batch["x"]), batch["y"])}


def _dataset(count: int = 5) -> list[dict[str, torch.Tensor]]:
    return [{"x": torch.ones(1, 1), "y": torch.zeros(1, 1)} for _ in range(count)]


def _ddp_worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
    result_queue: Any,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(100 + rank)
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        controller = EvacuationController(False)
        if rank == 1:
            controller.request()
        try:
            training_loop(
                model=model,
                dataset=_dataset(),
                step_fn=_step_fn,
                optimizer=optimizer,
                train_cfg={**_train_cfg(Path(out_dir), max_steps=3), "ddp": True},
                evacuation=controller,
            )
        except SystemExit as exc:
            result_queue.put((rank, "exit", exc.code))
        else:
            result_queue.put((rank, "return", None))
    finally:
        torch.distributed.destroy_process_group()


def test_controller_handler_only_sets_flag_and_rejects_other_signals() -> None:
    controller = EvacuationController(True)
    assert controller.requested is False
    controller._handle_signal(signal.SIGUSR1, None)
    assert controller.requested is True
    with pytest.raises(ValueError, match="SIGUSR1"):
        EvacuationController(True, signal_name="TERM")
    with pytest.raises(ValueError, match="reserved"):
        EvacuationController(True, exit_code=1)


def test_signal_saves_terminal_checkpoint_before_reserved_exit(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    completed_steps: list[int] = []
    events: list[str] = []

    def request_after_update(_logs: Mapping[str, Any], ctx: Mapping[str, Any]) -> None:
        completed_steps.append(int(ctx["step"]))
        if len(completed_steps) == 1:
            os.kill(os.getpid(), signal.SIGUSR1)

    hooks = {
        "on_step_end": [request_after_update],
        "on_checkpoint": [lambda *_args: events.append("artifact")],
        "on_evacuation": [lambda *_args: events.append("evacuation")],
        "on_train_end": [lambda _ctx: events.append("end")],
    }
    with pytest.raises(SystemExit) as raised:
        training_loop(
            model=model,
            dataset=_dataset(),
            step_fn=_step_fn,
            optimizer=optimizer,
            train_cfg=_train_cfg(tmp_path),
            hooks=hooks,
        )

    assert raised.value.code == EVACUATION_EXIT_CODE
    assert completed_steps == [0]
    terminal_path = tmp_path / "step000000001.pt"
    assert terminal_path.is_file()
    assert checkpoint.load(str(terminal_path))["next_step"] == 1
    assert terminal_path.with_name(terminal_path.name + ".ready.json").is_file()
    assert events == ["artifact", "evacuation", "end"]


def test_auto_resume_is_first_run_safe_and_uses_first_unexecuted_step(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = training_loop(
        model=model,
        dataset=_dataset(),
        step_fn=_step_fn,
        optimizer=optimizer,
        train_cfg=_train_cfg(tmp_path, max_steps=2),
        resume="auto",
    )
    assert result["next_step"] == 2

    resumed_model = torch.nn.Linear(1, 1, bias=False)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    steps: list[int] = []
    resumed = training_loop(
        model=resumed_model,
        dataset=_dataset(),
        step_fn=_step_fn,
        optimizer=resumed_optimizer,
        train_cfg=_train_cfg(tmp_path, max_steps=3),
        resume="auto",
        hooks={"on_step_end": [lambda _logs, ctx: steps.append(int(ctx["step"]))]},
    )
    assert steps == [2]
    assert resumed["next_step"] == 3


def test_resolve_auto_resume_returns_validated_path_and_cursor(tmp_path: Path) -> None:
    path = tmp_path / "step000000003.pt"
    checkpoint.save(
        {"step": 3, "next_step": 4, "model": {}, "optimizer": {}}, str(path)
    )

    resolved = checkpoint.resolve_auto_resume(str(tmp_path))

    assert resolved is not None
    resolved_path, resolved_checkpoint = resolved
    assert resolved_path == str(path.absolute())
    assert resolved_checkpoint["next_step"] == 4


def test_resolve_auto_resume_loads_the_validated_payload_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "step000000003.pt"
    checkpoint.save(
        {"step": 3, "next_step": 4, "model": {}, "optimizer": {}}, str(path)
    )
    calls = 0
    original_load = checkpoint.torch.load

    def counted_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_load(*args, **kwargs)

    def unexpected_path_load(_path):
        raise AssertionError("resolver must not reopen the checkpoint path")

    monkeypatch.setattr(checkpoint.torch, "load", counted_load)
    monkeypatch.setattr(checkpoint, "load", unexpected_path_load)
    resolved = checkpoint.resolve_auto_resume(str(tmp_path))
    assert resolved is not None
    assert calls == 1
    assert resolved[1]["next_step"] == 4


def test_preloaded_auto_resume_preserves_selection_event(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    training_loop(
        model=model,
        dataset=_dataset(),
        step_fn=_step_fn,
        optimizer=optimizer,
        train_cfg=_train_cfg(tmp_path, max_steps=1),
    )
    resolved = checkpoint.resolve_auto_resume(str(tmp_path))
    assert resolved is not None
    resume_path, resume_checkpoint = resolved

    events: list[tuple[str, bool]] = []
    resumed_model = torch.nn.Linear(1, 1, bias=False)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    training_loop(
        model=resumed_model,
        dataset=_dataset(),
        step_fn=_step_fn,
        optimizer=resumed_optimizer,
        train_cfg=_train_cfg(tmp_path, max_steps=2),
        checkpoint_dict=resume_checkpoint,
        resume="auto",
        auto_resume_path=resume_path,
        hooks={
            "on_resume_checkpoint": [
                lambda path, _checkpoint, ctx: events.append(
                    (path, bool(ctx["auto_resume_selected"]))
                )
            ]
        },
    )

    assert events == [(resume_path, True)]


def test_highest_valid_published_ignores_latest_and_bad_candidates(tmp_path: Path) -> None:
    for step in (2, 4):
        checkpoint.save(
            {"step": step, "next_step": step + 1, "model": {}, "optimizer": {}},
            str(tmp_path / f"step{step:09d}.pt"),
        )
    # Highest candidate is corrupted after publication; lower valid evidence wins.
    highest = tmp_path / "step000000004.pt"
    highest.write_bytes(highest.read_bytes() + b"corruption")
    (tmp_path / "latest.pt").write_bytes(b"not a checkpoint")
    (tmp_path / "step000000009").mkdir()
    assert checkpoint.highest_valid_published(str(tmp_path)) == str(
        (tmp_path / "step000000002.pt").absolute()
    )


def test_manifest_path_size_digest_and_cursor_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "step000000003.pt"
    checkpoint.save(
        {"step": 3, "next_step": 4, "model": {}, "optimizer": {}}, str(path)
    )
    manifest_path = Path(checkpoint.publication_path(str(path)))
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = hashlib.sha256(b"wrong").hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    assert checkpoint.highest_valid_published(str(tmp_path)) is None


    checkpoint.save(
        {"step": 3, "next_step": 4, "model": {}, "optimizer": {}}, str(path)
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["path"] = str(tmp_path / "elsewhere.pt")
    manifest_path.write_text(json.dumps(manifest))
    assert checkpoint.highest_valid_published(str(tmp_path)) is None


def test_malformed_payload_and_symlinked_manifest_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "step000000003.pt"
    payload = b"not a torch checkpoint"
    path.write_bytes(payload)
    manifest_path = Path(checkpoint.publication_path(str(path)))
    manifest = {
        "v": 1,
        "artifact_id": path.name.replace("step", "checkpoint/step"),
        "path": str(path.absolute()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "manifest_path": str(manifest_path.absolute()),
    }
    manifest_path.write_text(json.dumps(manifest))
    assert checkpoint.highest_valid_published(str(tmp_path)) is None

    checkpoint.save(
        {"step": 3, "next_step": 4, "model": {}, "optimizer": {}}, str(path)
    )
    real_manifest = tmp_path / "real.ready.json"
    real_manifest.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(real_manifest.name)
    assert checkpoint.highest_valid_published(str(tmp_path)) is None


def test_validator_loads_the_hashed_payload_not_checkpoint_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "step000000003.pt"
    checkpoint.save(
        {"step": 3, "next_step": 4, "model": {}, "optimizer": {}}, str(path)
    )
    monkeypatch.setattr(
        checkpoint,
        "load",
        lambda _path: pytest.fail("validator must load its stable byte snapshot"),
    )
    assert checkpoint.highest_valid_published(str(tmp_path)) == str(path.absolute())


def test_auto_resume_retries_typed_artifact_publication(tmp_path: Path) -> None:
    failed = True

    def flaky_publish(kind: str, _data: dict[str, object]) -> None:
        nonlocal failed
        if kind == "workload.artifact" and failed:
            failed = False
            raise OSError("transient Scruffy publication loss")

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.warns(RuntimeWarning, match="publisher failed"):
        training_loop(
            model=model,
            dataset=_dataset(),
            step_fn=_step_fn,
            optimizer=optimizer,
            train_cfg=_train_cfg(tmp_path, max_steps=1),
            resume="none",
            hooks=make_event_hooks(flaky_publish),
        )

    healed: list[tuple[str, dict[str, object]]] = []
    resumed_model = torch.nn.Linear(1, 1, bias=False)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    training_loop(
        model=resumed_model,
        dataset=_dataset(),
        step_fn=_step_fn,
        optimizer=resumed_optimizer,
        train_cfg=_train_cfg(tmp_path, max_steps=2),
        resume="auto",
        hooks=make_event_hooks(lambda kind, data: healed.append((kind, data))),
    )
    artifacts = [data for kind, data in healed if kind == "workload.artifact"]
    assert artifacts[0]["location"].endswith("step000000001.pt")


def test_ddp_ranks_reconcile_evacuation_and_publish_per_rank_rng(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_ddp_worker,
            args=(rank, 2, str(tmp_path / "dist-init"), str(tmp_path / "run"), result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    results = sorted(result_queue.get(timeout=5) for _ in processes)
    assert results == [(0, "exit", EVACUATION_EXIT_CODE), (1, "exit", EVACUATION_EXIT_CODE)]
    terminal = tmp_path / "run" / "step000000001.pt"
    saved = checkpoint.load(str(terminal))
    assert saved["next_step"] == 1
    assert len(saved["rng"]["per_rank"]) == 2
