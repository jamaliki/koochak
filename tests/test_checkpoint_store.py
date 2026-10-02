"""Checkpoints published through a ``Store``: write-once mounts, URIs, background saves."""

from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from koochak.loop import training_loop
from koochak.storage import checkpoint
from koochak.storage import store as store_lib
from koochak.storage.store import LocalStore


@pytest.fixture(autouse=True)
def no_stores_file(monkeypatch, tmp_path):
    monkeypatch.delenv("KOOCHAK_STORES", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


class RecordingStore:
    """Delegate to a LocalStore and record every mutation in order."""

    def __init__(self, inner: LocalStore) -> None:
        self.inner = inner
        self.ops: list[tuple[str, str]] = []

    def get(self, key, offset=0, length=None):
        return self.inner.get(key, offset, length)

    def open(self, key):
        return self.inner.open(key)

    def put(self, key, data):
        self.ops.append(("put", key))
        return self.inner.put(key, data)

    def stat(self, key):
        return self.inner.stat(key)

    def list(self, prefix=""):
        return self.inner.list(prefix)

    def delete(self, key):
        self.ops.append(("delete", key))
        self.inner.delete(key)

    def local_path(self, key):
        return self.inner.local_path(key)


def _payload(step: int, *, next_step: int | None = None) -> bytes:
    return checkpoint.serialize(
        {"step": step, "next_step": step + 1 if next_step is None else next_step, "model": {}}
    )


def _train_cfg(out_dir: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "ddp": False,
        "device": "cpu",
        "max_steps": 3,
        "ckpt_every": 1,
        "log_every": 100,
        "keep_last_k": 10,
        "out_dir": str(out_dir),
        **overrides,
    }


def _step_fn(module: torch.nn.Module, batch: Mapping[str, torch.Tensor], _ctx: Mapping[str, Any]):
    return {"loss": torch.nn.functional.mse_loss(module(batch["x"]), batch["y"])}


def _dataset(count: int = 8) -> list[dict[str, torch.Tensor]]:
    return [{"x": torch.ones(1, 1), "y": torch.zeros(1, 1)} for _ in range(count)]


def _train(cfg: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return training_loop(
        model=model, dataset=_dataset(), step_fn=_step_fn, optimizer=optimizer, train_cfg=cfg, **kwargs
    )


def test_publish_commits_the_manifest_last_without_latest_on_write_once_stores(tmp_path: Path) -> None:
    root = tmp_path / "bucket"
    recording = RecordingStore(LocalStore(root, publish="exclusive"))
    data = _payload(1)

    path = checkpoint.publish(recording, "step000000001.pt", data)

    assert recording.ops == [("put", "step000000001.pt"), ("put", "step000000001.pt.ready.json")]
    assert checkpoint.publication(path) == {
        "v": 1,
        "artifact_id": "checkpoint/step000000001.pt",
        "path": str(root / "step000000001.pt"),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "manifest_path": str(root / "step000000001.pt.ready.json"),
    }
    # A write-once store gets no latest.pt: without symlinks it would be a full copy.
    checkpoint.publish(LocalStore(root, publish="exclusive"), "step000000002.pt", _payload(2))
    assert not (root / "latest.pt").exists()
    assert sorted(p.name for p in root.iterdir()) == [
        "step000000001.pt",
        "step000000001.pt.ready.json",
        "step000000002.pt",
        "step000000002.pt.ready.json",
    ]


def test_republishing_a_step_uncommits_it_before_replacing_it(tmp_path: Path) -> None:
    recording = RecordingStore(LocalStore(tmp_path, publish="exclusive"))
    checkpoint.publish(recording, "step000000003.pt", _payload(3))
    recording.ops.clear()
    replacement = _payload(3, next_step=3)

    path = checkpoint.publish(recording, "step000000003.pt", replacement)

    assert recording.ops == [
        ("delete", "step000000003.pt.ready.json"),
        ("delete", "step000000003.pt"),
        ("put", "step000000003.pt"),
        ("put", "step000000003.pt.ready.json"),
    ]
    assert Path(path).read_bytes() == replacement
    assert checkpoint.publication(path)["sha256"] == hashlib.sha256(replacement).hexdigest()


def test_prune_uncommits_each_old_checkpoint_before_deleting_it(tmp_path: Path) -> None:
    recording = RecordingStore(LocalStore(tmp_path, publish="exclusive"))
    for step in (1, 2, 3):
        checkpoint.publish(recording, f"step{step:09d}.pt", _payload(step), keep_last_k=2)

    deletes = [key for op, key in recording.ops if op == "delete"]
    assert deletes == ["step000000001.pt.ready.json", "step000000001.pt"]
    assert sorted(p.name for p in tmp_path.glob("step*.pt")) == ["step000000002.pt", "step000000003.pt"]


def test_unreadable_checkpoint_raises_instead_of_rolling_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalStore(tmp_path, publish="exclusive")
    for step in (2, 4):
        checkpoint.publish(store, f"step{step:09d}.pt", _payload(step))
    real_get = LocalStore.get

    def failing_get(self, key, offset=0, length=None):
        if key.startswith("step000000004"):
            raise OSError(errno.EIO, "Input/output error")
        return real_get(self, key, offset, length)

    monkeypatch.setattr(LocalStore, "get", failing_get)
    with pytest.raises(OSError) as raised:
        checkpoint.resolve_auto_resume(store)
    assert raised.value.errno == errno.EIO


def test_resume_waits_out_files_that_are_still_settling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalStore(tmp_path, publish="exclusive", read_settle_seconds=60)
    checkpoint.publish(store, "step000000005.pt", _payload(5))
    real_open = open
    refusals = []

    def settling_open(path, mode="r", *args, **kwargs):
        if len(refusals) < 2:
            refusals.append(path)
            raise OSError(errno.ETIME, "Timer expired")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(store_lib, "open", settling_open, raising=False)
    monkeypatch.setattr(store_lib.time, "sleep", lambda _seconds: None)

    resolved = checkpoint.resolve_auto_resume(store)

    assert resolved is not None and resolved[0] == str(tmp_path / "step000000005.pt")
    assert len(refusals) == 2


def test_checkpoint_dir_uri_keeps_checkpoints_apart_from_out_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bucket = tmp_path / "bucket"
    stores = tmp_path / "stores.yaml"
    stores.write_text(
        "version: 1\nstores:\n  objects:\n    type: local\n"
        f"    root: {bucket}\n    publish: exclusive\n    read_settle_seconds: 5\n"
    )
    monkeypatch.setenv("KOOCHAK_STORES", str(stores))
    out_dir = tmp_path / "run"

    _train(_train_cfg(out_dir, checkpoint_dir="objects://runs/a", resume="auto"))

    checkpoints = bucket / "runs" / "a"
    assert sorted(p.name for p in checkpoints.glob("step*.pt")) == [
        "step000000001.pt",
        "step000000002.pt",
        "step000000003.pt",
    ]
    assert not list(out_dir.glob("step*")) and not (checkpoints / "latest.pt").exists()
    assert checkpoint.highest_valid_published("objects://runs/a") == str(checkpoints / "step000000003.pt")
    assert checkpoint.latest("objects://runs/a") == str(checkpoints / "step000000003.pt")

    steps: list[int] = []
    resumed = _train(
        _train_cfg(out_dir, checkpoint_dir="objects://runs/a", resume="auto", max_steps=5),
        hooks={"on_step_end": [lambda _logs, ctx: steps.append(int(ctx["step"]))]},
    )
    assert steps == [3, 4]
    assert resumed["next_step"] == 5


def test_background_saves_announce_each_checkpoint_after_it_commits(tmp_path: Path) -> None:
    announced: list[tuple[str, bool]] = []

    def on_checkpoint(path: str, ckpt: Mapping[str, Any], _ctx: Mapping[str, Any]) -> None:
        record = checkpoint.publication(path)
        announced.append((Path(path).name, record["sha256"] == hashlib.sha256(Path(path).read_bytes()).hexdigest()))
        assert ckpt["step"] == int(Path(path).stem[4:])

    _train(
        _train_cfg(tmp_path, checkpoint_async=True, max_steps=4),
        hooks={"on_checkpoint": [on_checkpoint]},
    )

    assert announced == [
        ("step000000001.pt", True),
        ("step000000002.pt", True),
        ("step000000003.pt", True),
        ("step000000004.pt", True),
    ]


def test_failed_background_save_fails_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_publish(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(checkpoint, "publish", broken_publish)
    with pytest.raises(OSError) as raised:
        _train(_train_cfg(tmp_path, checkpoint_async=True, max_steps=4))
    assert raised.value.errno == errno.ENOSPC


def test_only_one_background_publication_may_be_pending(tmp_path: Path) -> None:
    publisher = checkpoint.BackgroundPublisher(LocalStore(tmp_path), keep_last_k=3)
    try:
        path = publisher.submit("step000000001.pt", _payload(1), token="first")
        with pytest.raises(RuntimeError):
            publisher.submit("step000000002.pt", _payload(2))
        assert publisher.wait() == (path, "first")
        assert json.loads(Path(checkpoint.publication_path(path)).read_text())["path"] == path
        assert publisher.wait() is None
    finally:
        publisher.close()
