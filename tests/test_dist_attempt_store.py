"""Each torchrun restart attempt rendezvouses under its own store prefix."""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch

from koochak.core import dist as dist_lib

_TIMEOUT = timedelta(seconds=10)


def test_no_attempt_store_outside_torchrun(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORCHELASTIC_USE_AGENT_STORE", raising=False)
    assert dist_lib.torchelastic_attempt_store(_TIMEOUT) is None


def test_a_restart_cannot_read_the_previous_attempts_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stands in for the torchrun agent's store, which outlives restarts.
    agent_store = torch.distributed.TCPStore("127.0.0.1", 0, is_master=True, timeout=_TIMEOUT)
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(agent_store.port))
    attempts = []
    for restart_count in ("0", "1"):
        monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", restart_count)
        attempts.append(dist_lib.torchelastic_attempt_store(_TIMEOUT))

    attempts[0].set("default_pg/0", "attempt 0's NCCL id")

    assert not attempts[1].check(["default_pg/0"])
    assert agent_store.get("/koochak/attempt_0/default_pg/0") == b"attempt 0's NCCL id"
