from __future__ import annotations

import json

from koochak.integrations.morbo import create_hooks, resolve_identity


def test_morbo_identity_keeps_run_and_rotates_attempt(tmp_path) -> None:
    first = resolve_identity({}, str(tmp_path))
    second = resolve_identity({}, str(tmp_path))

    assert first.run_id == second.run_id
    assert first.attempt_id != second.attempt_id
    assert json.loads((tmp_path / ".morbo-identity.json").read_text()) == {"run_id": first.run_id}


def test_morbo_identity_prefers_checkpoint_run_id(tmp_path) -> None:
    identity = resolve_identity(
        {},
        str(tmp_path),
        checkpoint={"morbo": {"run_id": "run-from-checkpoint"}},
    )

    assert identity.run_id == "run-from-checkpoint"


def test_create_hooks_is_optional_and_uses_configured_identity(tmp_path) -> None:
    disabled = create_hooks({"enabled": False}, {"out_dir": str(tmp_path)})
    assert disabled == (None, None, None)

    hooks, client, identity = create_hooks(
        {
            "enabled": True,
            "project_id": "project/test",
            "run_id": "run-explicit",
            "attempt_id": "attempt-explicit",
            "run_name": "test-run",
            "socket_path": str(tmp_path / "agent.sock"),
            "weight_reduction": "cpu",
        },
        {"out_dir": str(tmp_path)},
    )
    try:
        assert "on_train_start" in hooks
        assert client.project_id == "project/test"
        assert client.run_id == "run-explicit"
        assert client.attempt_id == "attempt-explicit"
        assert identity.run_name == "test-run"
    finally:
        client.close()
