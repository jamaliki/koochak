from dataclasses import replace
import json
import sys

import numpy as np
import pytest

from hierarchical_kaveh.config import (
    DataConfig,
    EmaConfig,
    LoggingConfig,
    ModelConfig,
    RunConfig,
    TrainingConfig,
    WandbConfig,
)
from hierarchical_kaveh.train import _parser
import hierarchical_kaveh.training as training_module
from hierarchical_kaveh.training import _hooks, _resume_checkpoint, run_training


class _FakeWandb:
    class Settings:
        pass

    def __init__(self) -> None:
        self.init_calls: list[dict] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return object()


def test_koochak_step_checkpoint_and_resume(tmp_path, monkeypatch) -> None:
    length = 5
    generator = np.random.default_rng(1)
    np.savez_compressed(
        tmp_path / "shard.npz",
        sample_offsets=np.array([0, length]),
        pos=generator.normal(size=(length, 14, 3)).astype(np.float32),
        mask=np.ones((length, 14), dtype=np.bool_),
        aatype=np.arange(length, dtype=np.int64),
        chain_idx=np.zeros(length, dtype=np.int64),
        res_idx=np.arange(length, dtype=np.int64),
        cond=np.array([[90.0]], dtype=np.float32),
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            [
                {
                    "shard": "shard.npz",
                    "cond_feature_names": ["mean_plddt"],
                    "ca_distance_validation_max": 4.0,
                    "excluded_ca_distance_samples": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    model = ModelConfig(
        node_dim=32,
        condition_dim=16,
        pair_dim=8,
        atom_dim=16,
        attention_heads=4,
        attention_head_dim=8,
        atom_heads=2,
        atom_head_dim=8,
        atom_encoder_depth=1,
        residue_encoder_depth=1,
        coarse_depth=1,
        residue_decoder_depth=1,
        atom_decoder_depth=1,
        residue_ffn_expansion=2,
        atom_ffn_expansion=2,
        dropout=0.0,
        pair_rbf_bins=4,
        distogram_bins=8,
    )
    data = DataConfig(
        metadata_path=str(metadata),
        min_length=4,
        max_length=8,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        length_buckets=(8,),
    )
    training = TrainingConfig(
        max_steps=1,
        log_every=1,
        ckpt_every=1,
        save_final=True,
        amp="fp32",
        ddp=False,
        device="cpu",
        out_dir=str(tmp_path / "run"),
        prefetch_batches=0,
        prefetch_threaded=False,
        ema=EmaConfig(offload_to_cpu=False, update_every=1),
    )
    config = RunConfig(
        model=model,
        data=data,
        train=training,
        logging=LoggingConfig(),
    )
    observed_loader_steps: list[int] = []
    real_build_loader = training_module.build_train_dataloader

    def record_loader_step(*args, **kwargs):
        observed_loader_steps.append(int(kwargs["global_step"]))
        return real_build_loader(*args, **kwargs)

    monkeypatch.setattr(training_module, "build_train_dataloader", record_loader_step)

    # The same immutable command is valid for both the first attempt and a
    # later attempt. Koochak starts cleanly when no publication exists.
    first = run_training(config, resume="auto")
    assert first["next_step"] == 1
    assert first["config"]["model"] == config.to_dict()["model"]
    assert "ema" in first

    resumed = run_training(replace(config, train=replace(training, max_steps=2)), resume="auto")
    assert resumed["next_step"] == 2
    assert observed_loader_steps == [0, 1]


def test_project_does_not_select_auto_resume_checkpoint() -> None:
    """Auto selection stays in Koochak, including publication validation."""

    assert _resume_checkpoint("auto", "/missing/run") is None


def test_training_cli_accepts_auto_resume() -> None:
    args = _parser().parse_args(["--config", "config.yaml", "--resume", "auto"])
    assert args.resume == "auto"


def test_auto_resume_wandb_identity_is_stable_and_allows_join(monkeypatch) -> None:
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = RunConfig(
        wandb=WandbConfig(
            enabled=True,
            project="hierarchical-kaveh-test",
            name="stable-training-run",
            id="stable-wandb-id",
            resume="allow",
        )
    )

    hooks = _hooks(config)
    context = {"auto_resume_selected": True, "config_json": {}, "train_cfg": {}}
    for _ in range(2):
        for callback in hooks["on_train_start"]:
            callback(context)

    assert [call["id"] for call in fake.init_calls] == [
        "stable-wandb-id",
        "stable-wandb-id",
    ]
    assert [call["resume"] for call in fake.init_calls] == ["allow", "allow"]


def test_scruffy_hooks_absent_for_ordinary_training(monkeypatch) -> None:
    monkeypatch.delenv("SCRUFFY_ROOT", raising=False)
    monkeypatch.delenv("SCRUFFY_JOB_ID", raising=False)

    def unexpected_scruffy_hook() -> None:
        raise AssertionError("ordinary training must not build Scruffy hooks")

    monkeypatch.setattr(training_module, "make_scruffy_hooks", unexpected_scruffy_hook)
    assert "on_checkpoint" not in _hooks(RunConfig())


def test_scruffy_hooks_attach_after_local_checkpoint_hooks(monkeypatch) -> None:
    monkeypatch.setenv("SCRUFFY_ROOT", "/queue")
    monkeypatch.setenv("SCRUFFY_JOB_ID", "job-1")
    order: list[str] = []
    monkeypatch.setattr(
        training_module,
        "make_stdout_hooks",
        lambda: {"on_checkpoint": [lambda *_args: order.append("local")]},
    )
    monkeypatch.setattr(
        training_module,
        "make_scruffy_hooks",
        lambda: {"on_checkpoint": [lambda *_args: order.append("scruffy")]},
    )
    hooks = _hooks(RunConfig())
    for callback in hooks["on_checkpoint"]:
        callback("/queue/step000000002.pt", {}, {})
    assert order == ["local", "scruffy"]


def test_scruffy_partial_worker_identity_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("SCRUFFY_ROOT", "/queue")
    monkeypatch.delenv("SCRUFFY_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="requires both SCRUFFY_ROOT and SCRUFFY_JOB_ID"):
        _hooks(RunConfig())
