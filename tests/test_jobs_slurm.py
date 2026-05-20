from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from koochak.jobs import (
    CommandResult,
    ConfigPatch,
    RemotePaths,
    RuntimeFlags,
    SlurmResources,
    SshCommandRunner,
    SshSlurmBackend,
    TrainJobSpec,
    materialize_config,
)


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, bool]] = []

    def run(self, remote_command: str, *, input_text: str | None = None, check: bool = True) -> CommandResult:
        self.calls.append((remote_command, input_text, check))
        if remote_command.startswith("sbatch "):
            return CommandResult(args=["fake", remote_command], returncode=0, stdout="12345\n", stderr="")
        if remote_command.startswith("squeue "):
            return CommandResult(
                args=["fake", remote_command],
                returncode=0,
                stdout="12345|RUNNING|00:01:00|02:00:00|None\n",
                stderr="",
            )
        if remote_command.startswith("sacct "):
            return CommandResult(
                args=["fake", remote_command],
                returncode=0,
                stdout="12345|smoke|COMPLETED|0:0|00:02:00\n",
                stderr="",
            )
        if remote_command.startswith("tail "):
            return CommandResult(args=["fake", remote_command], returncode=0, stdout="hello\n", stderr="")
        if "log.jsonl" in remote_command:
            return CommandResult(
                args=["fake", remote_command],
                returncode=0,
                stdout='{"step": 1, "loss": 2.5}\nnot-json\n{"step": 2, "loss": 2.0}\n',
                stderr="",
            )
        return CommandResult(args=["fake", remote_command], returncode=0, stdout="", stderr="")


def _base_config(tmp_path: Path) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        """
train:
  max_steps: 10
  out_dir: ./runs/default
data:
  max_length: 64
wandb:
  enabled: true
"""
    )
    return path


def _job(tmp_path: Path) -> TrainJobSpec:
    return TrainJobSpec(
        name="smoke len128",
        base_config=_base_config(tmp_path),
        patches=[
            ConfigPatch("train.max_steps", 500),
            ConfigPatch("data.max_length", 128),
            ConfigPatch("wandb.enabled", False),
        ],
        command=["-m", "my_pkg.train", "--config", "{config}"],
        runtime=RuntimeFlags(wandb_mode="disabled", wandb_disabled=True, omp_num_threads=4),
        resources=SlurmResources(partition="gpu", gpus=1, cpus=32, mem_gb=128, time="02:00:00"),
    )


def test_materialize_config_applies_patches_and_run_dir(tmp_path: Path) -> None:
    text = materialize_config(
        _base_config(tmp_path),
        run_dir="/remote/runs/smoke",
        patches=[ConfigPatch("train.max_steps", 7), ConfigPatch("new.section", {"x": 1})],
    )
    cfg = OmegaConf.create(text)

    assert cfg.train.out_dir == "/remote/runs/smoke"
    assert cfg.train.max_steps == 7
    assert cfg.new.section.x == 1


def test_slurm_resources_require_explicit_memory() -> None:
    with pytest.raises(ValueError, match="mem_gb"):
        SlurmResources(partition="gpu", gpus=1, cpus=8, mem_gb=None)


def test_render_writes_safe_config_and_sbatch(tmp_path: Path) -> None:
    backend = SshSlurmBackend(
        runner=FakeRunner(),
        remote_paths=RemotePaths(
            repo="/remote/repo",
            run_root="/remote/runs",
            python="/remote/env/bin/python",
        ),
    )
    rendered = backend.render(_job(tmp_path), local_dir=tmp_path / "rendered")
    cfg = OmegaConf.create(rendered.config_text)

    assert rendered.name == "smoke-len128"
    assert cfg.train.out_dir == "/remote/runs/smoke-len128"
    assert cfg.train.max_steps == 500
    assert cfg.data.max_length == 128
    assert cfg.wandb.enabled is False

    assert "#SBATCH --mem=128G" in rendered.sbatch_text
    assert "#SBATCH --gres=gpu:1" in rendered.sbatch_text
    assert "branch=$(git rev-parse --abbrev-ref HEAD)" in rendered.sbatch_text
    assert '"$PYTHON" -m my_pkg.train --config /remote/runs/smoke-len128/config.yaml' in rendered.sbatch_text
    assert "max_steps" not in rendered.sbatch_text

    assert (tmp_path / "rendered" / "config.yaml").exists()
    assert (tmp_path / "rendered" / "job.sbatch").exists()
    assert (tmp_path / "rendered" / "manifest.json").exists()


def test_ssh_command_runner_uses_injected_command_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class Proc:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    def fake_run(args, **kwargs):  # noqa: ANN001
        seen["args"] = args
        seen["kwargs"] = kwargs
        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = SshCommandRunner(["/tmp/mux-helper"])
    result = runner.run(["echo", "hello world"])

    assert result.stdout == "ok\n"
    assert seen["args"] == ["/tmp/mux-helper", "echo 'hello world'"]
    assert seen["kwargs"]["text"] is True  # type: ignore[index]


def test_submit_stages_files_and_parses_handle_status_tail_and_metrics(tmp_path: Path) -> None:
    runner = FakeRunner()
    backend = SshSlurmBackend(
        runner=runner,
        remote_paths=RemotePaths(repo="/remote/repo", run_root="/remote/runs", python="python"),
    )

    handle = backend.submit(_job(tmp_path))

    assert handle.job_id == "12345"
    assert handle.stdout_path == "/remote/runs/smoke-len128/slurm_logs/smoke-len128-12345.out"
    staged_inputs = [input_text for _, input_text, _ in runner.calls if input_text is not None]
    assert len(staged_inputs) == 3
    assert "max_steps: 500" in staged_inputs[0]
    assert "#SBATCH --mem=128G" in staged_inputs[1]
    assert '"name": "smoke len128"' in staged_inputs[2]

    status = backend.status(handle)
    assert status.state == "RUNNING"
    assert status.source == "squeue"
    assert status.elapsed == "00:01:00"

    assert backend.tail(handle) == "hello\n"
    assert backend.metrics(handle) == [{"step": 1, "loss": 2.5}, {"step": 2, "loss": 2.0}]


def test_status_falls_back_to_sacct(tmp_path: Path) -> None:
    class SacctRunner(FakeRunner):
        def run(self, remote_command: str, *, input_text: str | None = None, check: bool = True) -> CommandResult:
            if remote_command.startswith("squeue "):
                return CommandResult(args=["fake", remote_command], returncode=0, stdout="", stderr="")
            return super().run(remote_command, input_text=input_text, check=check)

    backend = SshSlurmBackend(
        runner=SacctRunner(),
        remote_paths=RemotePaths(repo="/remote/repo", run_root="/remote/runs", python="python"),
    )

    status = backend.status("12345")

    assert status.state == "COMPLETED"
    assert status.source == "sacct"
    assert status.exit_code == "0:0"
