from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from koochak.health.gpu import (
    PRIMARY_QUERY_FIELDS,
    GpuHealthSample,
    GpuHealthWatchdog,
    evaluate_sample,
    parse_nvidia_smi_csv,
)
from koochak.loop import training_loop
from koochak.storage import checkpoint as checkpoint_lib


def _sample(**overrides) -> GpuHealthSample:
    data = {
        "step": 20,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "hostname": "gpu-8",
        "slurm_node": "gpu-8",
        "cuda_device": 2,
        "gpu_query_id": "2",
        "gpu_index": "2",
        "gpu_uuid": "GPU-694e5085-2dbd-d289-d02c-540a63b4f602",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_temp_c": 87.0,
        "memory_temp_c": 93.0,
        "pstate": "P0",
        "power_draw_w": 257.83,
        "power_limit_w": 700.0,
        "sm_clock_mhz": 615.0,
        "mem_clock_mhz": 2619.0,
        "max_sm_clock_mhz": 1980.0,
        "gpu_util_pct": 100.0,
        "mem_util_pct": 47.0,
        "throttle_active_mask": "0x0000000000000020",
        "sw_thermal_slowdown": True,
        "hw_slowdown": False,
        "hw_thermal_slowdown": False,
        "hw_power_brake_slowdown": False,
        "sw_power_cap": False,
        "timestamp_unix": 1.0,
    }
    data.update(overrides)
    return GpuHealthSample(**data)


def test_throttled_h100_sample_triggers_multiple_reasons() -> None:
    reasons = evaluate_sample(_sample())

    assert "sw_thermal_slowdown" in reasons
    assert "gpu_temp_high" in reasons
    assert "sm_clock_low" in reasons


def test_healthy_h100_sample_does_not_trigger() -> None:
    reasons = evaluate_sample(
        _sample(
            gpu_temp_c=45.0,
            memory_temp_c=50.0,
            sm_clock_mhz=1980.0,
            sw_thermal_slowdown=False,
            throttle_active_mask="0x0000000000000000",
        )
    )

    assert reasons == []


def test_sw_power_cap_alone_does_not_trigger() -> None:
    reasons = evaluate_sample(
        _sample(
            gpu_temp_c=60.0,
            memory_temp_c=70.0,
            sm_clock_mhz=1980.0,
            sw_thermal_slowdown=False,
            sw_power_cap=True,
            throttle_active_mask="0x0000000000000004",
        )
    )

    assert reasons == []


def test_low_utilization_suppresses_failure() -> None:
    assert evaluate_sample(_sample(gpu_util_pct=5.0)) == []


def test_parse_nvidia_smi_csv_preserves_rank_node_and_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_LOCALID", "2")
    row = [
        "2",
        "GPU-694e5085-2dbd-d289-d02c-540a63b4f602",
        "NVIDIA H100 80GB HBM3",
        "86",
        "93",
        "P0",
        "257.83",
        "700.00",
        "645",
        "2619",
        "1980",
        "100",
        "47",
        "0x0000000000000020",
        "Active",
        "Not Active",
        "Not Active",
        "Not Active",
        "Not Active",
    ]
    sample = parse_nvidia_smi_csv(
        stdout=",".join(row),
        fields=PRIMARY_QUERY_FIELDS,
        step=30,
        rank=9,
        world_size=16,
        hostname="gpu-8.eit-gbi.science",
        slurm_node="gpu-8",
        cuda_device=2,
        gpu_query_id="2",
    )

    assert sample is not None
    assert sample.rank == 9
    assert sample.local_rank == 2
    assert sample.slurm_node == "gpu-8"
    assert sample.gpu_uuid == "GPU-694e5085-2dbd-d289-d02c-540a63b4f602"
    assert sample.sw_thermal_slowdown is True
    assert sample.max_sm_clock_mhz == 1980.0


def test_watchdog_requires_two_consecutive_bad_samples(tmp_path: Path) -> None:
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)
    watchdog.enabled = True
    watchdog._query_sample = lambda step: _sample(step=step)  # type: ignore[method-assign]

    assert watchdog.should_check_step(19) is False
    assert watchdog.should_check_step(20) is True
    assert watchdog.check_local(20) is None

    failure = watchdog.check_local(30)

    assert failure is not None
    assert failure.consecutive_failures == 2
    assert "sw_thermal_slowdown" in failure.reasons


def test_good_sample_resets_consecutive_failure_counter(tmp_path: Path) -> None:
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)
    watchdog.enabled = True
    samples = iter(
        [
            _sample(step=20),
            _sample(step=30, gpu_temp_c=40.0, memory_temp_c=45.0, sm_clock_mhz=1980.0, sw_thermal_slowdown=False),
            _sample(step=40),
        ]
    )
    watchdog._query_sample = lambda step: next(samples)  # type: ignore[method-assign]

    assert watchdog.check_local(20) is None
    assert watchdog.check_local(30) is None
    assert watchdog.check_local(40) is None


def test_slurm_requeue_is_default_and_updates_pending_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("USER", "kiarash-eitgbi")
    monkeypatch.setenv("SLURM_JOB_ID", "7832")
    monkeypatch.setenv("SLURM_JOB_NAME", "kaveh_pf_norm_p2n")
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)

    commands = []

    def fake_run(cmd):
        commands.append(cmd)
        if cmd[0] == "squeue":
            return {"cmd": " ".join(cmd), "returncode": 0, "stdout": "7833\n7834\n", "stderr": ""}
        return {"cmd": " ".join(cmd), "returncode": 0, "stdout": "", "stderr": ""}

    watchdog._run_command = fake_run  # type: ignore[method-assign]

    results = watchdog.perform_slurm_recovery([_sample().to_dict()])

    assert results
    assert commands[0] == ["scontrol", "update", "JobId=7832", "ExcNodeList=gpu-8"]
    assert ["scontrol", "update", "JobId=7833", "ExcNodeList=gpu-8"] in commands
    assert ["scontrol", "update", "JobId=7834", "ExcNodeList=gpu-8"] in commands
    assert commands[-1] == ["scontrol", "requeue", "7832"]


def test_slurm_disable_prevents_all_slurm_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KOOCHAK_GPU_HEALTH_SLURM_DISABLE", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "7832")
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)
    watchdog._run_command = lambda cmd: pytest.fail(f"unexpected slurm command: {cmd}")  # type: ignore[method-assign]

    assert watchdog.perform_slurm_recovery([_sample().to_dict()]) == []


def test_slurm_cancel_action_uses_scancel_not_requeue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("USER", "kiarash-eitgbi")
    monkeypatch.setenv("SLURM_JOB_ID", "7832")
    monkeypatch.setenv("SLURM_JOB_NAME", "kaveh_pf_norm_p2n")
    monkeypatch.setenv("KOOCHAK_GPU_HEALTH_SLURM_ACTION", "cancel")
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)

    commands = []

    def fake_run(cmd):
        commands.append(cmd)
        if cmd[0] == "squeue":
            return {"cmd": " ".join(cmd), "returncode": 0, "stdout": "7833\n", "stderr": ""}
        return {"cmd": " ".join(cmd), "returncode": 0, "stdout": "", "stderr": ""}

    watchdog._run_command = fake_run  # type: ignore[method-assign]

    watchdog.perform_slurm_recovery([_sample().to_dict()])

    assert ["scancel", "7832"] in commands
    assert not any(cmd[:2] == ["scontrol", "requeue"] for cmd in commands)


def test_slurm_exit_action_calls_neither_requeue_nor_cancel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "7832")
    monkeypatch.setenv("KOOCHAK_GPU_HEALTH_SLURM_ACTION", "exit")
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)
    watchdog._run_command = lambda cmd: pytest.fail(f"unexpected slurm command: {cmd}")  # type: ignore[method-assign]

    assert watchdog.perform_slurm_recovery([_sample().to_dict()]) == []


def test_training_loop_writes_emergency_checkpoint_and_failure_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeWatchdog:
        enabled = True
        slurm_disabled = True
        slurm_action = "exit"
        exit_code = 42

        def __init__(self, *, device, out_dir, rank, world_size):
            self.out_dir = Path(out_dir)

        def should_check_step(self, step):
            return step == 0

        def check_local(self, step):
            return None

        def gather_failures(self, local_failure):
            return [_sample(step=0).to_dict()]

        def bad_nodes_from_failures(self, failures):
            return ["gpu-8"]

        def write_failure_summary(self, *, step, failures, checkpoint_path, slurm_results=None):
            path = self.out_dir / "gpu_health" / f"gpu_health_failure_step{step:09d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "step": step,
                        "failures": failures,
                        "checkpoint_path": checkpoint_path,
                        "slurm_action": self.slurm_action,
                    }
                ),
                encoding="utf-8",
            )
            return str(path)

        def perform_slurm_recovery(self, failures):
            return []

    monkeypatch.setattr("koochak.loop.GpuHealthWatchdog", FakeWatchdog)

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def step_fn(model, batch, ctx):
        return {"loss": model(batch).square().sum()}

    train_cfg = {
        "device": "cpu",
        "max_steps": 2,
        "grad_accum": 1,
        "log_every": 1000,
        "eval_every": 1000,
        "ckpt_every": 1000,
        "amp": "fp32",
        "out_dir": str(tmp_path),
        "keep_last_k": 5,
    }

    with pytest.raises(SystemExit) as exc:
        training_loop(
            model=model,
            dataset=[torch.ones(1, 1)],
            step_fn=step_fn,
            optimizer=optimizer,
            train_cfg=train_cfg,
        )

    assert exc.value.code == 42
    checkpoint_path = tmp_path / "step000000000.pt"
    failure_path = tmp_path / "gpu_health" / "gpu_health_failure_step000000000.json"
    assert checkpoint_path.exists()
    assert failure_path.exists()
    assert (tmp_path / "latest.pt").exists()

    ckpt = checkpoint_lib.load(str(checkpoint_path))
    assert ckpt["step"] == 0
    assert ckpt["metrics"]["gpu_health"]["bad_nodes"] == ["gpu-8"]
    assert ckpt["metrics"]["gpu_health"]["failures"][0]["gpu_uuid"] == "GPU-694e5085-2dbd-d289-d02c-540a63b4f602"

    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["failures"][0]["rank"] == 0
    assert failure["failures"][0]["slurm_node"] == "gpu-8"
