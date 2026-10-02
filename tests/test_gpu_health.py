from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import koochak.loop as loop_module
import koochak.health.gpu as gpu_health_module
from koochak.health.gpu import (
    PRIMARY_QUERY_FIELDS,
    THERMAL_GRACE_SECONDS,
    THERMAL_MAX_SAMPLE_GAP_SECONDS,
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
        hostname="gpu-8.example.org",
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


def test_watchdog_requires_two_consecutive_nonthermal_bad_samples(tmp_path: Path) -> None:
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)
    watchdog.enabled = True
    watchdog._query_sample = lambda step: _sample(  # type: ignore[method-assign]
        step=step, gpu_temp_c=50., sw_thermal_slowdown=False,
    )

    assert watchdog.should_check_step(19) is False
    assert watchdog.should_check_step(20) is True
    assert watchdog.check_local(20) is None

    failure = watchdog.check_local(30)

    assert failure is not None
    assert failure.consecutive_failures == 2
    assert failure.reasons == ["sm_clock_low"]


def test_good_sample_resets_consecutive_failure_counter(tmp_path: Path) -> None:
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)
    watchdog.enabled = True
    samples = iter(
        [
            _sample(step=20, gpu_temp_c=50., sw_thermal_slowdown=False),
            _sample(step=30, gpu_temp_c=40.0, memory_temp_c=45.0, sm_clock_mhz=1980.0, sw_thermal_slowdown=False),
            _sample(step=40, gpu_temp_c=50., sw_thermal_slowdown=False),
        ]
    )
    watchdog._query_sample = lambda step: next(samples)  # type: ignore[method-assign]

    assert watchdog.check_local(20) is None
    assert watchdog.check_local(30) is None
    assert watchdog.check_local(40) is None


@pytest.fixture
def timed_watchdog(tmp_path, monkeypatch):
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)
    clock = [0.0]
    monkeypatch.setattr(gpu_health_module.time, "monotonic", lambda: clock[0])

    def check(seconds, sample):
        clock[0] = seconds
        watchdog._query_sample = lambda step: sample
        return watchdog.check_local(20)

    return check


@pytest.mark.parametrize("overrides", [
    {},  # Software throttling, high temperature and low clocks together.
    {"gpu_temp_c": 73., "sm_clock_mhz": 1950.},  # Actual gpu-0 incident.
    {"sw_thermal_slowdown": False},  # Temperature alone, as on gpu-5.
    {"gpu_temp_c": 50., "sw_thermal_slowdown": False, "memory_temp_c": 95.},
    {"hw_thermal_slowdown": True, "hw_slowdown": True},
])
def test_thermal_shutdown_requires_two_hours_not_two_samples(timed_watchdog, tmp_path, overrides):
    assert THERMAL_GRACE_SECONDS == 7200
    sample = _sample(**overrides)
    for seconds in range(0, THERMAL_GRACE_SECONDS, 30):
        assert timed_watchdog(seconds, sample) is None
    assert timed_watchdog(THERMAL_GRACE_SECONDS - .001, sample) is None
    failure = timed_watchdog(THERMAL_GRACE_SECONDS, sample)
    assert failure is not None
    assert failure.thermal_duration_seconds == THERMAL_GRACE_SECONDS
    assert failure.consecutive_failures == 242
    assert failure.to_dict()["thermal_duration_seconds"] == THERMAL_GRACE_SECONDS
    rows = (tmp_path / "gpu_health/gpu_health_rank0.jsonl").read_text().splitlines()
    assert json.loads(rows[-2])["thermal_duration_seconds"] < THERMAL_GRACE_SECONDS
    assert json.loads(rows[-1])["thermal_grace_seconds"] == THERMAL_GRACE_SECONDS


@pytest.mark.parametrize("interruption", ["healthy", "missing", "idle", "gap"])
def test_thermal_grace_resets_without_continuing_evidence(timed_watchdog, interruption):
    for seconds in range(0, THERMAL_GRACE_SECONDS, 30):
        assert timed_watchdog(seconds, _sample()) is None
    now = THERMAL_GRACE_SECONDS
    if interruption == "gap":
        now += THERMAL_MAX_SAMPLE_GAP_SECONDS + 1
        assert timed_watchdog(now, _sample()) is None
    else:
        sample = {"healthy": _sample(gpu_temp_c=50., sm_clock_mhz=1980., sw_thermal_slowdown=False),
                  "idle": _sample(gpu_util_pct=0.), "missing": None}[interruption]
        assert timed_watchdog(now, sample) is None
    assert timed_watchdog(now + 30, _sample()) is None


@pytest.mark.parametrize("fault", ["hw_power_brake_slowdown", "hw_slowdown", "sm_clock_low"])
def test_nonthermal_faults_do_not_borrow_thermal_sample_count(timed_watchdog, fault):
    assert timed_watchdog(0, _sample()) is None
    kwargs = {"gpu_temp_c": 50., "sm_clock_mhz": 1980., "sw_thermal_slowdown": False}
    if fault == "sm_clock_low":
        kwargs["sm_clock_mhz"] = 500.
    else:
        kwargs[fault] = True
    bad = _sample(**kwargs)
    assert timed_watchdog(30, bad) is None
    failure = timed_watchdog(60, bad)
    assert failure is not None and failure.reasons == [fault]
    assert failure.consecutive_failures == 2


def test_power_brake_remains_fast_even_during_thermal_grace(timed_watchdog):
    bad = _sample(hw_power_brake_slowdown=True, hw_slowdown=True)
    assert timed_watchdog(0, bad) is None
    failure = timed_watchdog(30, bad)
    assert failure is not None and failure.reasons == ["hw_power_brake_slowdown"]


def test_wall_clock_jump_does_not_expire_thermal_grace(timed_watchdog):
    assert timed_watchdog(0, _sample(timestamp_unix=0)) is None
    assert timed_watchdog(30, _sample(timestamp_unix=100000)) is None


def test_slurm_exit_is_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "7832")
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)
    watchdog._run_command = lambda cmd: pytest.fail(f"unexpected slurm command: {cmd}")  # type: ignore[method-assign]

    assert watchdog.slurm_action == "exit"
    assert watchdog.perform_slurm_recovery([_sample().to_dict()]) == []


def test_explicit_slurm_requeue_updates_pending_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("USER", "researcher")
    monkeypatch.setenv("SLURM_JOB_ID", "7832")
    monkeypatch.setenv("SLURM_JOB_NAME", "project_training")
    monkeypatch.setenv("KOOCHAK_GPU_HEALTH_SLURM_ACTION", "requeue")
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


@pytest.mark.parametrize(
    ("runtime_env", "reason"),
    [
        ({"SCRUFFY_JOB_ID": "job-123"}, "Scruffy workloads"),
        ({"SCRUFFY_ROOT": "/shared/scruffy"}, "Scruffy workloads"),
        ({"SLURM_STEP_ID": "17"}, "nested step 17"),
        ({"SLURM_STEPID": "18"}, "nested step 18"),
    ],
)
def test_nested_workload_cannot_mutate_parent_slurm_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime_env: dict[str, str],
    reason: str,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "263106")
    monkeypatch.setenv("KOOCHAK_GPU_HEALTH_SLURM_ACTION", "requeue")
    for key, value in runtime_env.items():
        monkeypatch.setenv(key, value)
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)
    watchdog._run_command = lambda cmd: pytest.fail(f"unexpected slurm command: {cmd}")  # type: ignore[method-assign]

    assert watchdog.slurm_action == "exit"
    assert watchdog.requested_slurm_action == "requeue"
    assert reason in (watchdog.slurm_mutation_blocked_reason or "")
    assert watchdog.perform_slurm_recovery([_sample().to_dict()]) == []


def test_slurm_disable_prevents_all_slurm_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KOOCHAK_GPU_HEALTH_SLURM_DISABLE", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "7832")
    watchdog = GpuHealthWatchdog(device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1)
    watchdog._run_command = lambda cmd: pytest.fail(f"unexpected slurm command: {cmd}")  # type: ignore[method-assign]

    assert watchdog.perform_slurm_recovery([_sample().to_dict()]) == []


def test_slurm_cancel_action_uses_scancel_not_requeue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("USER", "researcher")
    monkeypatch.setenv("SLURM_JOB_ID", "7832")
    monkeypatch.setenv("SLURM_JOB_NAME", "project_training")
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

    monkeypatch.setattr(loop_module, "GpuHealthWatchdog", FakeWatchdog)

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
