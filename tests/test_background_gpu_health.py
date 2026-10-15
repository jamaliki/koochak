from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from koochak.health import background
from koochak.health.gpu import PRIMARY_QUERY_FIELDS, GpuHealthWatchdog, sample_from_nvidia_smi_row
from koochak.health.node_query import NodeQuery, query_node_samples


def _row(index=0, uuid="GPU-first", **values):
    data = {
        "index": str(index), "uuid": uuid, "name": "H100", "temperature.gpu": "60",
        "temperature.memory": "65", "pstate": "P0", "power.draw": "500",
        "power.limit": "700", "clocks.sm": "1980", "clocks.mem": "2619",
        "clocks.max.sm": "1980", "utilization.gpu": "100", "utilization.memory": "45",
    }
    data.update(values)
    return [data.get(field, "Not Active") for field in PRIMARY_QUERY_FIELDS]


def _sample(watchdog, step):
    return sample_from_nvidia_smi_row(
        step=step, rank=watchdog.rank, world_size=watchdog.world_size,
        hostname=watchdog.hostname, slurm_node=watchdog.slurm_node,
        cuda_device=watchdog.cuda_device, gpu_query_id=watchdog.gpu_query_id,
        fields=PRIMARY_QUERY_FIELDS, row=_row(uuid=watchdog.gpu_query_id),
    )


def _watchdog(tmp_path, rank=0, query_id="GPU-first"):
    watchdog = GpuHealthWatchdog(
        device=torch.device("cuda", rank), out_dir=str(tmp_path), rank=rank, world_size=2,
    )
    watchdog.gpu_query_id = query_id
    watchdog.nvidia_smi = "nvidia-smi"
    return watchdog


def _monitor(tmp_path, monkeypatch):
    monitor = background.BackgroundGpuHealth(
        device=torch.device("cuda", 0), out_dir=str(tmp_path), rank=0, world_size=1,
    )
    monitor.enabled = True

    def properties(device):
        assert threading.current_thread() is threading.main_thread()
        return SimpleNamespace(uuid="GPU-a557a652-f56f-5884-75a6-5e0cf15a769e")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    return monitor


def _await(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    pytest.fail("monitor did not reach the expected state")


def test_node_query_maps_reordered_physical_uuids_in_one_call(tmp_path):
    watchdogs = [_watchdog(tmp_path), _watchdog(tmp_path, 1, "GPU-second")]
    calls = []

    def run(argv):
        calls.append(argv)
        rows = [_row(3, "GPU-second"), _row(7, "GPU-first")]
        return subprocess.CompletedProcess(argv, 0, "\n".join(",".join(row) for row in rows), "")

    samples = query_node_samples(watchdogs, 100, run=run)
    assert len(calls) == 1
    assert calls[0][2] == "GPU-first,GPU-second"
    assert [(sample.rank, sample.gpu_index) for sample in samples] == [(0, "7"), (1, "3")]


@pytest.mark.parametrize("rows", [
    [], [_row(), _row()], [_row(uuid="GPU-other")], [_row(**{"temperature.gpu": "N/A"})],
    [_row(**{"temperature.gpu": "nan"})], [_row(**{"clocks.sm": "inf"})],
    [_row(**{"temperature.memory": "-1"})], [_row()[:-1]],
])
def test_node_query_rejects_missing_ambiguous_or_invalid_rows(tmp_path, rows):
    def run(argv):
        return subprocess.CompletedProcess(argv, 0, "\n".join(",".join(row) for row in rows), "")

    with pytest.raises(RuntimeError):
        query_node_samples([_watchdog(tmp_path)], 0, run=run)


def test_node_query_fallback_preserves_optional_missing_fields(tmp_path):
    def run(argv):
        fields = argv[3].removeprefix("--query-gpu=").split(",")
        if "temperature.memory" in fields:
            return subprocess.CompletedProcess(argv, 1, "", "Unsupported field")
        values = dict(zip(PRIMARY_QUERY_FIELDS, _row()))
        return subprocess.CompletedProcess(argv, 0, ",".join(values[field] for field in fields), "")

    sample, = query_node_samples([_watchdog(tmp_path)], 0, run=run)
    assert sample.memory_temp_c is sample.max_sm_clock_mhz is None
    assert sample.gpu_temp_c == 60


def test_monitor_polls_off_training_thread_and_closes_before_return(tmp_path, monkeypatch):
    monitor = _monitor(tmp_path, monkeypatch)
    thread_names = []
    monkeypatch.setattr(background, "INTERVAL_SECONDS", .01)

    def query(watchdogs, step, **kwargs):
        thread_names.append(threading.current_thread().name)
        return [_sample(watchdog, step) for watchdog in watchdogs]

    monkeypatch.setattr(background, "query_node_samples", query)
    monitor.update_step(123)
    monitor.start()
    monitor.start()
    try:
        _await(lambda: len(thread_names) >= 2)
        assert monitor.failures() == []
    finally:
        monitor.close()
    assert monitor._thread is not None and not monitor._thread.is_alive()
    assert set(thread_names) == {"koochak-gpu-health"}
    rows = [json.loads(line) for line in (tmp_path / "gpu_health/gpu_health_rank0.jsonl").read_text().splitlines()]
    assert rows[0]["step"] == 123
    assert [row["poll_sequence"] for row in rows] == list(range(1, len(rows) + 1))
    assert all(row["poll_monotonic_seconds"] > 0 for row in rows)


def test_query_exception_becomes_a_stop_request(tmp_path, monkeypatch):
    monitor = _monitor(tmp_path, monkeypatch)

    def query(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(background, "query_node_samples", query)
    monitor.start()
    try:
        _await(lambda: bool(monitor.failures()))
        assert monitor.failures()[0]["reasons"] == ["monitor_query_failed"]
    finally:
        monitor.close()


@pytest.mark.parametrize("state", ["dead", "stale"])
def test_dead_and_stale_monitor_fail_closed(tmp_path, monkeypatch, state):
    monitor = _monitor(tmp_path, monkeypatch)
    monitor._leader = True
    monitor._thread = SimpleNamespace(is_alive=lambda: state == "stale")
    monitor._last_sample_at = time.monotonic() - background.MAX_SAMPLE_AGE_SECONDS - 1
    assert monitor.failures()[0]["reasons"] == [f"monitor_{'died' if state == 'dead' else state}"]


def test_mapping_failure_is_reported_without_starting_a_query(tmp_path, monkeypatch):
    monitor = _monitor(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid=None))
    monitor.start()
    assert monitor._thread is None
    assert monitor.failures()[0]["reasons"] == ["monitor_mapping_failed"]
    monitor.close()


@pytest.mark.parametrize("value", [
    "a557a652-f56f-5884-75a6-5e0cf15a769e",
    "GPU-a557a652-f56f-5884-75a6-5e0cf15a769e",
    b"GPU-a557a652-f56f-5884-75a6-5e0cf15a769e",
])
def test_physical_uuid_normalizes_strings_and_bytes(value):
    assert background.physical_gpu_uuid(value) == "GPU-a557a652-f56f-5884-75a6-5e0cf15a769e"


def test_physical_uuid_accepts_torch_cuuuid_wrapper():
    class CUuuid:
        def __str__(self):
            return "a557a652-f56f-5884-75a6-5e0cf15a769e"

        @property
        def bytes(self):
            pytest.fail("CUDA's string form suffices; do not access GPU state here")

    assert background.physical_gpu_uuid(CUuuid()) == "GPU-a557a652-f56f-5884-75a6-5e0cf15a769e"


def test_close_reaps_an_inflight_query():
    query = NodeQuery()
    errors = []

    def run():
        try:
            query.run([sys.executable, "-c", "import time; time.sleep(60)"])
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    _await(lambda: query._process is not None)
    process = query._process
    query.cancel()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert process.poll() is not None
    assert query._process is None
    with pytest.raises(RuntimeError, match="stopped"):
        query.run([sys.executable, "-c", "pass"])
