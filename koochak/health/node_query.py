"""Query all assigned GPUs once from a CPU-only monitor thread."""

from __future__ import annotations

import csv
import math
import subprocess
import threading
from typing import Callable, Sequence

from .gpu import (
    FALLBACK_QUERY_FIELDS,
    PRIMARY_QUERY_FIELDS,
    GpuHealthSample,
    GpuHealthWatchdog,
    sample_from_nvidia_smi_row,
)

QUERY_TIMEOUT_SECONDS = 10.0


class NodeQuery:
    """Own and reap the monitor's only child process, including on shutdown."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._closing = False

    def run(self, argv: list[str]) -> subprocess.CompletedProcess:
        with self._lock:
            if self._closing:
                raise RuntimeError("GPU health query stopped")
            process = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self._process = process
        try:
            stdout, stderr = process.communicate(timeout=QUERY_TIMEOUT_SECONDS)
            return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        except BaseException:
            process.kill()
            process.communicate()
            raise
        finally:
            with self._lock:
                self._process = None

    def cancel(self) -> None:
        with self._lock:
            self._closing = True
            process = self._process
            if process is not None:
                process.kill()
        if process is not None:
            process.wait(timeout=QUERY_TIMEOUT_SECONDS)


def query_node_samples(
    watchdogs: Sequence[GpuHealthWatchdog], step: int,
    *, run: Callable[[list[str]], subprocess.CompletedProcess],
) -> list[GpuHealthSample]:
    """Require one distinct, parseable row for each assigned physical GPU."""
    executable = watchdogs[0].nvidia_smi
    if executable is None:
        raise RuntimeError("nvidia-smi is not available on PATH")
    ids = [watchdog.gpu_query_id for watchdog in watchdogs]
    error = "nvidia-smi did not return GPU health rows"
    for fields in (PRIMARY_QUERY_FIELDS, FALLBACK_QUERY_FIELDS):
        proc = run(
            [executable, "-i", ",".join(ids), f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
        )
        if proc.returncode != 0:
            error = f"nvidia-smi exited {proc.returncode}: {proc.stderr.strip()}"
            continue
        rows = list(csv.reader(proc.stdout.strip().splitlines()))
        if len(rows) != len(watchdogs) or any(len(row) != len(fields) for row in rows):
            raise RuntimeError("nvidia-smi returned an incomplete GPU health query")
        samples = [_mapped_sample(watchdog, rows, fields, step) for watchdog in watchdogs]
        if len({sample.gpu_uuid for sample in samples}) != len(samples):
            raise RuntimeError("GPU health query mapped several ranks to the same GPU")
        return samples
    raise RuntimeError(error)


def _mapped_sample(watchdog, rows, fields, step) -> GpuHealthSample:
    query_id = watchdog.gpu_query_id
    matches = [row for row in rows if row[0].strip() == query_id
               or (query_id.startswith("GPU-") and row[1].strip().startswith(query_id))]
    if len(matches) != 1:
        raise RuntimeError(f"GPU health query cannot identify assigned GPU {query_id}")
    sample = sample_from_nvidia_smi_row(
        step=step, rank=watchdog.rank, world_size=watchdog.world_size,
        hostname=watchdog.hostname, slurm_node=watchdog.slurm_node,
        cuda_device=watchdog.cuda_device, gpu_query_id=query_id,
        fields=fields, row=matches[0],
    )
    sample.local_rank = watchdog.local_rank
    required = (sample.gpu_temp_c, sample.sm_clock_mhz, sample.gpu_util_pct)
    if any(value is None or not math.isfinite(value) or value < 0 for value in required) or not sample.gpu_uuid:
        raise RuntimeError(f"GPU health query lacks required values for GPU {query_id}")
    optional = (sample.memory_temp_c, sample.power_draw_w, sample.power_limit_w,
                sample.mem_clock_mhz, sample.max_sm_clock_mhz, sample.mem_util_pct)
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in optional):
        raise RuntimeError(f"GPU health query has invalid readings for GPU {query_id}")
    return sample
