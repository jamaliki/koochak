"""One CPU-only GPU monitor per node, with completed-update stop flags."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import torch

from .gpu import GpuHealthWatchdog, current_local_rank
from .node_query import NodeQuery, QUERY_TIMEOUT_SECONDS, query_node_samples

INTERVAL_SECONDS = 30.0
MAX_SAMPLE_AGE_SECONDS = 90.0
JOIN_TIMEOUT_SECONDS = 2 * QUERY_TIMEOUT_SECONDS + 5.0


def physical_gpu_uuid(value: Any) -> str:
    """Normalize CUDA's UUID wrapper or string to nvidia-smi's physical ID."""
    text = value.decode("ascii") if isinstance(value, bytes) else str(value)
    return "GPU-" + str(uuid.UUID(text.removeprefix("GPU-")))


def device_uuid(device: torch.device) -> Any:
    properties = torch.cuda.get_device_properties(device)
    try:
        return properties.uuid
    except AttributeError:
        pass
    # Older Torch: CUDA is already initialized; resolve its visible ordinal.
    import ctypes

    driver = ctypes.CDLL("libcuda.so.1")
    index = torch.cuda.current_device() if device.index is None else device.index
    handle = ctypes.c_int()
    value = (ctypes.c_ubyte * 16)()
    for name, argtypes, args in (
        ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
         (ctypes.byref(handle), index)),
        ("cuDeviceGetUuid_v2", [ctypes.POINTER(type(value)), ctypes.c_int],
         (ctypes.byref(value), handle)),
    ):
        function = getattr(driver, name)
        function.argtypes, function.restype = argtypes, ctypes.c_int
        result = function(*args)
        if result != 0:
            raise RuntimeError(f"{name} failed with CUDA driver error {result}")
    return str(uuid.UUID(bytes=bytes(value)))


class BackgroundGpuHealth(GpuHealthWatchdog):
    """Keep GPU queries off training threads and publish local fault details."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.local_rank = current_local_rank()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._failures: list[dict[str, Any]] = []
        self._watchdogs: list[GpuHealthWatchdog] = []
        self._last_sample_at = time.monotonic()
        self._step = -1
        self._started = False
        self._leader = False
        self._closed = False
        self._sequence = 0
        self._strict_logging = True
        self._query = NodeQuery()

    def _metadata(self) -> dict[str, Any]:
        mapping_error = None
        if self.enabled:
            try:
                self.gpu_query_id = physical_gpu_uuid(device_uuid(self.device))
            except Exception as exc:
                mapping_error = f"{type(exc).__name__}: {exc}"
        return {
            "enabled": self.enabled, "rank": self.rank,
            "hostname": self.hostname, "slurm_node": self.slurm_node,
            "cuda_device": self.cuda_device, "query_id": self.gpu_query_id,
            "local_rank": self.local_rank,
            "mapping_error": mapping_error,
        }

    def start(self) -> None:
        """Elect one leader per hostname, then start its monitor once."""
        if self._started:
            return
        self._started = True
        metadata = self._metadata()
        peers = [metadata]
        if self.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            peers = [None] * self.world_size
            torch.distributed.all_gather_object(peers, metadata)
        owned = [item for item in peers if item["enabled"]
                 and item["hostname"] == self.hostname]
        if not owned or self.rank != min(item["rank"] for item in owned):
            return
        self._leader = True
        try:
            self._start_node(owned)
        except Exception as exc:
            self._thread = None
            self._record_error("monitor_start_failed", str(exc))

    def _start_node(self, owned: list[dict[str, Any]]) -> None:
        errors = [item["mapping_error"] for item in owned if item["mapping_error"]]
        if errors:
            self._record_error("monitor_mapping_failed", "; ".join(errors))
            return
        ids = [item["query_id"] for item in owned]
        if len(set(ids)) != len(ids):
            self._record_error("monitor_mapping_failed", "Several ranks name the same GPU")
            return
        for item in owned:
            watchdog = GpuHealthWatchdog(
                device=torch.device("cuda", item["cuda_device"]), out_dir=self.out_dir,
                rank=item["rank"], world_size=self.world_size,
            )
            watchdog.gpu_query_id = item["query_id"]
            watchdog.local_rank = item["local_rank"]
            watchdog._strict_logging = True
            self._watchdogs.append(watchdog)
        self._thread = threading.Thread(
            target=self._poll, name="koochak-gpu-health", daemon=True,
        )
        self._last_sample_at = time.monotonic()
        self._thread.start()

    def update_step(self, step: int) -> None:
        self._step = int(step)

    def failures(self) -> list[dict[str, Any]]:
        """Read local state without GPU queries, files, or distributed calls."""
        if not self._leader:
            return []
        with self._lock:
            failed = bool(self._failures)
            last_sample_at = self._last_sample_at
        if not failed and not self._closed:
            if self._thread is None or not self._thread.is_alive():
                self._record_error("monitor_died", "GPU health monitor stopped")
            elif time.monotonic() - last_sample_at > MAX_SAMPLE_AGE_SECONDS:
                self._record_error("monitor_stale", "GPU health monitor sample is stale")
        with self._lock:
            return list(self._failures)

    def close(self) -> None:
        """Stop polling and wait for any bounded query and log writes."""
        if self._closed:
            return
        self._stop.set()
        try:
            self._query.cancel()
        except Exception as exc:
            self._record_error("monitor_teardown_failed", f"{type(exc).__name__}: {exc}")
        if self._thread is not None:
            self._thread.join(timeout=JOIN_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                self._record_error("monitor_teardown_failed", "GPU health monitor did not stop")
        self._closed = True

    def _poll(self) -> None:
        try:
            while not self._stop.is_set():
                samples = query_node_samples(self._watchdogs, self._step, run=self._query.run)
                self._sequence += 1
                failures = []
                for watchdog, sample in zip(self._watchdogs, samples):
                    sample.poll_sequence = self._sequence
                    sample.poll_monotonic_seconds = time.monotonic()
                    failure = watchdog.observe_sample(sample)
                    if failure is not None:
                        failures.append(failure.to_dict())
                with self._lock:
                    self._last_sample_at = time.monotonic()
                    self._failures.extend(failures)
                    failed = bool(self._failures)
                if failed:
                    return
                self._stop.wait(INTERVAL_SECONDS)
        except Exception as exc:
            if not self._stop.is_set():
                self._record_error("monitor_query_failed", f"{type(exc).__name__}: {exc}")

    def _record_error(self, reason: str, message: str) -> None:
        payload = {
            "event": "gpu_health_monitor_failure", "rank": self.rank,
            "local_rank": self.local_rank, "hostname": self.hostname,
            "slurm_node": self.slurm_node, "step": self._step,
            "timestamp_unix": time.time(), "reasons": [reason], "message": message,
            "slurm_action": self.slurm_action,
        }
        with self._lock:
            if any(reason in failure["reasons"] for failure in self._failures):
                return
            self._failures.append(payload)
        try:
            self._append_jsonl("gpu_health_failures", payload)
        except OSError as exc:
            print(f"[gpu-health] could not write monitor failure: {exc}", flush=True)
