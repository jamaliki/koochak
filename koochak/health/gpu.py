from __future__ import annotations

import csv
import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from ..utils.paths import canonical_dir


WARMUP_STEPS = 20
CHECK_EVERY_STEPS = 10
REQUIRE_UTILIZATION_PCT = 80.0
CONSECUTIVE_FAILURES = 2
MAX_GPU_TEMP_C = 86.0
MAX_MEMORY_TEMP_C = 94.0
MIN_SM_CLOCK_FRAC = 0.70
EXIT_CODE = 42


PRIMARY_QUERY_FIELDS = [
    "index",
    "uuid",
    "name",
    "temperature.gpu",
    "temperature.memory",
    "pstate",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
    "clocks.max.sm",
    "utilization.gpu",
    "utilization.memory",
    "clocks_throttle_reasons.active",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown",
    "clocks_throttle_reasons.sw_power_cap",
]

FALLBACK_QUERY_FIELDS = [
    "index",
    "uuid",
    "name",
    "temperature.gpu",
    "pstate",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
    "utilization.gpu",
    "utilization.memory",
    "clocks_throttle_reasons.active",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown",
    "clocks_throttle_reasons.sw_power_cap",
]


def _env_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text.split()[0])
    except ValueError:
        return None


def _is_active(value: Any) -> bool:
    return str(value).strip().lower() == "active"


def _normalize_node_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return text.split(".")[0]


def current_slurm_node() -> str:
    return (
        _normalize_node_name(os.environ.get("SLURMD_NODENAME"))
        or _normalize_node_name(os.environ.get("HOSTNAME"))
        or _normalize_node_name(socket.gethostname())
        or "unknown"
    )


def current_local_rank() -> Optional[int]:
    for key in (
        "LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
    ):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            return None
    return None


def query_id_for_cuda_device(cuda_device: Optional[int]) -> str:
    if cuda_device is None:
        return "0"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        parts = [part.strip() for part in visible.split(",") if part.strip()]
        if 0 <= int(cuda_device) < len(parts):
            return parts[int(cuda_device)]
    return str(int(cuda_device))


@dataclass
class GpuHealthSample:
    step: int
    rank: int
    local_rank: Optional[int]
    world_size: int
    hostname: str
    slurm_node: str
    cuda_device: Optional[int]
    gpu_query_id: str
    gpu_index: Optional[str]
    gpu_uuid: Optional[str]
    gpu_name: Optional[str]
    gpu_temp_c: Optional[float]
    memory_temp_c: Optional[float]
    pstate: Optional[str]
    power_draw_w: Optional[float]
    power_limit_w: Optional[float]
    sm_clock_mhz: Optional[float]
    mem_clock_mhz: Optional[float]
    max_sm_clock_mhz: Optional[float]
    gpu_util_pct: Optional[float]
    mem_util_pct: Optional[float]
    throttle_active_mask: Optional[str]
    sw_thermal_slowdown: bool
    hw_slowdown: bool
    hw_thermal_slowdown: bool
    hw_power_brake_slowdown: bool
    sw_power_cap: bool
    timestamp_unix: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GpuHealthFailure:
    sample: GpuHealthSample
    reasons: List[str]
    consecutive_failures: int
    slurm_action: str

    def to_dict(self) -> Dict[str, Any]:
        payload = self.sample.to_dict()
        payload.update(
            {
                "event": "gpu_health_failure",
                "reasons": list(self.reasons),
                "consecutive_failures": int(self.consecutive_failures),
                "slurm_action": self.slurm_action,
            }
        )
        return payload


def sample_from_nvidia_smi_row(
    *,
    step: int,
    rank: int,
    world_size: int,
    hostname: str,
    slurm_node: str,
    cuda_device: Optional[int],
    gpu_query_id: str,
    fields: Sequence[str],
    row: Sequence[str],
) -> GpuHealthSample:
    values = {field: (row[index].strip() if index < len(row) else "N/A") for index, field in enumerate(fields)}
    return GpuHealthSample(
        step=int(step),
        rank=int(rank),
        local_rank=current_local_rank(),
        world_size=int(world_size),
        hostname=hostname,
        slurm_node=slurm_node,
        cuda_device=cuda_device,
        gpu_query_id=gpu_query_id,
        gpu_index=values.get("index"),
        gpu_uuid=values.get("uuid"),
        gpu_name=values.get("name"),
        gpu_temp_c=_parse_float(values.get("temperature.gpu")),
        memory_temp_c=_parse_float(values.get("temperature.memory")),
        pstate=values.get("pstate"),
        power_draw_w=_parse_float(values.get("power.draw")),
        power_limit_w=_parse_float(values.get("power.limit")),
        sm_clock_mhz=_parse_float(values.get("clocks.sm")),
        mem_clock_mhz=_parse_float(values.get("clocks.mem")),
        max_sm_clock_mhz=_parse_float(values.get("clocks.max.sm")),
        gpu_util_pct=_parse_float(values.get("utilization.gpu")),
        mem_util_pct=_parse_float(values.get("utilization.memory")),
        throttle_active_mask=values.get("clocks_throttle_reasons.active"),
        sw_thermal_slowdown=_is_active(values.get("clocks_throttle_reasons.sw_thermal_slowdown")),
        hw_slowdown=_is_active(values.get("clocks_throttle_reasons.hw_slowdown")),
        hw_thermal_slowdown=_is_active(values.get("clocks_throttle_reasons.hw_thermal_slowdown")),
        hw_power_brake_slowdown=_is_active(values.get("clocks_throttle_reasons.hw_power_brake_slowdown")),
        sw_power_cap=_is_active(values.get("clocks_throttle_reasons.sw_power_cap")),
        timestamp_unix=time.time(),
    )


def parse_nvidia_smi_csv(
    *,
    stdout: str,
    fields: Sequence[str],
    step: int,
    rank: int,
    world_size: int,
    hostname: str,
    slurm_node: str,
    cuda_device: Optional[int],
    gpu_query_id: str,
) -> Optional[GpuHealthSample]:
    text = stdout.strip()
    if not text:
        return None
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return None
    return sample_from_nvidia_smi_row(
        step=step,
        rank=rank,
        world_size=world_size,
        hostname=hostname,
        slurm_node=slurm_node,
        cuda_device=cuda_device,
        gpu_query_id=gpu_query_id,
        fields=fields,
        row=rows[0],
    )


def evaluate_sample(sample: GpuHealthSample) -> List[str]:
    util = sample.gpu_util_pct
    busy = util is not None and util >= REQUIRE_UTILIZATION_PCT
    if not busy:
        return []

    reasons: List[str] = []
    if sample.sw_thermal_slowdown:
        reasons.append("sw_thermal_slowdown")
    if sample.hw_slowdown:
        reasons.append("hw_slowdown")
    if sample.hw_thermal_slowdown:
        reasons.append("hw_thermal_slowdown")
    if sample.hw_power_brake_slowdown:
        reasons.append("hw_power_brake_slowdown")
    if sample.gpu_temp_c is not None and sample.gpu_temp_c >= MAX_GPU_TEMP_C:
        reasons.append("gpu_temp_high")
    if sample.memory_temp_c is not None and sample.memory_temp_c >= MAX_MEMORY_TEMP_C:
        reasons.append("memory_temp_high")
    if (
        sample.sm_clock_mhz is not None
        and sample.max_sm_clock_mhz is not None
        and sample.max_sm_clock_mhz > 0
        and sample.sm_clock_mhz / sample.max_sm_clock_mhz < MIN_SM_CLOCK_FRAC
    ):
        reasons.append("sm_clock_low")
    return reasons


class GpuHealthWatchdog:
    def __init__(self, *, device: torch.device, out_dir: str, rank: int, world_size: int) -> None:
        self.device = device
        self.out_dir = canonical_dir(out_dir)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.hostname = socket.gethostname()
        self.slurm_node = current_slurm_node()
        self.cuda_device = int(device.index) if device.type == "cuda" and device.index is not None else None
        self.gpu_query_id = query_id_for_cuda_device(self.cuda_device)
        self.nvidia_smi = shutil.which("nvidia-smi")
        self.enabled = device.type == "cuda" and not _env_enabled("KOOCHAK_GPU_HEALTH_DISABLE")
        self.slurm_disabled = _env_enabled("KOOCHAK_GPU_HEALTH_SLURM_DISABLE")
        self.requested_slurm_action = os.environ.get("KOOCHAK_GPU_HEALTH_SLURM_ACTION", "exit").strip().lower()
        if self.requested_slurm_action not in {"requeue", "cancel", "exit"}:
            self.requested_slurm_action = "exit"
        self.slurm_mutation_blocked_reason = self._slurm_mutation_blocked_reason()
        self.slurm_action = "exit" if self.slurm_mutation_blocked_reason else self.requested_slurm_action
        self.exit_code = EXIT_CODE
        self._consecutive_failures = 0
        self._notice_written = False

    def should_check_step(self, step: int) -> bool:
        return self.enabled and step >= WARMUP_STEPS and step % CHECK_EVERY_STEPS == 0

    def check_local(self, step: int) -> Optional[GpuHealthFailure]:
        sample = self._query_sample(step)
        if sample is None:
            self._consecutive_failures = 0
            return None

        self._append_jsonl("gpu_health", sample.to_dict())
        reasons = evaluate_sample(sample)
        if not reasons:
            self._consecutive_failures = 0
            return None

        self._consecutive_failures += 1
        if self._consecutive_failures < CONSECUTIVE_FAILURES:
            return None

        failure = GpuHealthFailure(
            sample=sample,
            reasons=reasons,
            consecutive_failures=self._consecutive_failures,
            slurm_action=self.slurm_action,
        )
        self._append_jsonl("gpu_health_failures", failure.to_dict())
        return failure

    def gather_failures(self, local_failure: Optional[GpuHealthFailure]) -> List[Dict[str, Any]]:
        payload = local_failure.to_dict() if local_failure is not None else None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered, payload)
            return [item for item in gathered if item is not None]
        return [payload] if payload is not None else []

    def bad_nodes_from_failures(self, failures: Sequence[Dict[str, Any]]) -> List[str]:
        nodes = []
        for item in failures:
            node = item.get("slurm_node")
            if not node or str(node).lower() == "unknown":
                node = item.get("hostname")
            if node:
                nodes.append(str(node))
        nodes = sorted(set(nodes))
        return self._normalize_against_slurm_nodelist(nodes)

    def write_failure_summary(
        self,
        *,
        step: int,
        failures: Sequence[Dict[str, Any]],
        checkpoint_path: Optional[str],
        slurm_results: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        bad_nodes = self.bad_nodes_from_failures(failures)
        summary = {
            "event": "gpu_health_shutdown",
            "step": int(step),
            "timestamp_unix": time.time(),
            "checkpoint_path": checkpoint_path,
            "failures": list(failures),
            "bad_nodes": bad_nodes,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_action": self.slurm_action,
            "requested_slurm_action": self.requested_slurm_action,
            "slurm_mutation_blocked_reason": self.slurm_mutation_blocked_reason,
            "slurm_disabled": self.slurm_disabled,
            "slurm_results": list(slurm_results or []),
        }
        path = self._health_dir() / f"gpu_health_failure_step{step:09d}.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return str(path)

    def perform_slurm_recovery(self, failures: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.slurm_disabled or self.slurm_action == "exit":
            return []
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return [{"cmd": "<slurm>", "returncode": None, "stderr": "SLURM_JOB_ID is not set"}]

        bad_nodes = self.bad_nodes_from_failures(failures)
        if not bad_nodes:
            return [{"cmd": "<slurm>", "returncode": None, "stderr": "no bad nodes reported"}]

        results: List[Dict[str, Any]] = []
        bad_nodes_csv = ",".join(bad_nodes)

        if self.slurm_action == "requeue":
            results.append(self._run_command(["scontrol", "update", f"JobId={job_id}", f"ExcNodeList={bad_nodes_csv}"]))
            results.extend(self._exclude_pending_same_name(bad_nodes_csv))
            results.append(self._run_command(["scontrol", "requeue", job_id]))
        elif self.slurm_action == "cancel":
            results.extend(self._exclude_pending_same_name(bad_nodes_csv))
            results.append(self._run_command(["scancel", job_id]))
        return results

    @staticmethod
    def _slurm_mutation_blocked_reason() -> Optional[str]:
        if os.environ.get("SCRUFFY_JOB_ID") or os.environ.get("SCRUFFY_ROOT"):
            return "parent Slurm mutation is forbidden for Scruffy workloads"

        step_id = os.environ.get("SLURM_STEP_ID") or os.environ.get("SLURM_STEPID")
        if step_id and step_id.strip().lower() not in {"batch", "extern"}:
            return f"parent Slurm mutation is forbidden from nested step {step_id}"
        return None

    def _query_sample(self, step: int) -> Optional[GpuHealthSample]:
        if self.nvidia_smi is None:
            self._write_notice("nvidia-smi is not available on PATH")
            return None
        for fields in (PRIMARY_QUERY_FIELDS, FALLBACK_QUERY_FIELDS):
            cmd = [
                self.nvidia_smi,
                "-i",
                self.gpu_query_id,
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10.0,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self._write_notice(f"nvidia-smi query failed: {exc}")
                return None
            if proc.returncode != 0:
                continue
            sample = parse_nvidia_smi_csv(
                stdout=proc.stdout,
                fields=fields,
                step=step,
                rank=self.rank,
                world_size=self.world_size,
                hostname=self.hostname,
                slurm_node=self.slurm_node,
                cuda_device=self.cuda_device,
                gpu_query_id=self.gpu_query_id,
            )
            if sample is not None:
                return sample
        self._write_notice("nvidia-smi did not return a parseable GPU health row")
        return None

    def _exclude_pending_same_name(self, bad_nodes_csv: str) -> List[Dict[str, Any]]:
        user = os.environ.get("USER")
        job_name = os.environ.get("SLURM_JOB_NAME")
        if not user or not job_name:
            return []

        list_cmd = ["squeue", "-h", "-u", user, "-t", "PENDING", "-n", job_name, "-o", "%i"]
        result = self._run_command(list_cmd)
        results = [result]
        if result.get("returncode") != 0:
            return results

        for job_id in str(result.get("stdout", "")).split():
            results.append(self._run_command(["scontrol", "update", f"JobId={job_id}", f"ExcNodeList={bad_nodes_csv}"]))
        return results

    def _normalize_against_slurm_nodelist(self, nodes: Sequence[str]) -> List[str]:
        nodelist = os.environ.get("SLURM_JOB_NODELIST")
        if not nodelist:
            return sorted({_normalize_node_name(node) or node for node in nodes})

        result = self._run_command(["scontrol", "show", "hostnames", nodelist])
        if result.get("returncode") != 0:
            return sorted({_normalize_node_name(node) or node for node in nodes})

        slurm_nodes = [line.strip() for line in str(result.get("stdout", "")).splitlines() if line.strip()]
        by_short = {_normalize_node_name(node): node for node in slurm_nodes}
        normalized = []
        for node in nodes:
            short = _normalize_node_name(node)
            normalized.append(by_short.get(short, short or node))
        return sorted(set(normalized))

    def _run_command(self, cmd: Sequence[str]) -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                list(cmd),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20.0,
            )
            return {
                "cmd": " ".join(cmd),
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        except (OSError, subprocess.SubprocessError) as exc:
            return {"cmd": " ".join(cmd), "returncode": None, "stdout": "", "stderr": str(exc)}

    def _health_dir(self) -> Path:
        path = Path(self.out_dir) / "gpu_health"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _append_jsonl(self, stem: str, payload: Dict[str, Any]) -> None:
        try:
            path = self._health_dir() / f"{stem}_rank{self.rank}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError as exc:
            # Health logging must not crash training; record one notice and move on.
            if stem != "gpu_health_notices" and not self._notice_written:
                self._notice_written = True
                # No filesystem available — just print so the operator can see why we stopped logging.
                print(f"[gpu-health] failed to write {stem} log: {exc}", flush=True)

    def _write_notice(self, message: str) -> None:
        if self._notice_written:
            return
        self._notice_written = True
        self._append_jsonl(
            "gpu_health_notices",
            {
                "event": "gpu_health_notice",
                "rank": self.rank,
                "hostname": self.hostname,
                "slurm_node": self.slurm_node,
                "cuda_device": self.cuda_device,
                "gpu_query_id": self.gpu_query_id,
                "message": message,
                "timestamp_unix": time.time(),
            },
        )


def summarize_failures_for_stdout(failures: Iterable[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in failures:
        reasons = ",".join(str(reason) for reason in item.get("reasons", []))
        parts.append(
            "rank={rank} local_rank={local_rank} host={host} node={node} gpu={gpu} uuid={uuid} "
            "reasons={reasons} temp={temp}C mem={mem}C sm={sm}/{max_sm}MHz util={util}%".format(
                rank=item.get("rank"),
                local_rank=item.get("local_rank"),
                host=item.get("hostname"),
                node=item.get("slurm_node"),
                gpu=item.get("gpu_index"),
                uuid=item.get("gpu_uuid"),
                reasons=reasons,
                temp=item.get("gpu_temp_c"),
                mem=item.get("memory_temp_c"),
                sm=item.get("sm_clock_mhz"),
                max_sm=item.get("max_sm_clock_mhz"),
                util=item.get("gpu_util_pct"),
            )
        )
    return " | ".join(parts)
