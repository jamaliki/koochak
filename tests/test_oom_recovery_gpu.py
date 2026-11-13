"""Real CUDA OOMs through the training loop, each in fresh processes.

Every test runs ``oom_gpu_trainer.py`` as a subprocess, under
``torch.distributed.run`` for DDP, so CUDA contexts, exit codes and NCCL
teardown are real.  Each test prints one ``[oom-gpu]`` JSON summary line;
run with ``-s`` to keep them in a cluster log.

``KOOCHAK_OOM_TEST_DEVICE=cpu`` runs the same flows with simulated OOMs and
Gloo, which checks the harness without a GPU.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pytest
import torch

from koochak import OOM_RESTART_EXIT_CODE
from koochak import oom as oom_lib
from koochak.storage import checkpoint
from koochak.storage.store import open_store

TRAINER = Path(__file__).with_name("oom_gpu_trainer.py")
ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("KOOCHAK_OOM_TEST_DEVICE", "cuda")
_MIB = 1 << 20

requires_device = pytest.mark.skipif(
    DEVICE == "cuda" and not torch.cuda.is_available(), reason="requires CUDA"
)
requires_two_devices = pytest.mark.skipif(
    DEVICE == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < 2),
    reason="requires two CUDA devices",
)


def _launch(
    out_dir: Path,
    result: Path,
    *trainer_args: str,
    nproc: int = 0,
    max_restarts: int = 0,
    timeout: float = 900.0,
) -> Tuple[subprocess.CompletedProcess, float, float]:
    """Run the trainer; return the process, its wall time and its end time."""

    command: List[str] = [sys.executable]
    if nproc:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            f"--max-restarts={max_restarts}",
        ]
        if sys.platform == "darwin":
            command.append("--local-addr=127.0.0.1")  # macOS may not resolve its own hostname

    command += [
        str(TRAINER),
        "--out-dir",
        str(out_dir),
        "--result",
        str(result),
        "--device",
        DEVICE,
        *trainer_args,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH")) if p)
    started = time.monotonic()
    proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
    ended = time.time()
    sys.stdout.write(proc.stdout[-20000:])
    sys.stderr.write(proc.stderr[-20000:])
    return proc, time.monotonic() - started, ended


def _result(template: Path, rank: int) -> Dict[str, Any]:
    return json.loads(Path(str(template).format(rank=rank)).read_text())


def _records(out_dir: Path) -> List[str]:
    return sorted(open_store(str(out_dir)).list(oom_lib.RECORD_PREFIX))


def _report(name: str, **fields: Any) -> None:
    print("[oom-gpu] " + json.dumps({"test": name, "device": DEVICE, **fields}, sort_keys=True), flush=True)


def _assert_memory_was_held_then_released(result: Mapping[str, Any], step: int) -> Dict[str, Any]:
    """The failed step held most of the GPU, and the next step started without it."""

    [event] = result["oom"]
    at_oom = event["memory"]
    by_step = {m["step"]: m for m in result["memory"]}
    before, after = by_step[step], by_step[step + 1]
    summary = {
        "total_gib": at_oom["total_bytes"] / (1 << 30),
        "allocated_at_oom_gib": at_oom["allocated_bytes"] / (1 << 30),
        "allocated_before_mib": before["allocated"] / _MIB,
        "allocated_after_mib": after["allocated"] / _MIB,
        "free_after_gib": after["free"] / (1 << 30),
    }
    assert at_oom["allocated_bytes"] > 0.8 * at_oom["total_bytes"], summary
    assert after["allocated"] <= before["allocated"] + 64 * _MIB, summary
    assert after["free"] > 0.8 * after["total"], summary
    return summary


# ------------------------------------------------------------------ one process


@requires_device
@pytest.mark.parametrize(("phase", "loop_phase"), [("forward", "step_fn"), ("backward", "backward")])
def test_oom_holding_the_gpu_is_skipped_in_place_and_released(
    tmp_path: Path, phase: str, loop_phase: str
) -> None:
    result_path = tmp_path / "result-{rank}.json"
    proc, elapsed, _ = _launch(tmp_path / "run", result_path, "--max-steps", "6", "--oom", f"2:0:{phase}")

    assert proc.returncode == 0, proc.stderr[-4000:]
    result = _result(result_path, 0)
    assert result["seen"] == [0, 1, 2, 3, 4, 5]
    assert result["oom_skipped_steps"] == [2]
    [event] = result["oom"]
    assert (event["phase"], event["action"], event["error_type"]) == (loop_phase, "skip", "OutOfMemoryError")
    memory = _assert_memory_was_held_then_released(result, 2) if DEVICE == "cuda" else {}
    _report(f"skip_in_place[{phase}]", elapsed_s=round(elapsed, 1), message=event["message"][:200], **memory)


@requires_device
def test_consecutive_real_ooms_restart_in_a_fresh_process(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    args = ("--max-steps", "6", "--max-consecutive", "2", "--oom", "2:0:forward", "--oom", "3:0:backward")
    first, first_s, _ = _launch(out_dir, tmp_path / "first-{rank}.json", *args)

    assert first.returncode == OOM_RESTART_EXIT_CODE, first.stderr[-4000:]
    result = _result(tmp_path / "first-{rank}.json", 0)
    assert [e["action"] for e in result["oom"]] == [oom_lib.SKIP, oom_lib.RESTART]
    path, saved = checkpoint.resolve_auto_resume(str(out_dir))
    assert path.endswith("step000000004.pt")
    assert (saved["next_step"], saved["oom_skipped_steps"]) == (4, [2, 3])

    second, second_s, _ = _launch(out_dir, tmp_path / "second-{rank}.json", *args)
    assert second.returncode == 0, second.stderr[-4000:]
    resumed = _result(tmp_path / "second-{rank}.json", 0)
    assert (resumed["start"], resumed["seen"], resumed["oom"]) == (4, [4, 5], [])
    assert resumed["oom_skipped_steps"] == [2, 3]
    _report(
        "consecutive_restart",
        first_exit=first.returncode,
        first_s=round(first_s, 1),
        second_s=round(second_s, 1),
    )


# ------------------------------------------------------------------------- DDP


def _torchrun_exitcodes(stderr: str) -> List[int]:
    return [int(code) for code in re.findall(r"exitcode\s*:\s*(-?\d+)", stderr)]


@requires_two_devices
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_ddp_oom_on_one_rank_tears_down_then_resume_skips_the_step(tmp_path: Path, phase: str) -> None:
    out_dir = tmp_path / "run"
    args = ("--max-steps", "6", "--ckpt-every", "2", "--oom", f"3:1:{phase}")
    first, first_s, ended = _launch(out_dir, tmp_path / "first-{rank}.json", *args, nproc=2)

    failed = _result(tmp_path / "first-{rank}.json", 1)
    teardown_s = ended - failed["exit_time"]
    reported = _torchrun_exitcodes(first.stderr)
    _report(
        f"ddp_teardown[{phase}]",
        torchrun_returncode=first.returncode,
        torchrun_reported_exitcodes=reported,
        rank1_exit=failed["exit"],
        teardown_s=round(teardown_s, 1),
        first_s=round(first_s, 1),
    )
    assert failed["exit"] == OOM_RESTART_EXIT_CODE
    assert failed["oom"][0]["phase"] == ("step_fn" if phase == "forward" else "backward")
    assert first.returncode != 0
    assert teardown_s < 120, f"peers took {teardown_s:.0f}s to tear down"
    assert _records(out_dir) == ["oom/step000000003-rank00001.json"]
    # The step-2 checkpoint was published before rank 1 could reach step 3.
    selection = checkpoint.resolve_auto_resume(str(out_dir))
    assert selection is not None
    next_step = selection[1]["next_step"]
    assert next_step == 3

    second, second_s, _ = _launch(out_dir, tmp_path / "second-{rank}.json", *args, nproc=2)
    assert second.returncode == 0, second.stderr[-4000:]
    results = [_result(tmp_path / "second-{rank}.json", rank) for rank in (0, 1)]
    for result in results:
        assert result["seen"] == [s for s in range(next_step, 6) if s != 3]
        assert result["oom_skipped_steps"] == [3]
        assert result["oom"] == []
    assert results[0]["weight_digest"] == results[1]["weight_digest"]
    _report(f"ddp_resume[{phase}]", resumed_from=next_step, second_s=round(second_s, 1))


@requires_two_devices
def test_torchrun_max_restarts_recovers_within_one_launch(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    result_path = tmp_path / "result-{rank}.json"
    proc, elapsed, _ = _launch(
        out_dir,
        result_path,
        "--max-steps", "6", "--ckpt-every", "2", "--oom", "3:1:backward",
        nproc=2,
        max_restarts=1,
    )

    assert proc.returncode == 0, proc.stderr[-4000:]
    results = [_result(result_path, rank) for rank in (0, 1)]
    for result in results:
        assert result["restart_count"] == 1
        assert result["exit"] == 0
        assert result["oom_skipped_steps"] == [3]
        assert 3 not in result["seen"] and result["seen"][-2:] == [4, 5]
    assert results[0]["weight_digest"] == results[1]["weight_digest"]
    _report("torchrun_max_restarts", elapsed_s=round(elapsed, 1), resumed_from=results[0]["start"])
