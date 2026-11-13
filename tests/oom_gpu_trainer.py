"""Trainer for ``test_oom_recovery_gpu.py``: real CUDA OOMs in a fresh process.

Run it directly for one process, or under ``python -m torch.distributed.run``
for DDP.  Each ``--oom STEP:RANK:PHASE`` makes that rank fail that step while
the step holds nearly all of the GPU's memory:

* ``forward``: ``step_fn`` keeps allocating 1 GiB blocks until the caching
  allocator fails, so the failed frame pins the whole GPU.
* ``backward``: the first backward node fills the GPU and keeps the fill on the
  graph, so the next native backward kernel fails to allocate its output.

With ``--device cpu`` the same places raise ``torch.OutOfMemoryError``
directly, so the harness and its exit codes can be checked without a GPU.

Each rank writes ``--result`` (with ``{rank}`` substituted) as it returns or
exits, including on the reserved OOM restart exit.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import torch

from koochak.core import dist as dist_lib
from koochak.core import launch
from koochak.loop import training_loop
from koochak.storage import checkpoint

_GIB = 1 << 30
_FILL_SIZES = (_GIB, 1 << 26, 1 << 22)  # 1 GiB, 64 MiB, 4 MiB


def _oom_spec(text: str) -> Tuple[int, int, str]:
    step, rank, phase = text.split(":")
    if phase not in ("forward", "backward"):
        raise argparse.ArgumentTypeError(f"phase must be forward or backward, got {phase!r}")
    return int(step), int(rank), phase


class _Indices:
    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for index in range(self.start, self.stop):
            yield {"i": torch.tensor(index)}


class _FillInBackward(torch.autograd.Function):
    """Identity whose backward fills the GPU and keeps the fill alive on the graph.

    The fill is a ``ctx`` attribute, not a saved tensor: autograd frees saved
    tensors as soon as their node has run, which would release it before the
    next kernel allocates.
    """

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        return value.clone()

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        if grad.device.type != "cuda":
            raise torch.OutOfMemoryError("simulated backward OOM (CPU harness check)")
        ctx.ballast = _ballast(grad.device)
        return grad


def _oom_in_forward(device: torch.device) -> None:
    if device.type != "cuda":
        raise torch.OutOfMemoryError("simulated forward OOM (CPU harness check)")
    held: List[torch.Tensor] = []
    while True:  # ends with the allocator's OutOfMemoryError
        held.append(torch.empty(_GIB, dtype=torch.uint8, device=device))


def _ballast(device: torch.device) -> List[torch.Tensor]:
    """Allocate until less than 4 MiB of the GPU is left; return what was allocated."""

    held: List[torch.Tensor] = []
    for size in _FILL_SIZES:
        while True:
            try:
                held.append(torch.empty(size, dtype=torch.uint8, device=device))
            except torch.OutOfMemoryError:  # the fill is meant to exhaust the GPU
                break
    return held


def _memory(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "free": int(free),
        "total": int(total),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--result", required=True, help="result JSON path; {rank} is substituted")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--max-consecutive", type=int, default=3)
    parser.add_argument("--oom", type=_oom_spec, action="append", default=[])
    parser.add_argument("--width", type=int, help="hidden width (default 4096 on CUDA, 64 on CPU)")
    parser.add_argument("--batch", type=int, help="rows per step (default 2048 on CUDA, 8 on CPU)")
    args = parser.parse_args(argv)

    distributed = "WORLD_SIZE" in os.environ
    if distributed:
        launch.initialize({"backend": "nccl" if args.device == "cuda" else "gloo"})
    rank = dist_lib.rank()
    world_size = dist_lib.world_size()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if args.device == "cuda"
        else torch.device("cpu")
    )
    width = args.width or (4096 if device.type == "cuda" else 64)
    batch = args.batch or (2048 if device.type == "cuda" else 8)
    ooms = {(step, oom_rank): phase for step, oom_rank, phase in args.oom}

    selection = checkpoint.resolve_auto_resume(args.out_dir)
    start = 0 if selection is None else int(selection[1]["next_step"])
    del selection

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, width)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    seen: List[int] = []
    memory: List[Dict[str, int]] = []
    oom_events: List[Dict[str, Any]] = []

    def step_fn(module: torch.nn.Module, batch_: Mapping[str, torch.Tensor], ctx: Mapping[str, Any]):
        step = int(ctx["step"])
        seen.append(step)
        memory.append({"step": step, **_memory(device)})  # state the previous step left behind
        generator = torch.Generator().manual_seed(int(batch_["i"]) * 1009 + rank)
        x = torch.randn(batch, width, generator=generator).to(device)
        loss = module(x).square().mean()
        phase = ooms.get((step, rank))
        if phase == "forward":
            _oom_in_forward(device)
        elif phase == "backward":
            loss = _FillInBackward.apply(loss)
        return {"loss": loss}

    def on_oom(record: Mapping[str, Any], _ctx: Mapping[str, Any]) -> None:
        oom_events.append({k: v for k, v in record.items() if k != "memory_summary"})

    result: Dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
        "start": start,
        "restart_count": int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")),
        "seen": seen,
        "memory": memory,
        "oom": oom_events,
        "exit": None,
    }
    result_path = Path(args.result.format(rank=rank))
    try:
        final = training_loop(
            model=model,
            dataset=_Indices(start, args.max_steps + 8),
            step_fn=step_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_cfg={
                "ddp": world_size > 1,
                "device": str(device),
                "max_steps": args.max_steps,
                "ckpt_every": args.ckpt_every,
                "log_every": 1000,
                "out_dir": args.out_dir,
                "warn_unsharded": False,
                "oom_recovery": {"enabled": True, "max_consecutive": args.max_consecutive},
            },
            hooks={"on_oom": [on_oom]},
            resume="auto",
        )
        result["exit"] = 0
        result["next_step"] = int(final["next_step"])
        result["oom_skipped_steps"] = final.get("oom_skipped_steps")
        result["weight_digest"] = float(
            sum(t.double().sum().item() for t in final["model"].values())
        )
    except SystemExit as exc:
        result["exit"] = exc.code
        raise
    finally:
        result["exit_time"] = time.time()
        _write_json(result_path, result)
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
