"""Async-safe evacuation signalling for long-running training jobs."""

from __future__ import annotations

import signal
import threading

import torch

from .core import dist as dist_lib

__all__ = [
    "EVACUATION_EXIT_CODE",
    "EvacuationController",
    "install_evacuation_handler",
]


EVACUATION_EXIT_CODE = 75


def _signal_number(name: str | int) -> int:
    if isinstance(name, int):
        number = name
    else:
        normalized = str(name).upper()
        if not normalized.startswith("SIG"):
            normalized = "SIG" + normalized
        number = getattr(signal, normalized, None)
        if not isinstance(number, int):
            raise ValueError(f"unsupported evacuation signal: {name!r}")
    if number != signal.SIGUSR1:
        raise ValueError("evacuation signal must be SIGUSR1")
    return number


class EvacuationController:
    """Own an opt-in, side-effect-free evacuation signal handler.

    The Python signal handler only assigns a boolean.  Collective
    reconciliation is deliberately explicit and happens at a training safe
    boundary, never from the handler.
    """

    def __init__(
        self,
        enabled: bool = False,
        *,
        signal_name: str | int = "USR1",
        exit_code: int = EVACUATION_EXIT_CODE,
    ) -> None:
        self.enabled = bool(enabled)
        self.signal_number = _signal_number(signal_name)
        if int(exit_code) != EVACUATION_EXIT_CODE:
            raise ValueError(f"evacuation exit code is reserved: {EVACUATION_EXIT_CODE}")
        self.exit_code = EVACUATION_EXIT_CODE
        self._requested = False
        self._previous_handler: object = signal.SIG_DFL
        self._installed = False

    @property
    def requested(self) -> bool:
        return self._requested

    def request(self) -> None:
        """Request evacuation programmatically at the next safe boundary."""

        self._requested = True

    def _handle_signal(self, _signum: int, _frame: object) -> None:
        # Keep this handler async-signal-safe: no I/O, locks, allocations, or
        # distributed calls.  A boolean assignment is the only operation.
        self._requested = True

    def install(self) -> "EvacuationController":
        if not self.enabled or self._installed:
            return self
        # signal.signal itself is only legal in the process' main thread.  A
        # worker-thread caller must opt in from its owner/main thread instead
        # of silently running without evacuation protection.
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("evacuation signal handler must be installed from the main thread")
        self._previous_handler = signal.getsignal(self.signal_number)
        signal.signal(self.signal_number, self._handle_signal)
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        if threading.current_thread() is threading.main_thread():
            signal.signal(self.signal_number, self._previous_handler)
        self._installed = False

    def reconcile(self) -> bool:
        """Reconcile local requests across DDP ranks using a MAX reduction."""

        requested = bool(self._requested)
        if (
            dist_lib.is_initialized()
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and dist_lib.world_size() > 1
        ):
            backend = str(torch.distributed.get_backend()).lower()
            device = "cuda" if backend == "nccl" else "cpu"
            value = torch.tensor(
                [int(requested)],
                dtype=torch.int64,
                device=device,
            )
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
            requested = bool(int(value.item()))
        self._requested = requested
        return requested


def install_evacuation_handler(
    enabled: bool = True,
    *,
    signal_name: str | int = "USR1",
) -> EvacuationController:
    """Construct and install an explicit evacuation controller."""

    return EvacuationController(enabled, signal_name=signal_name).install()
