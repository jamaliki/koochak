from __future__ import annotations

import gc
import sys

__all__ = ["WARM_UPDATES", "WarmFreeze", "compiled_graphs"]

# The first two updates of a compiled model build its graphs and their dynamic-shape variants.
WARM_UPDATES = 2


def compiled_graphs() -> int:
    """Count the graphs Dynamo has compiled in this process, without importing it."""
    utils = sys.modules.get("torch._dynamo.utils")
    return 0 if utils is None else int(utils.counters["stats"]["unique_graphs"])


class WarmFreeze:
    """Freeze the objects alive after warm-up, and again after any later update that compiled.

    A full cyclic collection traverses every tracked object, including the million or so that
    compilation leaves behind, so each one paused a rank for about a second. Ranks reach CPython's
    threshold at different updates, and DDP makes the others wait. Frozen objects drop out of every
    later collection. Reference counting still frees them; only a cycle among them is never
    collected. Counting this process's updates freezes a resumed run after its own warm-up. The
    objects stay frozen after training, so later stages in the process do not traverse them either.
    """

    def __init__(self) -> None:
        self.updates = 0
        self.graphs: int | None = None

    def update(self) -> None:
        self.updates += 1
        if self.updates < WARM_UPDATES:
            return
        graphs = compiled_graphs()
        if graphs != self.graphs:
            gc.collect()
            gc.freeze()
            self.graphs = graphs
