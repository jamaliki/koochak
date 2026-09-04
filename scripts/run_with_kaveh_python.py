#!/usr/bin/env python3
"""Execute a script with the scientific kaveh Python environment."""

from __future__ import annotations

import os
import sys

PYTHON = "/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/kaveh/bin/python"


if __name__ == "__main__":
    os.execv(PYTHON, [PYTHON, *sys.argv[1:]])
