"""Submission hosts need config tools, not the training runtime."""

from pathlib import Path
import subprocess
import sys


def test_jobs_import_without_torch():
    script = """
import importlib.abc
import sys

class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ImportError('Submission imported Torch')

sys.meta_path.insert(0, RejectTorch())
from koochak.jobs import ConfigPatch, prepare_run, submit_pazuzu
from koochak import EvacuationController
controller = EvacuationController(enabled=False).install()
controller.uninstall()
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
