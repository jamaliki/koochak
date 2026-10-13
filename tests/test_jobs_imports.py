"""Keep isolated worker imports independent of launcher packages."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_worker_modules_import_without_site_packages():
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import koochak.jobs.runner
import koochak.jobs.isolated_exec
assert 'omegaconf' not in sys.modules
assert 'yaml' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", code, str(root)], check=True)


def test_public_launcher_exports():
    import koochak.jobs as jobs

    assert set(jobs.__all__) <= set(dir(jobs))
    for name in jobs.__all__:
        assert getattr(jobs, name) is getattr(jobs, name)
    with pytest.raises(AttributeError, match="unknown"):
        getattr(jobs, "unknown")
