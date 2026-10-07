import subprocess
import sys


def test_launch_and_runner_import_without_torch():
    result = subprocess.run([sys.executable, "-c", """
import sys
sys.modules['torch'] = None
from koochak import EVACUATION_EXIT_CODE, EvacuationController
from koochak.jobs import prepare_run, runner
assert EVACUATION_EXIT_CODE == 75
assert not EvacuationController().requested
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
