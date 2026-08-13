from __future__ import annotations

import sys
from pathlib import Path

import pytest

from koochak.jobs import EnvironmentProfile, load_environment_profile


def _profile_text(**replacements: str) -> str:
    values = {
        "python": sys.executable,
        "extra": "",
        "environment": "",
        "preflight": "[]",
    }
    values.update(replacements)
    return f"""
version: 1
id: portable-gpu
python: {values['python']}
environment:
  set:
    PATH: /opt/toolchain/bin:/usr/bin:/bin
    CC: /opt/toolchain/bin/cc
    CXX: /opt/toolchain/bin/c++
    TRITON_CACHE_DIR: "{{run_dir}}/triton-cache"
  secrets: [TRACKING_TOKEN]
  create_directories: ["{{run_dir}}/triton-cache"]
  {values['environment']}
requirements:
  executables: [/opt/toolchain/bin/cc]
  files: ["{{cwd}}/checkpoint.pt"]
  packages:
    torch: "*"
preflight: {values['preflight']}
{values['extra']}
"""


def test_profile_loads_strict_data_and_resolves_only_named_paths(tmp_path: Path) -> None:
    source = tmp_path / "environment.yaml"
    source.write_text(_profile_text())

    profile = load_environment_profile(source)
    resolved = profile.resolve(cwd="/shared/repo", run_dir="/shared/runs/a")

    assert profile.profile_id == "portable-gpu"
    assert resolved.variables["TRITON_CACHE_DIR"] == "/shared/runs/a/triton-cache"
    assert resolved.files == ("/shared/repo/checkpoint.pt",)
    assert "SLURM_*" in resolved.preserve
    assert resolved.secrets == ("TRACKING_TOKEN",)
    assert profile.sha256 == load_environment_profile(source).sha256


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_profile_text(extra="unknown: true"), "unexpected"),
        (_profile_text(environment="inherit: true"), "unexpected"),
        (_profile_text(environment="preserve: [NCCL_*]"), "unexpected"),
        (_profile_text(python="python"), "absolute POSIX"),
        (_profile_text(extra="note: ${oc.env:HOME}"), "do not permit"),
    ],
)
def test_profile_rejects_implicit_or_unknown_configuration(
    tmp_path: Path, text: str, message: str
) -> None:
    source = tmp_path / "environment.yaml"
    source.write_text(text)

    with pytest.raises(ValueError, match=message):
        load_environment_profile(source)


def test_profile_rejects_scheduler_owned_overrides() -> None:
    with pytest.raises(ValueError, match="runtime-owned"):
        EnvironmentProfile(
            profile_id="bad",
            python=sys.executable,
            variables={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "0"},
        )

    with pytest.raises(ValueError, match="runtime-owned"):
        EnvironmentProfile(
            profile_id="bad",
            python=sys.executable,
            variables={"PATH": "/usr/bin:/bin"},
            secrets=("SCRUFFY_JOB_ID",),
        )


def test_compilation_checks_require_explicit_absolute_compilers() -> None:
    with pytest.raises(ValueError, match="requires explicit CC and CXX"):
        EnvironmentProfile(
            profile_id="bad",
            python=sys.executable,
            variables={"PATH": "/usr/bin:/bin"},
            preflight=("torch_compile",),
        )

    with pytest.raises(ValueError, match="absolute POSIX"):
        EnvironmentProfile(
            profile_id="bad",
            python=sys.executable,
            variables={"PATH": "/usr/bin:/bin", "CC": "cc"},
            preflight=("c_compiler",),
        )
