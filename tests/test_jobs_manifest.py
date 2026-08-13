from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from koochak.jobs import (
    ConfigPatch,
    EnvironmentProfile,
    materialize_config,
    prepare_run,
    stage_run,
)


def _profile() -> EnvironmentProfile:
    return EnvironmentProfile(
        profile_id="portable",
        python=sys.executable,
        variables={
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "TRITON_CACHE_DIR": "{run_dir}/triton-cache",
        },
        create_directories=("{run_dir}/triton-cache",),
    )


def _config(tmp_path: Path) -> Path:
    source = tmp_path / "base.yaml"
    source.write_text("train:\n  max_steps: 10\n  out_dir: old\n")
    return source


def test_materialize_config_applies_patches(tmp_path: Path) -> None:
    text = materialize_config(
        _config(tmp_path),
        run_dir="/runs/a",
        patches=[ConfigPatch("train.max_steps", 20)],
    )
    config = OmegaConf.create(text)

    assert config.train.max_steps == 20
    assert config.train.out_dir == "/runs/a"


def test_prepare_run_is_deterministic_and_contains_resolved_inputs(tmp_path: Path) -> None:
    run_dir = str(tmp_path / "runs" / "a")
    cwd = str(tmp_path / "repo")
    Path(cwd).mkdir()
    arguments = ["-m", "project.train", "--config", "{config}"]

    first = prepare_run(
        name="training-a",
        profile=_profile(),
        python_args=arguments,
        cwd=cwd,
        run_dir=run_dir,
        base_config=_config(tmp_path),
        patches=[ConfigPatch("train.max_steps", 20)],
    )
    second = prepare_run(
        name="training-a",
        profile=_profile(),
        python_args=arguments,
        cwd=cwd,
        run_dir=run_dir,
        base_config=_config(tmp_path),
        patches=[ConfigPatch("train.max_steps", 20)],
    )

    assert first.manifest_sha256 == second.manifest_sha256
    manifest = json.loads(first.artifacts[-1].content)
    assert manifest["argv"] == [
        sys.executable,
        "-m",
        "project.train",
        "--config",
        f"{run_dir}/config.yaml",
    ]
    assert manifest["environment"]["sha256"] == _profile().sha256
    assert manifest["config"]["sha256"] == first.artifacts[0].sha256
    assert first.runner_argv()[1:4] == ["-I", "-m", "koochak.jobs.runner"]


def test_stage_is_idempotent_but_never_clobbers_another_launch(tmp_path: Path) -> None:
    run_dir = str(tmp_path / "run")
    common = {
        "name": "job",
        "profile": _profile(),
        "cwd": str(tmp_path),
        "run_dir": run_dir,
    }
    first = prepare_run(python_args=["-c", "print(1)"], **common)
    conflict = prepare_run(python_args=["-c", "print(2)"], **common)

    stage_run(first)
    stage_run(first)
    with pytest.raises(FileExistsError, match="different artifact"):
        stage_run(conflict)

    assert Path(first.manifest_path).read_bytes() == first.artifacts[-1].content
    assert Path(first.manifest_path).stat().st_mode & 0o777 == 0o444


def test_config_placeholder_requires_a_materialized_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires base_config"):
        prepare_run(
            name="job",
            profile=_profile(),
            python_args=["train.py", "--config", "{config}"],
            cwd=str(tmp_path),
            run_dir=str(tmp_path / "run"),
        )
