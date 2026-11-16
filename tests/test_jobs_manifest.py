from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from koochak.jobs import (
    ConfigPatch,
    EnvironmentProfile,
    Preemptible,
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
    assert first.runner_argv()[1:3] == ["-I", "-c"]
    assert first.runner_argv()[-2:] == [first.manifest_path, first.manifest_sha256]
    assert json.loads(first.artifacts[-1].content)["runner_runtime"]["sha256"]


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


def test_preemptible_run_enables_evacuation_and_auto_resume_and_records_policy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    source.write_text(
        "train:\n  max_steps: 10\n  evacuation_enabled: false\n  resume: none\n"
    )
    prepared = prepare_run(
        name="preemptible",
        profile=_profile(),
        python_args=["-m", "koochak.cli.train", "--config", "{config}"],
        cwd="/shared/project",
        run_dir="/shared/runs/preemptible",
        base_config=source,
        preemptible=Preemptible(grace_seconds=900, max_attempts=6),
    )

    artifacts = {artifact.path: artifact.content for artifact in prepared.artifacts}
    config = OmegaConf.create(artifacts["/shared/runs/preemptible/config.yaml"].decode())
    assert config.train.evacuation_enabled is True
    assert config.train.resume == "auto"
    manifest = json.loads(artifacts[prepared.manifest_path])
    assert manifest["preemption"] == {
        "max_attempts": 6,
        "retry_on": ["allocation_replaced", "allocation_incarnation_changed", "evacuated"],
        "evacuation": {"signal": "USR1", "grace_seconds": 900},
    }
    assert prepared.preemption == Preemptible(grace_seconds=900, max_attempts=6)


def test_runs_are_not_preemptible_by_default(tmp_path: Path) -> None:
    prepared = prepare_run(
        name="steady",
        profile=_profile(),
        python_args=["-m", "project.train"],
        cwd="/shared/project",
        run_dir="/shared/runs/steady",
        base_config=_config(tmp_path),
    )
    manifest = json.loads(prepared.artifacts[-1].content)
    assert "preemption" not in manifest
    assert prepared.preemption is None


def test_preemptible_needs_a_config_unless_the_entrypoint_owns_evacuation(
    tmp_path: Path,
) -> None:
    arguments = dict(
        name="preemptible",
        profile=_profile(),
        python_args=["-m", "project.train"],
        cwd="/shared/project",
        run_dir="/shared/runs/preemptible",
    )
    with pytest.raises(ValueError, match="base_config"):
        prepare_run(**arguments, preemptible=True)
    prepared = prepare_run(**arguments, preemptible=Preemptible(config_section=None))
    manifest = json.loads(prepared.artifacts[-1].content)
    assert manifest["preemption"]["evacuation"] == {"signal": "USR1", "grace_seconds": 600}
    assert "config" not in manifest


@pytest.mark.parametrize(
    "patch",
    [
        ConfigPatch("train.evacuation_enabled", False),
        ConfigPatch("train.resume", "none"),
    ],
)
def test_preemptible_rejects_a_conflicting_patch(tmp_path: Path, patch: ConfigPatch) -> None:
    with pytest.raises(ValueError, match="conflicts with preemptible"):
        prepare_run(
            name="preemptible",
            profile=_profile(),
            python_args=["-m", "project.train"],
            cwd="/shared/project",
            run_dir="/shared/runs/preemptible",
            base_config=_config(tmp_path),
            patches=[patch],
            preemptible=True,
        )


@pytest.mark.parametrize(
    "values",
    [
        {"grace_seconds": 0},
        {"grace_seconds": 1.5},
        {"grace_seconds": True},
        {"max_attempts": 1},
        {"max_attempts": 11},
        {"config_section": ""},
    ],
)
def test_preemptible_policy_is_validated(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Preemptible(**values)  # type: ignore[arg-type]


def test_preemptible_flag_must_be_a_bool_or_policy(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="preemptible"):
        prepare_run(
            name="preemptible",
            profile=_profile(),
            python_args=["-m", "project.train"],
            cwd="/shared/project",
            run_dir="/shared/runs/preemptible",
            base_config=_config(tmp_path),
            preemptible="yes",  # type: ignore[arg-type]
        )
