from __future__ import annotations

import json
import posixpath
import shlex
from pathlib import Path
from typing import Any, Sequence

from .specs import (
    JobHandle,
    JobStatus,
    RemotePaths,
    RenderedJob,
    TrainJobSpec,
    _safe_job_name,
    manifest_for,
    materialize_config,
)
from .ssh import CommandResult, SshCommandRunner


class SshSlurmBackend:
    """Render, stage, submit, and inspect Slurm jobs over an injected SSH command."""

    def __init__(
        self,
        *,
        ssh_command: Sequence[str] | None = None,
        remote_paths: RemotePaths,
        runner: Any | None = None,
    ) -> None:
        if runner is None and ssh_command is None:
            raise ValueError("Provide either ssh_command or runner")
        self.remote_paths = remote_paths
        self.runner = runner if runner is not None else SshCommandRunner(ssh_command=ssh_command or ())

    def render(self, job: TrainJobSpec, *, local_dir: str | Path | None = None) -> RenderedJob:
        name = _safe_job_name(job.name)
        run_dir = job.run_dir or self.remote_paths.run_dir_for(name)
        config_path = posixpath.join(run_dir, job.config_filename)
        sbatch_path = posixpath.join(run_dir, job.sbatch_filename)
        manifest_path = posixpath.join(run_dir, job.manifest_filename)
        log_dir = posixpath.join(run_dir, "slurm_logs")
        stdout_path = posixpath.join(log_dir, "%x-%j.out")
        stderr_path = posixpath.join(log_dir, "%x-%j.err")
        config_text = materialize_config(
            job.base_config,
            run_dir=run_dir,
            patches=job.patches,
            out_dir_config_key=job.out_dir_config_key,
        )
        placeholder = RenderedJob(
            name=name,
            run_dir=run_dir,
            config_path=config_path,
            sbatch_path=sbatch_path,
            manifest_path=manifest_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            config_text=config_text,
            sbatch_text="",
            manifest_text="",
        )
        sbatch_text = self._render_sbatch(job, placeholder)
        rendered = RenderedJob(
            name=name,
            run_dir=run_dir,
            config_path=config_path,
            sbatch_path=sbatch_path,
            manifest_path=manifest_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            config_text=config_text,
            sbatch_text=sbatch_text,
            manifest_text="",
        )
        manifest_text = manifest_for(job, rendered, self.remote_paths)
        rendered = RenderedJob(
            name=name,
            run_dir=run_dir,
            config_path=config_path,
            sbatch_path=sbatch_path,
            manifest_path=manifest_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            config_text=config_text,
            sbatch_text=sbatch_text,
            manifest_text=manifest_text,
        )
        if local_dir is not None:
            rendered.write_local(local_dir)
        return rendered

    def stage(self, rendered: RenderedJob) -> None:
        self.runner.run(f"mkdir -p {shlex.quote(rendered.run_dir)} {shlex.quote(posixpath.join(rendered.run_dir, 'slurm_logs'))}")
        self._write_remote_text(rendered.config_path, rendered.config_text)
        self._write_remote_text(rendered.sbatch_path, rendered.sbatch_text)
        self._write_remote_text(rendered.manifest_path, rendered.manifest_text)

    def submit(self, job: TrainJobSpec) -> JobHandle:
        rendered = self.render(job)
        self.stage(rendered)
        result = self.runner.run(f"sbatch --parsable {shlex.quote(rendered.sbatch_path)}")
        job_id = _parse_sbatch_job_id(result.stdout)
        return JobHandle(
            job_id=job_id,
            name=rendered.name,
            run_dir=rendered.run_dir,
            config_path=rendered.config_path,
            sbatch_path=rendered.sbatch_path,
            manifest_path=rendered.manifest_path,
            stdout_path=rendered.stdout_path.replace("%j", job_id).replace("%x", rendered.name),
            stderr_path=rendered.stderr_path.replace("%j", job_id).replace("%x", rendered.name),
        )

    def status(self, job: JobHandle | str) -> JobStatus:
        job_id = job.job_id if isinstance(job, JobHandle) else str(job)
        squeue_cmd = f"squeue -h -j {shlex.quote(job_id)} -o {shlex.quote('%i|%T|%M|%l|%R')}"
        queued = self.runner.run(squeue_cmd, check=False)
        status = _parse_squeue(job_id, queued.stdout)
        if status is not None:
            return status

        sacct_cmd = (
            f"sacct -n -P -j {shlex.quote(job_id)} "
            "--format=JobID,JobName,State,ExitCode,Elapsed"
        )
        accounted = self.runner.run(sacct_cmd, check=False)
        status = _parse_sacct(job_id, accounted.stdout)
        if status is not None:
            return status
        return JobStatus(job_id=job_id, state="UNKNOWN", source="none")

    def tail(self, job: JobHandle, *, stream: str = "out", lines: int = 80) -> str:
        if stream not in {"out", "err"}:
            raise ValueError("stream must be 'out' or 'err'")
        path = job.stdout_path if stream == "out" else job.stderr_path
        result = self.runner.run(f"tail -n {int(lines)} {shlex.quote(path)}", check=False)
        return result.stdout

    def metrics(
        self,
        job: JobHandle,
        *,
        relative_path: str = "log.jsonl",
        max_lines: int = 200,
    ) -> list[dict[str, Any]]:
        if relative_path.startswith("/") or ".." in Path(relative_path).parts:
            raise ValueError("relative_path must be relative and must not contain '..'")
        path = posixpath.join(job.run_dir, relative_path)
        command = f"if test -f {shlex.quote(path)}; then tail -n {int(max_lines)} {shlex.quote(path)}; fi"
        result = self.runner.run(command, check=False)
        records = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
        return records

    def _write_remote_text(self, path: str, text: str) -> None:
        parent = posixpath.dirname(path)
        command = f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(path)}"
        self.runner.run(command, input_text=text)

    def _render_sbatch(self, job: TrainJobSpec, rendered: RenderedJob) -> str:
        resources = job.resources
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={rendered.name}",
        ]
        if resources.partition:
            lines.append(f"#SBATCH --partition={resources.partition}")
        if resources.account:
            lines.append(f"#SBATCH --account={resources.account}")
        if resources.qos:
            lines.append(f"#SBATCH --qos={resources.qos}")
        if resources.constraint:
            lines.append(f"#SBATCH --constraint={resources.constraint}")
        lines.extend(
            [
                f"#SBATCH --nodes={resources.nodes}",
                f"#SBATCH --ntasks-per-node={resources.tasks_per_node}",
                f"#SBATCH --cpus-per-task={resources.cpus}",
                f"#SBATCH --mem={int(resources.mem_gb)}G",
                f"#SBATCH --time={resources.time}",
            ]
        )
        if resources.gpus:
            lines.append(f"#SBATCH --gres=gpu:{resources.gpus}")
        if resources.array:
            lines.append(f"#SBATCH --array={resources.array}")
        if resources.dependency:
            lines.append(f"#SBATCH --dependency={resources.dependency}")
        lines.extend(
            [
                f"#SBATCH --output={rendered.stdout_path}",
                f"#SBATCH --error={rendered.stderr_path}",
            ]
        )
        for directive in resources.additional_directives:
            directive = str(directive).strip()
            if not directive:
                continue
            lines.append(directive if directive.startswith("#SBATCH") else f"#SBATCH {directive}")

        env = job.runtime.environment(rendered.run_dir)
        workdir = job.workdir or self.remote_paths.repo
        command = _format_command(job.command, rendered, self.remote_paths)
        path_prefixes = ":".join(shlex.quote(str(p)) for p in self.remote_paths.path_prefixes)
        if path_prefixes:
            path_export = f"export PATH={path_prefixes}:$PATH"
        else:
            path_export = ""

        lines.extend(
            [
                "",
                "set -euo pipefail",
                "",
                f"REPO={shlex.quote(self.remote_paths.repo)}",
                f"RUN_DIR={shlex.quote(rendered.run_dir)}",
                f"CONFIG={shlex.quote(rendered.config_path)}",
                f"PYTHON={shlex.quote(self.remote_paths.python)}",
            ]
        )
        if path_export:
            lines.append(path_export)
        for key, value in env.items():
            lines.append(f"export {key}={shlex.quote(value)}")
        lines.extend(
            [
                "",
                'mkdir -p "$RUN_DIR/slurm_logs"',
                'if [[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then mkdir -p "$TORCHINDUCTOR_CACHE_DIR"; fi',
                'if [[ -n "${TRITON_CACHE_DIR:-}" ]]; then mkdir -p "$TRITON_CACHE_DIR"; fi',
                f"cd {shlex.quote(workdir)}",
                "",
                'echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname) date_utc=$(date -u --iso-8601=seconds)"',
                'echo "repo=$REPO run_dir=$RUN_DIR config=$CONFIG"',
                "if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then",
                '  echo "branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse HEAD)"',
                "fi",
                "",
                f'"$PYTHON" {command}',
                "",
            ]
        )
        return "\n".join(lines)


def _format_command(command: Sequence[str], rendered: RenderedJob, remote_paths: RemotePaths) -> str:
    values = {
        "config": rendered.config_path,
        "run_dir": rendered.run_dir,
        "repo": remote_paths.repo,
    }
    return " ".join(shlex.quote(str(token).format(**values)) for token in command)


def _parse_sbatch_job_id(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("sbatch did not return a job id")
    return lines[-1].split(";", 1)[0]


def _parse_squeue(job_id: str, stdout: str) -> JobStatus | None:
    for line in stdout.splitlines():
        parts = line.strip().split("|", 4)
        if len(parts) != 5:
            continue
        raw_id, state, elapsed, time_limit, reason = parts
        if raw_id.split("_", 1)[0] == job_id:
            return JobStatus(
                job_id=raw_id,
                state=state,
                source="squeue",
                elapsed=elapsed or None,
                time_limit=time_limit or None,
                reason=reason or None,
            )
    return None


def _parse_sacct(job_id: str, stdout: str) -> JobStatus | None:
    for line in stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5:
            continue
        raw_id, job_name, state, exit_code, elapsed = parts[:5]
        if "." in raw_id:
            continue
        if raw_id.split("_", 1)[0] != job_id:
            continue
        return JobStatus(
            job_id=raw_id,
            state=state,
            source="sacct",
            elapsed=elapsed or None,
            exit_code=exit_code or None,
            job_name=job_name or None,
        )
    return None
