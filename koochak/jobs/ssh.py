from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CommandResult:
    args: Sequence[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class SshCommandRunner:
    """Run remote shell commands through an injected SSH-like command prefix."""

    ssh_command: Sequence[str]
    timeout: float | None = None

    def __post_init__(self) -> None:
        if not self.ssh_command:
            raise ValueError("ssh_command must contain at least one executable")

    def run(
        self,
        remote_command: str | Sequence[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> CommandResult:
        command = _remote_command_to_string(remote_command)
        args = [*map(str, self.ssh_command), command]
        proc = subprocess.run(
            args,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
        )
        result = CommandResult(
            args=args,
            returncode=int(proc.returncode),
            stdout=str(proc.stdout),
            stderr=str(proc.stderr),
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                "Remote command failed "
                f"(returncode={result.returncode}): {command}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return result


def _remote_command_to_string(command: str | Sequence[str]) -> str:
    if isinstance(command, str):
        return command
    return " ".join(shlex.quote(str(part)) for part in command)
