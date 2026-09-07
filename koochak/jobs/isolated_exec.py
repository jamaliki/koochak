"""Run a Python target after installing explicit profile import roots."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


def _target(arguments: list[str]) -> tuple[str, str, list[str]]:
    for index, argument in enumerate(arguments):
        if argument in {"-m", "-c"}:
            if index + 1 >= len(arguments):
                raise ValueError(f"{argument} requires a target")
            return argument, arguments[index + 1], arguments[index + 2 :]
        if not argument.startswith("-"):
            return "script", argument, arguments[index + 1 :]
    raise ValueError("isolated Python child has no module, script, or command target")


def main() -> int:
    if "--" not in sys.argv[1:]:
        raise SystemExit(
            "isolated_exec requires '--' before the original Python arguments"
        )
    separator = sys.argv.index("--")
    try:
        import_paths = json.loads(sys.argv[1])
        if not isinstance(import_paths, list) or not all(
            isinstance(item, str) for item in import_paths
        ):
            raise ValueError("import roots must be a list of strings")
        mode, target, arguments = _target(sys.argv[separator + 1 :])
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid isolated Python invocation: {exc}") from exc

    for import_path in reversed(import_paths):
        sys.path.insert(0, import_path)
    try:
        if mode == "-m":
            sys.argv = [target, *arguments]
            runpy.run_module(target, run_name="__main__", alter_sys=True)
        elif mode == "-c":
            sys.argv = ["-c", *arguments]
            namespace = {"__name__": "__main__", "__file__": "<string>"}
            exec(compile(target, "<string>", "exec"), namespace, namespace)  # noqa: S102
        else:
            script = str(Path(target))
            sys.argv = [script, *arguments]
            runpy.run_path(script, run_name="__main__")
    except Exception as exc:
        # A strict checkpoint was already written before its acknowledgement
        # became uncertain. Preserve that safe cursor and let a compatible
        # scheduler classify the exit as retryable rather than application_exit.
        reason = getattr(exc, "reason", None)
        exit_code = getattr(exc, "exit_code", None)
        if reason == "checkpoint_ack_timeout" and type(exit_code) is int:
            print(
                json.dumps(
                    {
                        "kind": "checkpoint_safe_retry",
                        "reason": reason,
                        "checkpoint_path": getattr(exc, "checkpoint_path", None),
                        "event_id": getattr(exc, "event_id", None),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return exit_code
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
