"""Reproducible Python launches for Pazuzu and Scruffy."""

from importlib import import_module

# The isolated worker reads JSON manifests with the standard library. Load
# launcher dependencies only when a caller requests a public launcher symbol.
_MODULES = {
    "ConfigPatch": ".manifest",
    "EnvironmentProfile": ".profile",
    "PreparedRun": ".manifest",
    "PreparedTask": ".workflow",
    "PreparedWorkflow": ".workflow",
    "DeclaredOutput": "..storage.artifact",
    "load_environment_profile": ".profile",
    "materialize_config": ".manifest",
    "prepare_run": ".manifest",
    "stage_run": ".manifest",
    "submit_pazuzu": ".backends",
    "submit_scruffy": ".backends",
    "submit_scruffy_workflow": ".backends",
}

__all__ = [
    "ConfigPatch",
    "EnvironmentProfile",
    "PreparedRun",
    "PreparedTask",
    "PreparedWorkflow",
    "DeclaredOutput",
    "load_environment_profile",
    "materialize_config",
    "prepare_run",
    "stage_run",
    "submit_pazuzu",
    "submit_scruffy",
    "submit_scruffy_workflow",
]


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
