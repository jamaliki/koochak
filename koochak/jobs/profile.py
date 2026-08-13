"""Strict, site-neutral execution environment profiles."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from string import Formatter
from types import MappingProxyType
from typing import Any

from omegaconf import OmegaConf

_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_PROFILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_PACKAGE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_PLACEHOLDERS = {"cwd", "run_dir"}
PREFLIGHTS = frozenset({"c_compiler", "cuda", "torch_compile"})

# These values belong to the scheduler or host runtime, not to a submitted
# profile. The runner always preserves them and applies them after profile data.
RUNTIME_ENVIRONMENT = (
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_*",
    "TMPDIR",
    "XDG_RUNTIME_DIR",
    "SLURM_*",
    "SCRUFFY_*",
    "CUDA_VISIBLE_DEVICES",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)
_PROTECTED_ENVIRONMENT = RUNTIME_ENVIRONMENT


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return dict(value)


def _keys(value: Mapping[str, Any], *, allowed: set[str], required: set[str], name: str) -> None:
    missing = required - value.keys()
    extra = value.keys() - allowed
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if extra:
            details.append(f"unexpected {sorted(extra)!r}")
        raise ValueError(f"invalid {name}: {', '.join(details)}")


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return tuple(value)


def _absolute(value: str, name: str) -> str:
    if not isinstance(value, str) or not PurePosixPath(value).is_absolute() or "\x00" in value:
        raise ValueError(f"{name} must be an absolute POSIX path")
    return value


def _pattern(value: str, name: str) -> str:
    base = value.removesuffix("*")
    if not _ENVIRONMENT_NAME.fullmatch(base) or "*" in base:
        raise ValueError(f"{name} contains an invalid environment pattern: {value!r}")
    return value


def _matches(name: str, pattern: str) -> bool:
    return name.startswith(pattern[:-1]) if pattern.endswith("*") else name == pattern


def _template(value: str, variables: Mapping[str, str], name: str) -> str:
    fields = {field for _, field, _, _ in Formatter().parse(value) if field is not None}
    unknown = fields - _PLACEHOLDERS
    if unknown:
        raise ValueError(f"{name} contains unsupported placeholders: {sorted(unknown)!r}")
    return value.format_map(variables)


@dataclass(frozen=True, slots=True)
class ResolvedEnvironment:
    """The complete non-secret environment contract embedded in a launch."""

    profile_id: str
    profile_sha256: str
    python: str
    variables: Mapping[str, str]
    preserve: tuple[str, ...]
    secrets: tuple[str, ...]
    create_directories: tuple[str, ...]
    executables: tuple[str, ...]
    files: tuple[str, ...]
    packages: Mapping[str, str]
    preflight: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "sha256": self.profile_sha256,
            "python": self.python,
            "environment": {
                "set": dict(self.variables),
                "preserve": list(self.preserve),
                "secrets": list(self.secrets),
                "create_directories": list(self.create_directories),
            },
            "requirements": {
                "executables": list(self.executables),
                "files": list(self.files),
                "packages": dict(self.packages),
            },
            "preflight": list(self.preflight),
        }


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    """A validated reusable environment, before run-specific paths resolve."""

    profile_id: str
    python: str
    variables: Mapping[str, str]
    secrets: tuple[str, ...] = ()
    create_directories: tuple[str, ...] = ()
    executables: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    packages: Mapping[str, str] = MappingProxyType({})
    preflight: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not _PROFILE_ID.fullmatch(self.profile_id):
            raise ValueError("profile id must contain only letters, digits, '.', '_' or '-'")
        _absolute(self.python, "python")
        variables = dict(self.variables)
        if not all(
            isinstance(key, str)
            and _ENVIRONMENT_NAME.fullmatch(key)
            and isinstance(value, str)
            and "\x00" not in value
            for key, value in variables.items()
        ):
            raise ValueError("environment.set must map valid names to strings without NUL bytes")
        if "PATH" not in variables:
            raise ValueError("environment.set must define an explicit PATH")
        if any(not part.startswith("/") for part in variables["PATH"].split(":")):
            raise ValueError("PATH must contain only non-empty absolute entries")
        for key in variables:
            if any(_matches(key, pattern) for pattern in _PROTECTED_ENVIRONMENT):
                raise ValueError(f"environment.set cannot override runtime-owned {key}")

        secrets = tuple(_pattern(item, "environment.secrets") for item in self.secrets)
        if any("*" in item for item in secrets):
            raise ValueError("environment.secrets entries must be exact names")
        if set(secrets) & variables.keys():
            raise ValueError("secret values cannot also appear in environment.set")
        if any(
            _matches(name, pattern)
            for name in secrets
            for pattern in _PROTECTED_ENVIRONMENT
        ):
            raise ValueError("secrets cannot override runtime-owned environment names")
        executables = tuple(_absolute(item, "requirements.executables") for item in self.executables)
        files = tuple(self.files)
        packages = dict(self.packages)
        if not all(
            isinstance(name, str)
            and _PACKAGE_NAME.fullmatch(name)
            and isinstance(version, str)
            and version
            for name, version in packages.items()
        ):
            raise ValueError("requirements.packages must map import names to versions or '*'")
        if any(check not in PREFLIGHTS for check in self.preflight):
            raise ValueError(f"preflight entries must be chosen from {sorted(PREFLIGHTS)!r}")
        if "c_compiler" in self.preflight and "CC" not in variables:
            raise ValueError("c_compiler preflight requires an explicit CC")
        if "torch_compile" in self.preflight and not {"CC", "CXX"} <= variables.keys():
            raise ValueError("torch_compile preflight requires explicit CC and CXX")
        for name in ({"CC", "CXX"} & variables.keys()):
            _absolute(variables[name], f"environment.set.{name}")
        create_directories = tuple(self.create_directories)
        for value in (*create_directories, *files):
            resolved = _template(
                value, {"cwd": "/cwd", "run_dir": "/run"}, "profile path"
            )
            _absolute(resolved, "profile path")

        object.__setattr__(self, "variables", MappingProxyType(dict(sorted(variables.items()))))
        object.__setattr__(self, "secrets", secrets)
        object.__setattr__(self, "create_directories", create_directories)
        object.__setattr__(self, "executables", executables)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "packages", MappingProxyType(dict(sorted(packages.items()))))
        object.__setattr__(self, "preflight", tuple(self.preflight))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "id": self.profile_id,
            "python": self.python,
            "environment": {
                "set": dict(self.variables),
                "secrets": list(self.secrets),
                "create_directories": list(self.create_directories),
            },
            "requirements": {
                "executables": list(self.executables),
                "files": list(self.files),
                "packages": dict(self.packages),
            },
            "preflight": list(self.preflight),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def resolve(self, *, cwd: str, run_dir: str) -> ResolvedEnvironment:
        values = {"cwd": _absolute(cwd, "cwd"), "run_dir": _absolute(run_dir, "run_dir")}
        return ResolvedEnvironment(
            profile_id=self.profile_id,
            profile_sha256=self.sha256,
            python=self.python,
            variables=MappingProxyType(
                {
                    key: _template(value, values, f"environment.set.{key}")
                    for key, value in self.variables.items()
                }
            ),
            preserve=RUNTIME_ENVIRONMENT,
            secrets=self.secrets,
            create_directories=tuple(
                _absolute(_template(value, values, "environment.create_directories"), "directory")
                for value in self.create_directories
            ),
            executables=self.executables,
            files=tuple(
                _absolute(_template(value, values, "requirements.files"), "file")
                for value in self.files
            ),
            packages=self.packages,
            preflight=self.preflight,
        )


def load_environment_profile(source: str | Path) -> EnvironmentProfile:
    """Load one strict YAML profile without environment interpolation."""

    profile_path = Path(source)
    text = profile_path.read_text()
    if "${" in text:
        raise ValueError("environment profiles do not permit OmegaConf interpolation")
    document = OmegaConf.to_container(OmegaConf.create(text), resolve=False)
    data = _mapping(document, "profile")
    _keys(
        data,
        allowed={"version", "id", "python", "environment", "requirements", "preflight"},
        required={"version", "id", "python"},
        name="profile",
    )
    if data["version"] != 1:
        raise ValueError("profile version must be 1")
    environment = _mapping(data.get("environment", {}), "environment")
    _keys(
        environment,
        allowed={"set", "secrets", "create_directories"},
        required=set(),
        name="environment",
    )
    requirements = _mapping(data.get("requirements", {}), "requirements")
    _keys(
        requirements,
        allowed={"executables", "files", "packages"},
        required=set(),
        name="requirements",
    )
    variables = _mapping(environment.get("set", {}), "environment.set")
    packages = _mapping(requirements.get("packages", {}), "requirements.packages")
    return EnvironmentProfile(
        profile_id=data["id"],
        python=data["python"],
        variables=variables,
        secrets=_strings(environment.get("secrets", []), "environment.secrets"),
        create_directories=_strings(
            environment.get("create_directories", []), "environment.create_directories"
        ),
        executables=_strings(requirements.get("executables", []), "requirements.executables"),
        files=_strings(requirements.get("files", []), "requirements.files"),
        packages=packages,
        preflight=_strings(data.get("preflight", []), "preflight"),
    )
