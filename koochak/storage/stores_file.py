"""Named stores declared in a private YAML file.

Koochak holds all storage logic; the site-specific facts (mount roots,
publish modes, measured profiles) live in a small YAML file that stays out of
public repositories::

    version: 1
    stores:
      scratch:
        type: local
        root: /path/to/parallel-filesystem/${oc.env:USER}
        publish: link
        profile: {request_seconds: 0.0003, stream_mb_s: 1000, streams: 4, range_streams: 32, part_bytes: 16MiB}
      archive:
        type: local
        root: /path/to/object-storage-mount/${oc.env:USER}
        publish: exclusive
        profile: {request_seconds: 0.14, stream_mb_s: 16, streams: 32, list_is_cheap: false}

``open_store("archive://datasets/foo")`` then returns a ``LocalStore`` rooted
at ``<root>/datasets/foo`` with that entry's settings, so the same relative
path under two schemes names the same data on two tiers.

The file is ``$KOOCHAK_STORES`` when set (it must exist), else
``$XDG_CONFIG_HOME/koochak/stores.yaml`` (default ``~/.config/koochak``) when
present. Values are resolved with OmegaConf, so ``${oc.env:USER}`` works.
Unknown keys fail. The file holds no secrets: backends that need credentials
name environment variables instead.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

from omegaconf import OmegaConf

from ..utils.sizes import parse_size
from .store import LocalStore, StoreProfile, validate_scheme

__all__ = [
    "ENV_VAR",
    "StoreSpec",
    "default_stores_path",
    "load_default_or_given",
    "load_stores_file",
]

ENV_VAR = "KOOCHAK_STORES"
VERSION = 1

_ENTRY_KEYS = frozenset(
    {"type", "root", "publish", "fsync", "verify_readback", "settle_seconds", "profile"}
)
_PROFILE_KEYS = frozenset(field.name for field in dataclasses.fields(StoreProfile))
_CACHE: Dict[str, tuple[int, int, Dict[str, "StoreSpec"]]] = {}


@dataclass(frozen=True, slots=True)
class StoreSpec:
    """One named store from the stores file."""

    scheme: str
    root: str
    publish: str = "link"
    fsync: bool = True
    verify_readback: bool = False
    settle_seconds: float = 0.0
    profile: Optional[StoreProfile] = None

    def open(self, subpath: str = "") -> LocalStore:
        """A store rooted at ``root/subpath`` (``subpath`` is a validated key or empty)."""

        root = os.path.join(self.root, *subpath.split("/")) if subpath else self.root
        return LocalStore(
            root,
            publish=self.publish,
            fsync=self.fsync,
            verify_readback=self.verify_readback,
            settle_seconds=self.settle_seconds,
            profile=self.profile,
        )


def default_stores_path() -> Optional[str]:
    """``$KOOCHAK_STORES``, else the XDG config file if it exists, else ``None``."""

    explicit = os.environ.get(ENV_VAR)
    if explicit:
        return explicit
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    candidate = os.path.join(base, "koochak", "stores.yaml")
    return candidate if os.path.isfile(candidate) else None


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false")
    return value


def _profile(value: object, label: str) -> StoreProfile:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(value) - _PROFILE_KEYS
    if unknown:
        raise ValueError(f"{label} has unknown keys {sorted(unknown)}; allowed: {sorted(_PROFILE_KEYS)}")
    fields = dict(value)
    if "part_bytes" in fields:
        fields["part_bytes"] = parse_size(fields["part_bytes"])
    return StoreProfile(**fields)


def _entry(scheme: str, value: object, source: str) -> StoreSpec:
    label = f"{source}: stores.{scheme}"
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(value) - _ENTRY_KEYS
    if unknown:
        raise ValueError(f"{label} has unknown keys {sorted(unknown)}; allowed: {sorted(_ENTRY_KEYS)}")
    if value.get("type") != "local":
        raise ValueError(f"{label}.type must be 'local' (the only backend type so far)")
    root = value.get("root")
    if not isinstance(root, str) or not root or not os.path.isabs(os.path.expanduser(root)):
        raise ValueError(f"{label}.root must be an absolute path")
    settle = value.get("settle_seconds", 0.0)
    if isinstance(settle, bool) or not isinstance(settle, (int, float)) or not math.isfinite(settle):
        raise ValueError(f"{label}.settle_seconds must be a number")
    return StoreSpec(
        scheme=scheme,
        root=os.path.abspath(os.path.expanduser(root)),
        publish=str(value.get("publish", "link")),
        fsync=_require_bool(value.get("fsync", True), f"{label}.fsync"),
        verify_readback=_require_bool(value.get("verify_readback", False), f"{label}.verify_readback"),
        settle_seconds=float(settle),
        profile=_profile(value["profile"], f"{label}.profile") if "profile" in value else None,
    )


def load_stores_file(path: Union[str, os.PathLike[str]]) -> Dict[str, StoreSpec]:
    """Parse and strictly validate a stores file; results are cached until it changes."""

    absolute = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(absolute):
        raise FileNotFoundError(f"stores file not found: {absolute}")
    observed = os.stat(absolute)
    cached = _CACHE.get(absolute)
    if cached is not None and cached[:2] == (observed.st_mtime_ns, observed.st_size):
        return dict(cached[2])
    data: Any = OmegaConf.to_container(OmegaConf.load(absolute), resolve=True)
    if not isinstance(data, dict) or set(data) != {"version", "stores"}:
        raise ValueError(f"{absolute}: a stores file has exactly the keys 'version' and 'stores'")
    if data["version"] != VERSION:
        raise ValueError(f"{absolute}: unsupported stores file version {data['version']!r}")
    stores = data["stores"]
    if not isinstance(stores, dict) or not stores:
        raise ValueError(f"{absolute}: 'stores' must be a non-empty mapping")
    specs = {}
    for scheme, value in stores.items():
        try:
            validate_scheme(scheme)
        except ValueError as exc:
            raise ValueError(f"{absolute}: {exc}") from exc
        specs[scheme] = _entry(scheme, value, absolute)
        # Build once so invalid combinations (e.g. publish modes) fail at load time.
        specs[scheme].open()
    _CACHE[absolute] = (observed.st_mtime_ns, observed.st_size, specs)
    return dict(specs)


def load_default_or_given(
    stores_file: Optional[Union[str, os.PathLike[str]]] = None,
) -> Dict[str, StoreSpec]:
    """Specs from ``stores_file`` or the default location; empty when there is no file."""

    path = stores_file if stores_file is not None else default_stores_path()
    return {} if path is None else load_stores_file(path)
