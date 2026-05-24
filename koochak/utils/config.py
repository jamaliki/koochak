"""Compatibility shim: re-export config helpers from :mod:`koochak.config`.

Prefer :mod:`koochak.config` directly in new code. This module exists only
because earlier versions of koochak exposed config helpers under
``koochak.utils.config``; we keep the import path alive for downstream code
without re-implementing anything.
"""

from __future__ import annotations

from .. import config as _config

get = _config.get
as_dict = _config.as_dict

__all__ = ["get", "as_dict"]
