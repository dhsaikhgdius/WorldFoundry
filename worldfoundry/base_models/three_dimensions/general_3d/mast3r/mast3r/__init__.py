# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""Vendored MASt3R package bootstrap.

Modified by WorldFoundry: the sibling DUSt3R path is registered idempotently
(no ``remove + insert(0)`` preemption) and a conflicting, already-imported
``dust3r`` copy fails fast instead of being silently reused.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from worldfoundry.base_models._vendor_imports import (
    assert_top_level_not_shadowed,
    prepend_import_path,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
GENERAL_3D_ROOT = SOURCE_ROOT.parent
DUST3R_ROOT = GENERAL_3D_ROOT / "dust3r"


def ensure_dust3r_import_paths() -> str:
    """Expose the sibling DUSt3R integration to upstream MASt3R modules."""
    assert_top_level_not_shadowed("dust3r", DUST3R_ROOT)
    path = str(DUST3R_ROOT)
    prepend_import_path(path)
    return path


def reexport_dust3r():
    """Reexport dust3r."""
    ensure_dust3r_import_paths()
    importlib.import_module("dust3r")
    module = importlib.import_module(
        "worldfoundry.base_models.three_dimensions.general_3d.dust3r"
    )
    sys.modules[f"{__name__}.dust3r"] = module
    return module


dust3r = reexport_dust3r()


__all__ = [
    "DUST3R_ROOT",
    "GENERAL_3D_ROOT",
    "PACKAGE_ROOT",
    "SOURCE_ROOT",
    "dust3r",
    "ensure_dust3r_import_paths",
    "reexport_dust3r",
]
