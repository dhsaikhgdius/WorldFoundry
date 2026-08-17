"""Wrapper exposing the vendored MASt3R integration plus the canonical DUSt3R.

Modified by WorldFoundry: import paths are now registered idempotently (no
``remove + insert(0)`` preemption) and a conflicting, already-imported
``mast3r``/``dust3r``/``croco`` copy fails fast instead of being silently
reused.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from worldfoundry.base_models._vendor_imports import (
    assert_top_level_not_shadowed,
    prepend_import_path,
)


SOURCE_ROOT = Path(__file__).resolve().parent
GENERAL_3D_ROOT = SOURCE_ROOT.parent
DUST3R_ROOT = GENERAL_3D_ROOT / "dust3r"


def ensure_import_paths() -> tuple[str, str]:
    """Expose upstream MASt3R and DUSt3R packages without nesting them in model runtimes."""
    assert_top_level_not_shadowed("mast3r", SOURCE_ROOT)
    assert_top_level_not_shadowed("dust3r", DUST3R_ROOT)
    assert_top_level_not_shadowed("croco", DUST3R_ROOT / "croco")
    paths = (str(SOURCE_ROOT), str(DUST3R_ROOT))
    for path in paths:
        prepend_import_path(path)
    return paths


def reexport_dust3r():
    """Expose the canonical DUSt3R integration as ``worldfoundry...mast3r.dust3r``."""
    ensure_import_paths()
    module = importlib.import_module(
        "worldfoundry.base_models.three_dimensions.general_3d.dust3r"
    )
    sys.modules[f"{__name__}.dust3r"] = module
    return module


dust3r = reexport_dust3r()


__all__ = [
    "DUST3R_ROOT",
    "GENERAL_3D_ROOT",
    "SOURCE_ROOT",
    "dust3r",
    "ensure_import_paths",
    "reexport_dust3r",
]
