"""Wrapper exposing the canonical vendored DUSt3R/CroCo integration.

Modified by WorldFoundry: import paths are now registered idempotently (no
``remove + insert(0)`` preemption) and a conflicting, already-imported
``dust3r``/``croco`` copy fails fast instead of being silently reused.
"""

from __future__ import annotations

from pathlib import Path

from worldfoundry.base_models._vendor_imports import (
    assert_top_level_not_shadowed,
    prepend_import_path,
)


SOURCE_ROOT = Path(__file__).resolve().parent
DUST3R_PACKAGE_ROOT = SOURCE_ROOT / "dust3r"
CROCO_ROOT = SOURCE_ROOT / "croco"


def ensure_import_paths() -> tuple[str, str]:
    """Expose the upstream DUSt3R and CroCo packages from the canonical integration."""
    assert_top_level_not_shadowed("dust3r", SOURCE_ROOT)
    assert_top_level_not_shadowed("croco", CROCO_ROOT)
    paths = (str(SOURCE_ROOT), str(CROCO_ROOT))
    for path in reversed(paths):
        prepend_import_path(path)
    return paths


IMPORT_PATHS = ensure_import_paths()


__all__ = [
    "CROCO_ROOT",
    "DUST3R_PACKAGE_ROOT",
    "IMPORT_PATHS",
    "SOURCE_ROOT",
    "ensure_import_paths",
]
