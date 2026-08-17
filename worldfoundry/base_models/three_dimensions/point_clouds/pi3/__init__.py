"""Wrapper exposing the vendored Pi3 source as the top-level ``pi3`` package.

Modified by WorldFoundry: the import path is registered idempotently (no
``remove + insert(0)`` preemption) and a conflicting, already-imported ``pi3``
copy fails fast instead of being silently reused.
"""

from __future__ import annotations

from pathlib import Path

from worldfoundry.base_models._vendor_imports import (
    assert_top_level_not_shadowed,
    prepend_import_path,
)


SOURCE_ROOT = Path(__file__).resolve().parent


def ensure_import_paths() -> tuple[Path, ...]:
    """Expose the shared Pi3 source as the top-level ``pi3`` package."""

    assert_top_level_not_shadowed("pi3", SOURCE_ROOT)
    prepend_import_path(SOURCE_ROOT)
    return (SOURCE_ROOT,)


__all__ = ["SOURCE_ROOT", "ensure_import_paths"]
