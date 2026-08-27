"""Cleanable scratch directories under the WorldFoundry cache root.

Pipelines that receive no explicit ``output_dir`` need somewhere to write
intermediate artifacts. A bare ``tempfile.mkdtemp()`` scatters uncleanable
directories under ``/tmp``; the helpers here keep those defaults under
``${WORLDFOUNDRY_CACHE_DIR}/scratch/<YYYYMMDD>/`` instead, so leftovers are
easy to locate and prune, and each directory is registered for best-effort
removal at interpreter exit.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from worldfoundry.core.io.paths import cache_root_path


def scratch_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the scratch root under the WorldFoundry cache root (without creating it)."""
    return cache_root_path(env) / "scratch"


def make_scratch_dir(prefix: str = "worldfoundry_", *, cleanup_at_exit: bool = True) -> Path:
    """Create a unique scratch directory under the WorldFoundry cache root.

    The directory is created as ``<scratch root>/<UTC date>/<prefix><random>``
    and, unless ``cleanup_at_exit`` is false, registered for best-effort
    recursive removal at interpreter exit (``shutil.rmtree`` with errors
    ignored, so an already-deleted or busy directory never breaks shutdown).

    Exit cleanup is best-effort only: long-lived processes (e.g. Studio or
    model servers) accumulate scratch directories until they terminate, so
    they should pass an explicit ``output_dir`` instead of relying on this
    default. Directories orphaned by crashed runs stay under the dated
    scratch folder, where they are easy to find and delete.
    """
    day_root = scratch_root_path() / datetime.now(timezone.utc).strftime("%Y%m%d")
    day_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=prefix, dir=day_root))
    if cleanup_at_exit:
        atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    return scratch


__all__ = ["make_scratch_dir", "scratch_root_path"]
