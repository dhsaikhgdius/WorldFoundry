"""Bootstrap the WorldFoundry checkout onto ``sys.path`` for docs generators.

Fumadocs CI runs these scripts without an editable install, so
``worldfoundry.core.io.paths.project_root`` is only importable after the
repository root (the ancestor that owns ``pyproject.toml``) is on
``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_importable(start: str | Path | None = None) -> Path:
    """Locate the repo root from ``start`` and prepend it to ``sys.path``."""
    current = Path(start if start is not None else __file__).resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").is_file():
            root_s = str(parent)
            if root_s not in sys.path:
                sys.path.insert(0, root_s)
            return parent
    raise RuntimeError(f"Could not locate WorldFoundry project root from {start!r}")
