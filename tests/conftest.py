"""Shared collection guards for the ``tests/`` tree (infra plan C-09).

Several subtrees import torch (or torch-backed worldfoundry packages) at
module import time. Without torch, bare ``pytest`` collection ERRORs instead
of finishing. Skip collecting those heavy subtrees when torch is missing;
with torch installed, collection is unchanged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Relative to the repository ``tests/`` directory.
_TORCH_REQUIRED_PREFIXES = (
    "synthesis/",
    "training/",
    "base_models/",
    "core/",
    "pipelines/",
    "studio/",
    "studio_visualization/",
)


def _torch_usable() -> bool:
    """True when a real torch package is importable (not an empty namespace)."""
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.loader is None:
        return False
    try:
        import torch

        return hasattr(torch, "Tensor")
    except Exception:
        return False


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:  # noqa: ARG001
    if collection_path.suffix != ".py":
        return None
    if collection_path.name == "conftest.py":
        return None
    if _torch_usable():
        return None

    try:
        rel = collection_path.resolve().relative_to(Path(__file__).resolve().parent)
    except ValueError:
        return None
    rel_s = rel.as_posix()
    if any(rel_s.startswith(prefix) for prefix in _TORCH_REQUIRED_PREFIXES):
        return True
    return None
