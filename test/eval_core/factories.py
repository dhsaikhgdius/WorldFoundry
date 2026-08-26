"""Shared eval_core test factories (TE-03).

Centralize the small JSON/YAML manifest helpers that were previously copied
across multiple ``test/eval_core`` modules. Prefer these helpers for new tests;
migrate call sites gradually.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_json_document(path: Path, payload: object) -> Path:
    """Write ``payload`` as UTF-8 JSON, creating parent directories as needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_targets_manifest(path: Path, targets: Sequence[Mapping[str, Any]]) -> Path:
    """Write a VLA/VA/WAM-style ``{"targets": [...]}`` acquire/status manifest."""

    return write_json_document(path, {"targets": list(targets)})


def write_zoo_manifest(
    manifest_dir: Path,
    payload: object,
    *,
    filename: str = "models.yaml",
) -> Path:
    """Write a model/benchmark zoo manifest under ``manifest_dir / filename``."""

    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return write_json_document(manifest_dir / filename, payload)


def write_benchmark_manifest(manifest_dir: Path, payload: object) -> Path:
    """Write ``benchmarks.yaml`` under ``manifest_dir``."""

    return write_zoo_manifest(manifest_dir, payload, filename="benchmarks.yaml")


def write_model_manifest(manifest_dir: Path, payload: object) -> Path:
    """Write ``models.yaml`` under ``manifest_dir``."""

    return write_zoo_manifest(manifest_dir, payload, filename="models.yaml")


__all__ = [
    "write_benchmark_manifest",
    "write_json_document",
    "write_model_manifest",
    "write_targets_manifest",
    "write_zoo_manifest",
]
