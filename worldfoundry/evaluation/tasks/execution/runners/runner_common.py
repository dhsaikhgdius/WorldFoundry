"""Shared helpers for benchmark-runner import/normalize flows.

Most ``run_<benchmark>_official_runner`` modules follow the same
result-import pattern: resolve paths from environment variables, turn a
computed ``{"metrics": ..., "components": ...}`` mapping into per-metric
scorecard rows, and summarize generated-video coverage against the
expected prompt ids.  The helpers below hold the single canonical copy of
that logic so individual runners only supply benchmark-specific strings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def resolve_env_path(name: str) -> Path | None:
    """Return the path stored in environment variable *name*, expanded and resolved."""
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def build_import_metric_rows(
    *,
    computed: Mapping[str, Any],
    source_path: Path,
    metric_order: Sequence[str],
    metric_specs: Mapping[str, Mapping[str, Any]],
    source: str,
    unavailable_reason: str,
    extra_fields: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the per-metric scorecard rows shared by result-import runners.

    *computed* is the ``{"metrics": ..., "components": ...}`` mapping produced
    by a benchmark's ``compute_<benchmark>_metrics`` helper.  *source* labels
    where scores came from (e.g. official runtime vs. imported results file)
    and *unavailable_reason* is recorded for metrics missing a score.
    *extra_fields* lets a runner append benchmark-specific row fields such as
    ``evidence_scope``.
    """
    direct_metrics = computed.get("metrics") if isinstance(computed.get("metrics"), Mapping) else {}
    components = computed.get("components") if isinstance(computed.get("components"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for metric_id in metric_order:
        spec = metric_specs[metric_id]
        score = direct_metrics.get(metric_id)
        row: dict[str, Any] = {
            "metric_id": metric_id,
            "name": spec["name"],
            "available": score is not None,
            "raw_score": score,
            "normalized_score": score,
            "score": score,
            "higher_is_better": spec["higher_is_better"],
            "group": spec["group"],
            "source": source,
            "source_path": str(source_path),
        }
        if extra_fields:
            row.update(extra_fields)
        row["components"] = components
        row["reason"] = None if score is not None else unavailable_reason
        rows.append(row)
    return rows


def build_video_coverage(
    expected_ids: set[str],
    generated_dir: Path | None,
    *,
    video_suffixes: frozenset[str] = VIDEO_SUFFIXES,
) -> dict[str, Any]:
    """Summarize which expected video stems exist directly under *generated_dir*."""
    actual_names: set[str] = set()
    if generated_dir is not None and generated_dir.exists():
        for path in generated_dir.iterdir():
            if path.is_file() and path.suffix.lower() in video_suffixes:
                actual_names.add(path.stem)
    missing = sorted(expected_ids - actual_names)
    unexpected = sorted(actual_names - expected_ids)
    matched = sorted(expected_ids & actual_names)
    return {
        "expected_count": len(expected_ids),
        "actual_count": len(actual_names),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "complete": bool(expected_ids) and not missing,
        "missing_ids": missing[:50],
        "unexpected_ids": unexpected[:50],
    }
