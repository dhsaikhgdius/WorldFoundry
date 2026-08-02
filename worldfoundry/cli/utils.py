"""CLI utility helpers for JSON serialization, key-value parsing, and zoo-id resolution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_JSON_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")


def json_dump(payload: object) -> None:
    """Print the sole machine-readable payload for a CLI ``--json`` response.

    Operational progress, warnings, and logs must use stderr/the logging
    pipeline instead; callers use this helper only for their final result.

    Args:
        payload: JSON-serializable command response payload.
    """
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), flush=True)


def parse_json_value(value: str) -> object:
    """Parse a CLI scalar as JSON only when it is clearly JSON.

    Args:
        value: Raw command-line value.
    """
    stripped = value.strip()
    if stripped in {"true", "false", "null"} or stripped[:1] in {'"', "{", "["} or _JSON_NUMBER_RE.fullmatch(stripped):
        return json.loads(value)
    return value


def parse_key_value_mapping(values: list[str] | None) -> dict[str, object]:
    """Parse repeated `KEY=VALUE` flags.

    Args:
        values: Repeated key-value CLI flag values.
    """
    payload: dict[str, object] = {}
    for item in values or ():
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        payload[key.strip()] = parse_json_value(value)
    return payload


def load_json_mapping(path: Path | None) -> dict[str, Any] | None:
    """Load a JSON object from an optional path.

    Args:
        path: Optional JSON file path.
    """
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return payload


def canonical_model_zoo_id(value: str | None, manifest_dir: Path | None) -> str | None:
    """Resolve a model-zoo id or alias when a manifest directory is available.

    Args:
        value: Model id or alias.
        manifest_dir: Optional model-zoo manifest directory.
    """
    if value is None:
        return None
    if manifest_dir is None or not manifest_dir.exists():
        return value
    from worldfoundry.evaluation.models.catalog import load_model_zoo_registry

    return load_model_zoo_registry(manifest_dir).get(value).model_id


def canonical_benchmark_zoo_id(value: str | None, manifest_dir: Path | None) -> str | None:
    """Resolve a benchmark-zoo id or alias.

    Args:
        value: Benchmark id or alias.
        manifest_dir: Optional benchmark-zoo manifest directory or file.
    """
    if value is None:
        return None
    if manifest_dir is None or not manifest_dir.exists():
        return value
    from worldfoundry.evaluation.tasks.catalog.schema import load_entries
    from worldfoundry.evaluation.tasks.catalog.zoo_registry import BenchmarkZooRegistry, load_benchmark_zoo_registry

    if manifest_dir.is_file():
        return BenchmarkZooRegistry(load_entries(manifest_dir)).get(value).benchmark_id
    return load_benchmark_zoo_registry(manifest_dir).get(value).benchmark_id


def resolve_cli_benchmark_for_materialize(task_type: str, benchmark_name: str) -> Any:
    """Resolve a benchmark adapter for legacy task-type/materialize CLI flows."""
    raise ValueError(
        "Task-type/benchmark-name materialization is retired for benchmark-zoo entries. "
        "Use `worldfoundry-eval run --benchmark <id> --model <id>` or "
        "`worldfoundry-eval task materialize` with a filesystem task YAML."
    )
