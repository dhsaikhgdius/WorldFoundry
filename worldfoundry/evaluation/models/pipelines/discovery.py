"""Plugin discovery for pipeline bindings.

Supports two discovery channels: ``importlib.metadata`` entry points
registered under :data:`ENTRY_POINT_GROUP`, and comma-separated
``slug=module:attr`` pairs supplied via the :data:`ENV_VAR` environment
variable.  Discovery is intentionally tolerant — a broken plugin is
logged and skipped so that the built-in catalog always loads successfully.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, Mapping
from functools import lru_cache
from importlib import metadata
from typing import Any

from .bindings import PipelineBinding

# ── Constants ──────────────────────────────────────────────────────────

ENTRY_POINT_GROUP = "worldfoundry.pipelines"  #: ``importlib.metadata`` entry-point group name.
ENV_VAR = "WORLDFOUNDRY_PIPELINE_BINDINGS"  #: Environment variable for inline binding overrides.

LOGGER = logging.getLogger(__name__)


def discover_pipeline_bindings() -> dict[str, PipelineBinding]:
    """Discover third-party pipeline bindings.

    Discovery is intentionally tolerant: a broken plugin should not prevent the
    built-in catalog from loading or make CLI inspection commands unusable.

    Results are cached per ``WORLDFOUNDRY_PIPELINE_BINDINGS`` value for the
    process lifetime (entry-point scans are expensive).  Call
    :func:`clear_pipeline_binding_discovery_cache` after mutating the env var
    or installing entry points in-process (tests).
    """

    env_raw = os.environ.get(ENV_VAR, "")
    return dict(_discover_pipeline_bindings_cached(env_raw))


def clear_pipeline_binding_discovery_cache() -> None:
    """Drop cached entry-point and env-binding discovery results."""

    _discover_pipeline_bindings_cached.cache_clear()
    _pipeline_entry_points.cache_clear()


# ── Private helpers ────────────────────────────────────────────────────


@lru_cache(maxsize=8)
def _discover_pipeline_bindings_cached(env_raw: str) -> tuple[tuple[str, PipelineBinding], ...]:
    """Cached discovery keyed by the raw ``WORLDFOUNDRY_PIPELINE_BINDINGS`` value."""

    bindings: dict[str, PipelineBinding] = {}
    for name, target in _env_targets_from_raw(env_raw):
        _load_target_into(bindings, name, target, source=f"{ENV_VAR}:{name}")
    for entry_point in _pipeline_entry_points():
        try:
            value = entry_point.load()
            binding = _coerce_pipeline_binding(value, source=f"entry point {entry_point.name!r}")
            _register_discovered_binding(bindings, entry_point.name, binding)
        except Exception as exc:  # pragma: no cover - exercised through public behavior.
            LOGGER.warning("Skipping broken WorldFoundry pipeline entry point %r: %s", entry_point.name, exc)
    return tuple(sorted(bindings.items(), key=lambda item: item[0]))


@lru_cache(maxsize=1)
def _pipeline_entry_points() -> tuple[metadata.EntryPoint, ...]:
    """Select and return sorted EntryPoint list registered under the specific entry point group.

    Cached: entry-points cannot change within a process, and the full
    ``metadata.entry_points()`` scan costs ~200ms per call.  Use
    :func:`clear_pipeline_binding_discovery_cache` to force a re-scan (tests).
    """
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        selected = entry_points.select(group=ENTRY_POINT_GROUP)
    else:  # pragma: no cover - compatibility for older importlib.metadata.
        selected = entry_points.get(ENTRY_POINT_GROUP, ())
    return tuple(sorted(selected, key=lambda item: item.name))


def _env_targets() -> tuple[tuple[str, str], ...]:
    """Parse and return slug/target pairs from the designated environment variable."""
    return _env_targets_from_raw(os.environ.get(ENV_VAR, ""))


def _env_targets_from_raw(raw: str) -> tuple[tuple[str, str], ...]:
    """Parse slug/target pairs from a raw ``WORLDFOUNDRY_PIPELINE_BINDINGS`` string."""
    targets: list[tuple[str, str]] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        slug, separator, target = text.partition("=")
        if not separator or not slug.strip() or not target.strip():
            LOGGER.warning("Skipping malformed %s entry %r; expected slug=module:attr.", ENV_VAR, text)
            continue
        targets.append((slug.strip(), target.strip()))
    return tuple(sorted(targets, key=lambda item: item[0]))


def _load_target_into(
    bindings: dict[str, PipelineBinding],
    slug: str,
    target: str,
    *,
    source: str,
) -> None:
    """Import and load a dynamic binding target, registering it into bindings dictionary."""
    try:
        module_name, separator, attr_name = target.partition(":")
        if not separator or not module_name or not attr_name:
            raise ValueError(f"expected module:attr target, got {target!r}")
        module = importlib.import_module(module_name)
        value: Any = module
        for part in attr_name.split("."):
            value = getattr(value, part)
        binding = _coerce_pipeline_binding(value, source=source)
        _register_discovered_binding(bindings, slug, binding)
    except Exception as exc:
        LOGGER.warning("Skipping broken WorldFoundry pipeline binding %r from %s: %s", slug, source, exc)


def _coerce_pipeline_binding(value: Any, *, source: str) -> PipelineBinding:
    """Coerce any resolved factory, mapping, or object into a validated PipelineBinding."""
    if isinstance(value, PipelineBinding):
        binding = value
    elif isinstance(value, Mapping):
        binding = PipelineBinding.from_mapping(value)
    elif callable(value):
        produced = _call_binding_factory(value, source=source)
        binding = _coerce_pipeline_binding(produced, source=source)
    else:
        raise TypeError(f"{source} must provide a PipelineBinding, mapping, or zero-arg factory.")
    binding.validate()
    return binding


def _call_binding_factory(value: Callable[[], Any], *, source: str) -> Any:
    """Call a zero-argument callable factory with appropriate error raising."""
    try:
        return value()
    except TypeError as exc:
        raise TypeError(f"{source} factory must be callable without arguments.") from exc


def _register_discovered_binding(
    bindings: dict[str, PipelineBinding],
    slug: str,
    binding: PipelineBinding,
) -> None:
    """Register a discovered PipelineBinding in the results dictionary with duplicate checking."""
    key = _discovery_key(slug)
    binding_key = _discovery_key(binding.binding_id)
    for candidate in (key, binding_key):
        if candidate in bindings:
            LOGGER.warning("Skipping duplicate WorldFoundry pipeline binding %r from slug %r.", binding.binding_id, slug)
            return
    bindings[key] = binding


def _discovery_key(value: str) -> str:
    """Normalize a discovery slug or binding ID key for dictionary insertion/lookup."""
    return str(value).strip().lower()


__all__ = [
    "ENTRY_POINT_GROUP",
    "ENV_VAR",
    "clear_pipeline_binding_discovery_cache",
    "discover_pipeline_bindings",
]
