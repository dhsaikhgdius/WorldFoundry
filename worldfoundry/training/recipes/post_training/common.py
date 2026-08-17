"""Pure-data normalization helpers shared by post-training recipes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any

ADVANTAGE_NORMALIZATION_MODES = frozenset(
    {
        "group-population-variance",
        "group-population-std",
        "group-sample-std",
        "group-mean-global-population-std",
        "group-mean-global-sample-std",
    }
)
CLIP_RANGE_SCHEDULES = frozenset({"constant", "linear-decay", "cosine-decay"})


def advantage_normalization_mode(value: object, *, field_name: str) -> str:
    """Normalize one explicit grouped-advantage denominator contract."""

    resolved = str(value).strip().lower().replace("_", "-")
    if resolved not in ADVANTAGE_NORMALIZATION_MODES:
        raise ValueError(f"{field_name} must be one of {sorted(ADVANTAGE_NORMALIZATION_MODES)}")
    return resolved


def validate_clip_schedule(
    schedule: object,
    schedule_steps: object | None,
) -> tuple[str, int | None]:
    """Canonicalize a clip schedule and require an explicit decay horizon."""

    resolved = str(schedule).strip().lower().replace("_", "-")
    if resolved not in CLIP_RANGE_SCHEDULES:
        raise ValueError(f"clip_schedule must be one of {sorted(CLIP_RANGE_SCHEDULES)}")
    if resolved == "constant":
        if schedule_steps is not None:
            raise ValueError("clip_schedule_steps is unused by a constant clip schedule")
        return resolved, None
    if isinstance(schedule_steps, bool) or not isinstance(schedule_steps, int):
        raise TypeError("a decaying clip schedule requires integer clip_schedule_steps")
    if schedule_steps <= 0:
        raise ValueError("clip_schedule_steps must be positive")
    return resolved, schedule_steps


def scheduled_clip_range(
    base_clip_range: float,
    *,
    schedule: str,
    schedule_steps: int | None,
    optimizer_step: int,
) -> float:
    """Resolve UniRL-compatible clipping at an exact optimizer step."""

    base = float(base_clip_range)
    if not isfinite(base) or base < 0:
        raise ValueError("base_clip_range must be finite and non-negative")
    resolved, steps = validate_clip_schedule(schedule, schedule_steps)
    if isinstance(optimizer_step, bool) or not isinstance(optimizer_step, int):
        raise TypeError("optimizer_step must be an integer")
    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    if resolved == "constant":
        return base
    assert steps is not None
    progress = min(float(optimizer_step) / float(steps), 1.0)
    if resolved == "linear-decay":
        return base * (1.0 - 0.5 * progress)
    return base * (0.5 * (1.0 + math.cos(math.pi * progress)))


def mapping(value: object, *, field_name: str) -> dict[str, Any]:
    """Return a string-keyed copy of a mapping or fail with field context."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def strict_mapping(
    value: object,
    *,
    field_name: str,
    allowed: set[str],
) -> dict[str, Any]:
    """Normalize a mapping while rejecting every undeclared field."""

    normalized = mapping(value, field_name=field_name)
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {unknown}")
    return normalized


def positive_int(value: object, *, field_name: str) -> int:
    """Normalize a positive integer without accepting booleans."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def frozen_float_mapping(
    value: Mapping[str, object],
    *,
    field_name: str,
) -> Mapping[str, float]:
    """Normalize finite floating-point values into an immutable mapping."""

    normalized = {str(key): float(item) for key, item in value.items()}
    if not normalized or any(not key.strip() for key in normalized):
        raise ValueError(f"{field_name} must be a non-empty mapping")
    if any(not isfinite(item) for item in normalized.values()):
        raise ValueError(f"{field_name} values must be finite")
    return MappingProxyType(normalized)


def plain_data(value: object) -> object:
    """Recursively convert frozen recipe values to JSON-compatible data."""

    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, object] = {}
        for item in fields(value):
            key = "async" if item.name == "async_save" else item.name
            result[key] = plain_data(getattr(value, item.name))
        return result
    if isinstance(value, Mapping):
        return {str(key): plain_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_data(item) for item in value]
    return value


__all__ = [
    "ADVANTAGE_NORMALIZATION_MODES",
    "CLIP_RANGE_SCHEDULES",
    "advantage_normalization_mode",
    "frozen_float_mapping",
    "mapping",
    "plain_data",
    "positive_int",
    "scheduled_clip_range",
    "strict_mapping",
    "validate_clip_schedule",
]
