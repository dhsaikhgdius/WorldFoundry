"""Shared, algorithm-neutral RL objective utilities."""

from .group_advantages import (
    GroupAdvantageResult,
    WeightedComponentAdvantageResult,
    normalize_grouped_advantages,
    normalize_weighted_component_advantages,
)

__all__ = [
    "GroupAdvantageResult",
    "WeightedComponentAdvantageResult",
    "normalize_grouped_advantages",
    "normalize_weighted_component_advantages",
]
