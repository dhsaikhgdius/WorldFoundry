"""Strict recipe contract for GRPO-Guard flow-policy training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from .flow_policy import (
    FLOW_POLICY_ALGORITHM_FIELDS,
    FlowPolicyAlgorithmSpec,
    parse_flow_policy_fields,
)

GRPO_GUARD_ALGORITHM_FIELDS = FLOW_POLICY_ALGORITHM_FIELDS | {"clip_range"}


@dataclass(frozen=True, slots=True)
class GRPOGuardAlgorithmSpec(FlowPolicyAlgorithmSpec):
    """Mean-drift ratio normalization over stochastic flow transitions."""

    clip_range: float = 1.0e-4
    advantage_clip_max: float = 5.0
    type: str = "grpo-guard"

    algorithm_type: ClassVar[str] = "grpo-guard"

    def __post_init__(self) -> None:
        FlowPolicyAlgorithmSpec.__post_init__(self)
        clip_range = float(self.clip_range)
        if not isfinite(clip_range) or not 0 < clip_range < 1:
            raise ValueError("GRPO-Guard clip_range must be finite and in (0,1)")
        advantage_clip_max = float(self.advantage_clip_max)
        if not isfinite(advantage_clip_max) or advantage_clip_max <= 0:
            raise ValueError("GRPO-Guard advantage_clip_max must be finite and positive")
        object.__setattr__(self, "clip_range", clip_range)
        object.__setattr__(self, "advantage_clip_max", advantage_clip_max)


def parse_grpo_guard_algorithm(value: object) -> GRPOGuardAlgorithmSpec:
    return GRPOGuardAlgorithmSpec(**parse_flow_policy_fields(value, allowed=GRPO_GUARD_ALGORITHM_FIELDS))


__all__ = ["GRPOGuardAlgorithmSpec", "parse_grpo_guard_algorithm"]
