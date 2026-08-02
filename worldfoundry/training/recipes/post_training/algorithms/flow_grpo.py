"""Pure recipe contract for Flow-GRPO post-training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from ..common import validate_clip_schedule
from .flow_policy import (
    FLOW_POLICY_ALGORITHM_FIELDS,
    FlowPolicyAlgorithmSpec,
    parse_flow_policy_fields,
)

FLOW_GRPO_ALGORITHM_FIELDS = FLOW_POLICY_ALGORITHM_FIELDS | {
    "clip_range",
    "clip_schedule",
    "clip_schedule_steps",
}


@dataclass(frozen=True, slots=True)
class FlowGRPOAlgorithmSpec(FlowPolicyAlgorithmSpec):
    """Exact SDE trajectory and clipped-policy configuration."""

    clip_range: float = 1.0e-4
    clip_schedule: str = "constant"
    clip_schedule_steps: int | None = None
    type: str = "flow-grpo"

    algorithm_type: ClassVar[str] = "flow-grpo"

    def __post_init__(self) -> None:
        FlowPolicyAlgorithmSpec.__post_init__(self)
        clip_range = float(self.clip_range)
        if not isfinite(clip_range) or not 0 < clip_range < 1:
            raise ValueError("clip_range must be finite and in (0,1)")
        schedule, schedule_steps = validate_clip_schedule(
            self.clip_schedule,
            self.clip_schedule_steps,
        )
        object.__setattr__(self, "clip_range", clip_range)
        object.__setattr__(self, "clip_schedule", schedule)
        object.__setattr__(self, "clip_schedule_steps", schedule_steps)


def parse_flow_grpo_algorithm(value: object) -> FlowGRPOAlgorithmSpec:
    return FlowGRPOAlgorithmSpec(**parse_flow_policy_fields(value, allowed=FLOW_GRPO_ALGORITHM_FIELDS))


__all__ = ["FlowGRPOAlgorithmSpec"]
