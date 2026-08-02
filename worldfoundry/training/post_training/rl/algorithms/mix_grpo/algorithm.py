"""Loss stage for native MixGRPO policy updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..flow_grpo.algorithm import FlowGRPOStageAlgorithm


@dataclass(frozen=True, slots=True)
class MixGRPOStageAlgorithm(FlowGRPOStageAlgorithm):
    name: ClassVar[str] = "mix-grpo"

    @property
    def state_fields(self):
        return {
            **super(MixGRPOStageAlgorithm, self).state_fields,
            "advantage_aggregation": "component-first",
        }


__all__ = ["MixGRPOStageAlgorithm"]
