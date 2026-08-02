"""Loss stage for native DANCE policy updates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from ..flow_grpo.algorithm import FlowGRPOStageAlgorithm


@dataclass(frozen=True, slots=True)
class DanceGRPOStageAlgorithm(FlowGRPOStageAlgorithm):
    """Clipped policy loss over a stored per-sample timestep subset."""

    update_timestep_fraction: float = 0.6

    name: ClassVar[str] = "dance-grpo"
    supports_update_step_mask: ClassVar[bool] = True

    def __post_init__(self) -> None:
        FlowGRPOStageAlgorithm.__post_init__(self)
        fraction = float(self.update_timestep_fraction)
        if not isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("update_timestep_fraction must be finite and in (0,1]")
        object.__setattr__(self, "update_timestep_fraction", fraction)

    @property
    def state_fields(self):
        return {
            **super(DanceGRPOStageAlgorithm, self).state_fields,
            "update_timestep_fraction": self.update_timestep_fraction,
        }


__all__ = ["DanceGRPOStageAlgorithm"]
