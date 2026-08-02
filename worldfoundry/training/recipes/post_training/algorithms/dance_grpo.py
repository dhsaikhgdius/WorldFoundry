"""DANCE recipe contract for native stochastic flow-policy training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from .flow_grpo import (
    FLOW_GRPO_ALGORITHM_FIELDS,
    FlowGRPOAlgorithmSpec,
)
from .flow_policy import parse_flow_policy_fields

DANCE_GRPO_ALGORITHM_FIELDS = FLOW_GRPO_ALGORITHM_FIELDS | {
    "update_timestep_fraction",
}


@dataclass(frozen=True, slots=True)
class DanceGRPOAlgorithmSpec(FlowGRPOAlgorithmSpec):
    """All-SDE rollout with an independent random update subset per sample."""

    update_timestep_fraction: float = 0.6
    type: str = "dance-grpo"

    algorithm_type: ClassVar[str] = "dance-grpo"

    def __post_init__(self) -> None:
        FlowGRPOAlgorithmSpec.__post_init__(self)
        fraction = float(self.update_timestep_fraction)
        if not isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("update_timestep_fraction must be finite and in (0,1]")
        transition_count = len(self.sigmas) - 1
        if self.sde_step_indices != tuple(range(transition_count)):
            raise ValueError("DANCE requires every rollout transition to be stochastic")
        if int(transition_count * fraction) <= 0:
            raise ValueError("update_timestep_fraction selects no DANCE transition")
        if self.transition_strategy != "constant-diffusion":
            raise ValueError("DANCE requires the constant-diffusion transition strategy")
        if not self.init_same_noise:
            raise ValueError("DANCE requires shared initial noise within each prompt group")
        if self.old_log_prob_source != "rollout":
            raise ValueError("DANCE requires rollout old log-probabilities")
        if self.requires_reference_policy:
            raise ValueError("DANCE does not use reference-policy replay")
        if self.advantage_normalization != "group-sample-std":
            raise ValueError("DANCE requires group-sample-std advantage normalization")
        if self.clip_schedule != "constant":
            raise ValueError("DANCE uses a constant clipping range")
        object.__setattr__(self, "update_timestep_fraction", fraction)


def parse_dance_grpo_algorithm(value: object) -> DanceGRPOAlgorithmSpec:
    return DanceGRPOAlgorithmSpec(
        **parse_flow_policy_fields(value, allowed=DANCE_GRPO_ALGORITHM_FIELDS)
    )


__all__ = ["DanceGRPOAlgorithmSpec"]
