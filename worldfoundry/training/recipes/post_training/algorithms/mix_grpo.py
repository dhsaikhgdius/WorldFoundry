"""MixGRPO recipe contract for native progressive-window training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .flow_grpo import (
    FLOW_GRPO_ALGORITHM_FIELDS,
    FlowGRPOAlgorithmSpec,
)
from .flow_policy import parse_flow_policy_fields

MIX_GRPO_ALGORITHM_FIELDS = FLOW_GRPO_ALGORITHM_FIELDS


@dataclass(frozen=True, slots=True)
class MixGRPOAlgorithmSpec(FlowGRPOAlgorithmSpec):
    """Progressive ODE/SDE windows with component-first reward advantages."""

    type: str = "mix-grpo"

    algorithm_type: ClassVar[str] = "mix-grpo"

    def __post_init__(self) -> None:
        FlowGRPOAlgorithmSpec.__post_init__(self)
        if self.sde_window is None:
            raise ValueError("MixGRPO requires a progressive sde_window")
        if self.transition_strategy != "variance-preserving":
            raise ValueError("MixGRPO requires the variance-preserving transition strategy")
        if not self.init_same_noise:
            raise ValueError("MixGRPO requires shared initial noise within each prompt group")
        if self.old_log_prob_source != "rollout":
            raise ValueError("MixGRPO requires rollout old log-probabilities")
        if self.requires_reference_policy:
            raise ValueError("MixGRPO does not use reference-policy replay")
        if self.advantage_normalization != "group-sample-std":
            raise ValueError("MixGRPO requires group-sample-std component advantages")
        if self.clip_schedule != "constant":
            raise ValueError("MixGRPO uses a constant clipping range")
        if any(weight < 0 for weight in self.reward_weights.values()) or sum(
            self.reward_weights.values()
        ) <= 0:
            raise ValueError("MixGRPO reward weights must be non-negative with a positive sum")


def parse_mix_grpo_algorithm(value: object) -> MixGRPOAlgorithmSpec:
    return MixGRPOAlgorithmSpec(
        **parse_flow_policy_fields(value, allowed=MIX_GRPO_ALGORITHM_FIELDS)
    )


__all__ = ["MixGRPOAlgorithmSpec"]
