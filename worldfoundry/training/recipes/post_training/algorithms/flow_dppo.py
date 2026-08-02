"""Pure recipe contract for Flow-DPPO post-training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from .flow_policy import (
    FLOW_POLICY_ALGORITHM_FIELDS,
    FlowPolicyAlgorithmSpec,
    parse_flow_policy_fields,
)

FLOW_DPPO_ALGORITHM_FIELDS = FLOW_POLICY_ALGORITHM_FIELDS | {
    "add_kl_coefficient",
    "kl_mask_threshold",
}


@dataclass(frozen=True, slots=True)
class FlowDPPOAlgorithmSpec(FlowPolicyAlgorithmSpec):
    """Flow-DPPO KL-advantage masking and shared rollout configuration."""

    kl_mask_threshold: float = 1.0e-5
    add_kl_coefficient: bool = True
    type: str = "flow-dppo"

    algorithm_type: ClassVar[str] = "flow-dppo"

    def __post_init__(self) -> None:
        FlowPolicyAlgorithmSpec.__post_init__(self)
        threshold = float(self.kl_mask_threshold)
        if not isfinite(threshold) or threshold < 0:
            raise ValueError("kl_mask_threshold must be finite and non-negative")
        if not isinstance(self.add_kl_coefficient, bool):
            raise TypeError("add_kl_coefficient must be a bool")
        object.__setattr__(self, "kl_mask_threshold", threshold)


def parse_flow_dppo_algorithm(value: object) -> FlowDPPOAlgorithmSpec:
    return FlowDPPOAlgorithmSpec(**parse_flow_policy_fields(value, allowed=FLOW_DPPO_ALGORITHM_FIELDS))


__all__ = ["FlowDPPOAlgorithmSpec"]
