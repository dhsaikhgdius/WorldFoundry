"""Strict recipe contract for Bagel Flow-UniGRPO training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from .flow_policy import (
    FLOW_POLICY_ALGORITHM_FIELDS,
    FlowPolicyAlgorithmSpec,
    parse_flow_policy_fields,
)

BAGEL_FLOW_UNIGRPO_ALGORITHM_FIELDS = FLOW_POLICY_ALGORITHM_FIELDS | {
    "clip_range",
    "velocity_mse_weight",
    "ratio_norm",
    "grad_reweight",
}


@dataclass(frozen=True, slots=True)
class BagelFlowUniGRPOAlgorithmSpec(FlowPolicyAlgorithmSpec):
    """Clipped flow policy loss regularized toward a frozen base velocity."""

    velocity_mse_weight: float = 1.0
    clip_range: float = 1.0e-4
    ratio_norm: bool = False
    grad_reweight: bool = False
    type: str = "bagel-flow-unigrpo"

    algorithm_type: ClassVar[str] = "bagel-flow-unigrpo"

    @property
    def requires_reference_policy(self) -> bool:
        return float(self.velocity_mse_weight) > 0

    def __post_init__(self) -> None:
        FlowPolicyAlgorithmSpec.__post_init__(self)
        clip_range = float(self.clip_range)
        velocity_mse_weight = float(self.velocity_mse_weight)
        if not isfinite(clip_range) or not 0 < clip_range < 1:
            raise ValueError("Bagel Flow-UniGRPO clip_range must be finite and in (0,1)")
        if not isfinite(velocity_mse_weight) or velocity_mse_weight <= 0:
            raise ValueError("Bagel Flow-UniGRPO velocity_mse_weight must be finite and positive")
        if float(self.reference_kl_weight) != 0:
            raise ValueError("Bagel Flow-UniGRPO uses velocity MSE instead of reference KL")
        if self.guidance_scale != 1:
            raise ValueError("Bagel Flow-UniGRPO velocity regularization requires unguided replay")
        if self.transition_strategy != "variance-preserving":
            raise ValueError("Bagel Flow-UniGRPO requires the variance-preserving flow transition")
        if not isinstance(self.ratio_norm, bool) or not isinstance(
            self.grad_reweight,
            bool,
        ):
            raise TypeError("ratio_norm and grad_reweight must be bool values")
        if self.grad_reweight and not self.ratio_norm:
            raise ValueError("grad_reweight is only defined when ratio_norm is enabled")
        object.__setattr__(self, "clip_range", clip_range)
        object.__setattr__(self, "velocity_mse_weight", velocity_mse_weight)


def parse_bagel_flow_unigrpo_algorithm(
    value: object,
) -> BagelFlowUniGRPOAlgorithmSpec:
    return BagelFlowUniGRPOAlgorithmSpec(
        **parse_flow_policy_fields(
            value,
            allowed=BAGEL_FLOW_UNIGRPO_ALGORITHM_FIELDS,
        )
    )


__all__ = [
    "BagelFlowUniGRPOAlgorithmSpec",
    "parse_bagel_flow_unigrpo_algorithm",
]
