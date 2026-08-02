"""Typed Bagel Flow-UniGRPO session over shared flow-policy execution."""

from __future__ import annotations

from dataclasses import dataclass

from ....rewards.scalarization import RewardScalarizationResult
from ...contracts import FlowTrajectory
from ..flow_policy.session import (
    FlowPolicyIterationResult,
    NativeFlowPolicyTrainingSession,
)
from .engine import BagelFlowUniGRPOStepResult, NativeBagelFlowUniGRPOEngine


@dataclass(frozen=True, slots=True)
class BagelFlowUniGRPOIterationResult(FlowPolicyIterationResult):
    trajectory: FlowTrajectory
    rewards: RewardScalarizationResult
    updates: tuple[BagelFlowUniGRPOStepResult, ...]


class NativeBagelFlowUniGRPOTrainingSession(NativeFlowPolicyTrainingSession):
    engine_type = NativeBagelFlowUniGRPOEngine
    iteration_result_type = BagelFlowUniGRPOIterationResult
    event_schema = "worldfoundry-bagel-flow-unigrpo-step-event"
    event_metric_names = (
        "surrogate_loss",
        "velocity_mse",
        "approx_kl",
        "clip_fraction",
        "raw_ratio_mean",
        "ratio_mean_bias",
    )


__all__ = [
    "BagelFlowUniGRPOIterationResult",
    "NativeBagelFlowUniGRPOTrainingSession",
]
