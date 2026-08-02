"""Typed GRPO-Guard session over the shared flow-policy lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from ....rewards.scalarization import RewardScalarizationResult
from ...contracts import FlowTrajectory
from ..flow_policy.session import (
    FlowPolicyIterationResult,
    NativeFlowPolicyTrainingSession,
)
from .engine import GRPOGuardStepResult, NativeGRPOGuardEngine


@dataclass(frozen=True, slots=True)
class GRPOGuardIterationResult(FlowPolicyIterationResult):
    trajectory: FlowTrajectory
    rewards: RewardScalarizationResult
    updates: tuple[GRPOGuardStepResult, ...]


class NativeGRPOGuardTrainingSession(NativeFlowPolicyTrainingSession):
    """Bind GRPO-Guard result and event metrics to shared execution."""

    engine_type = NativeGRPOGuardEngine
    iteration_result_type = GRPOGuardIterationResult
    event_schema = "worldfoundry-grpo-guard-step-event"
    event_metric_names = (
        "ppo_kl",
        "approx_kl",
        "ratio_mean_bias",
        "scale",
        "sqrt_dt_mean",
        "clip_fraction",
    )


__all__ = ["GRPOGuardIterationResult", "NativeGRPOGuardTrainingSession"]
