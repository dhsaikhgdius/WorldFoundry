"""Flow-GRPO session wrapper over the shared flow-policy lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from ....rewards.scalarization import RewardScalarizationResult
from ...contracts import FlowTrajectory
from ..flow_policy.session import (
    FlowPolicyIterationResult,
    NativeFlowPolicyTrainingSession,
)
from .engine import FlowGRPOStepResult, NativeFlowGRPOEngine


@dataclass(frozen=True, slots=True)
class FlowGRPOIterationResult(FlowPolicyIterationResult):
    trajectory: FlowTrajectory
    rewards: RewardScalarizationResult
    updates: tuple[FlowGRPOStepResult, ...]


class NativeFlowGRPOTrainingSession(NativeFlowPolicyTrainingSession):
    """Bind the Flow-GRPO engine, result, and event contract to shared execution."""

    engine_type = NativeFlowGRPOEngine
    iteration_result_type = FlowGRPOIterationResult
    event_schema = "worldfoundry-flow-grpo-step-event"
    event_metric_names = ("approx_kl", "clip_fraction")


__all__ = ["FlowGRPOIterationResult", "NativeFlowGRPOTrainingSession"]
