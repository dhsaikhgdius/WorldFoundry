"""Flow-DPPO session wrapper over the shared flow-policy lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from ....rewards.scalarization import RewardScalarizationResult
from ...contracts import FlowTrajectory
from ..flow_policy.session import (
    FlowPolicyIterationResult,
    NativeFlowPolicyTrainingSession,
)
from .engine import FlowDPPOStepResult, NativeFlowDPPOEngine


@dataclass(frozen=True, slots=True)
class FlowDPPOIterationResult(FlowPolicyIterationResult):
    trajectory: FlowTrajectory
    rewards: RewardScalarizationResult
    updates: tuple[FlowDPPOStepResult, ...]


class NativeFlowDPPOTrainingSession(NativeFlowPolicyTrainingSession):
    """Bind Flow-DPPO's engine, result, and event metrics to shared execution."""

    engine_type = NativeFlowDPPOEngine
    iteration_result_type = FlowDPPOIterationResult
    event_schema = "worldfoundry-flow-dppo-step-event"
    event_metric_names = (
        "approx_kl",
        "old_policy_kl",
        "masked_fraction",
        "positive_masked_fraction",
        "negative_masked_fraction",
    )


__all__ = ["FlowDPPOIterationResult", "NativeFlowDPPOTrainingSession"]
