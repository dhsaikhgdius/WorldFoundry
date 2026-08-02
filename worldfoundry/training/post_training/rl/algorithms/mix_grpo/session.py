"""MixGRPO component-first reward and progressive-window lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ....rewards.scalarization import RewardScalarizationResult
from ...contracts import FlowTrajectory
from ...objectives.group_advantages import normalize_weighted_component_advantages
from ..flow_grpo.engine import FlowGRPOStepResult
from ..flow_policy.session import (
    FlowPolicyIterationResult,
    NativeFlowPolicyTrainingSession,
)
from .engine import NativeMixGRPOEngine


@dataclass(frozen=True, slots=True)
class MixGRPOIterationResult(FlowPolicyIterationResult):
    trajectory: FlowTrajectory
    rewards: RewardScalarizationResult
    updates: tuple[FlowGRPOStepResult, ...]


class NativeMixGRPOTrainingSession(NativeFlowPolicyTrainingSession):
    """Prepare component-wise group advantages before policy replay."""

    engine_type = NativeMixGRPOEngine
    iteration_result_type = MixGRPOIterationResult
    event_schema = "worldfoundry-mix-grpo-step-event"
    event_metric_names = ("approx_kl", "clip_fraction")

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.advantage_normalization != "group-sample-std":
            raise ValueError("MixGRPO session requires group-sample-std advantages")
        weights = tuple(self.scalarizer.weights.values())
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("MixGRPO session requires non-negative reward weights with a positive sum")

    def _prepare_engine_trajectory(
        self,
        trajectory: FlowTrajectory,
        components: Mapping[str, object],
        rewards: RewardScalarizationResult,
        *,
        generator: torch.Generator | None,
    ) -> tuple[FlowTrajectory, str]:
        del rewards, generator
        result = normalize_weighted_component_advantages(
            components,
            self.scalarizer.weights,
            trajectory.group_ids,
            parallel_context=self.engine.parallel_context,
            epsilon=self.advantage_epsilon,
            clip_max=self.advantage_clip_max,
            normalization=self.advantage_normalization,
        )
        anchor = self.engine.prepare_trajectory_from_advantages(
            trajectory,
            result.advantages,
            old_log_prob_source=self.old_log_prob_source,
        )
        return trajectory, anchor


__all__ = ["MixGRPOIterationResult", "NativeMixGRPOTrainingSession"]
