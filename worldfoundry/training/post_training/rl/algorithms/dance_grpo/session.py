"""DANCE rollout and masked-update lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch

from ....rewards.scalarization import RewardScalarizationResult
from ...contracts import FlowTrajectory
from ..flow_grpo.engine import FlowGRPOStepResult
from ..flow_policy.session import (
    FlowPolicyIterationResult,
    NativeFlowPolicyTrainingSession,
)
from .engine import NativeDanceGRPOEngine
from .update_steps import sample_dance_update_step_mask


@dataclass(frozen=True, slots=True)
class DanceGRPOIterationResult(FlowPolicyIterationResult):
    trajectory: FlowTrajectory
    rewards: RewardScalarizationResult
    updates: tuple[FlowGRPOStepResult, ...]


class NativeDanceGRPOTrainingSession(NativeFlowPolicyTrainingSession):
    """Attach DANCE's random per-sample update mask before replay."""

    engine_type = NativeDanceGRPOEngine
    iteration_result_type = DanceGRPOIterationResult
    event_schema = "worldfoundry-dance-grpo-step-event"
    event_metric_names = ("approx_kl", "clip_fraction")

    def __init__(self, *, update_timestep_fraction: float, **kwargs: object) -> None:
        super().__init__(**kwargs)
        fraction = float(update_timestep_fraction)
        if fraction != self.engine.update_timestep_fraction:
            raise ValueError("DANCE session update fraction differs from its engine")
        if self.advantage_normalization != "group-sample-std":
            raise ValueError("DANCE session requires group-sample-std advantages")
        self.update_timestep_fraction = fraction

    def _prepare_engine_trajectory(
        self,
        trajectory: FlowTrajectory,
        components: Mapping[str, object],
        rewards: RewardScalarizationResult,
        *,
        generator: torch.Generator | None,
    ) -> tuple[FlowTrajectory, str]:
        mask = sample_dance_update_step_mask(
            batch_size=trajectory.batch_size,
            transition_count=len(trajectory.step_indices),
            timestep_fraction=self.update_timestep_fraction,
            device=trajectory.old_log_probs.device,
            generator=generator,
        )
        masked = replace(trajectory, update_step_mask=mask)
        return super()._prepare_engine_trajectory(
            masked,
            components,
            rewards,
            generator=generator,
        )


__all__ = ["DanceGRPOIterationResult", "NativeDanceGRPOTrainingSession"]
