"""Flow-GRPO configuration wrapper over the shared flow-policy learner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from worldfoundry.training.recipes.post_training.algorithms.flow_grpo import (
    FlowGRPOAlgorithmSpec,
)

from ....shared.distributed import PostTrainingParallelContext
from ...contracts import FlowTrajectoryReplayAdapter
from ..flow_policy.engine import FlowPolicyStepResult, NativeFlowPolicyEngine
from .algorithm import FlowGRPOStageAlgorithm

FLOW_GRPO_ENGINE_STATE_SCHEMA = "worldfoundry-flow-grpo-engine"


@dataclass(frozen=True, slots=True)
class FlowGRPOStepResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_kl: torch.Tensor | None
    metrics: Mapping[str, object]
    trajectory_complete: bool
    sample_count: int
    token_count: int
    replay_microbatches: int


class NativeFlowGRPOEngine(NativeFlowPolicyEngine):
    """Preserve the Flow-GRPO API while supplying only its loss stage."""

    def __init__(
        self,
        replay_adapter: FlowTrajectoryReplayAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        initial_policy_revision: str,
        clip_range: float = 1.0e-4,
        clip_schedule: str = "constant",
        clip_schedule_steps: int | None = None,
        max_grad_norm: float = 1.0,
        updates_per_trajectory: int = 1,
        reference_replay_adapter: FlowTrajectoryReplayAdapter | None = None,
        reference_kl_weight: float = 0.0,
        parallel_context: PostTrainingParallelContext | None = None,
        replay_microbatch_size: int | None = None,
    ) -> None:
        super().__init__(
            replay_adapter,
            optimizer,
            algorithm=FlowGRPOStageAlgorithm(
                clip_range=clip_range,
                clip_schedule=clip_schedule,
                clip_schedule_steps=clip_schedule_steps,
            ),
            initial_policy_revision=initial_policy_revision,
            state_schema=FLOW_GRPO_ENGINE_STATE_SCHEMA,
            display_name="Flow-GRPO",
            anchor_schema="worldfoundry-flow-grpo-anchor",
            max_grad_norm=max_grad_norm,
            updates_per_trajectory=updates_per_trajectory,
            reference_replay_adapter=reference_replay_adapter,
            reference_kl_weight=reference_kl_weight,
            parallel_context=parallel_context,
            replay_microbatch_size=replay_microbatch_size,
        )

    @property
    def clip_range(self) -> float:
        algorithm = self.algorithm
        assert isinstance(algorithm, FlowGRPOStageAlgorithm)
        return algorithm.clip_range

    def train_step(self, *, anchor_id: str) -> FlowGRPOStepResult:
        result: FlowPolicyStepResult = super().train_step(anchor_id=anchor_id)
        return FlowGRPOStepResult(
            loss=result.loss,
            policy_loss=result.policy_loss,
            reference_kl=result.reference_kl,
            metrics=result.metrics,
            trajectory_complete=result.trajectory_complete,
            sample_count=result.sample_count,
            token_count=result.token_count,
            replay_microbatches=result.replay_microbatches,
        )


def build_native_flow_grpo_engine(
    algorithm: FlowGRPOAlgorithmSpec,
    replay_adapter: FlowTrajectoryReplayAdapter,
    optimizer: torch.optim.Optimizer,
    **kwargs: object,
) -> NativeFlowGRPOEngine:
    """Inject Flow-GRPO objective fields into the shared policy engine."""

    if not isinstance(algorithm, FlowGRPOAlgorithmSpec):
        raise TypeError("Flow-GRPO engine factory requires FlowGRPOAlgorithmSpec")
    return NativeFlowGRPOEngine(
        replay_adapter,
        optimizer,
        clip_range=algorithm.clip_range,
        clip_schedule=algorithm.clip_schedule,
        clip_schedule_steps=algorithm.clip_schedule_steps,
        **kwargs,
    )


__all__ = [
    "FLOW_GRPO_ENGINE_STATE_SCHEMA",
    "FlowGRPOStepResult",
    "NativeFlowGRPOEngine",
    "build_native_flow_grpo_engine",
]
