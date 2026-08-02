"""GRPO-Guard engine binding over shared flow-policy execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from worldfoundry.training.recipes.post_training.algorithms.grpo_guard import (
    GRPOGuardAlgorithmSpec,
)

from ....shared.distributed import PostTrainingParallelContext
from ...contracts import FlowTrajectoryReplayAdapter
from ..flow_policy.engine import FlowPolicyStepResult, NativeFlowPolicyEngine
from .algorithm import GRPOGuardStageAlgorithm

GRPO_GUARD_ENGINE_STATE_SCHEMA = "worldfoundry-grpo-guard-engine"


@dataclass(frozen=True, slots=True)
class GRPOGuardStepResult:
    """One GRPO-Guard optimizer update."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_kl: torch.Tensor | None
    metrics: Mapping[str, object]
    trajectory_complete: bool
    sample_count: int
    token_count: int
    replay_microbatches: int


class NativeGRPOGuardEngine(NativeFlowPolicyEngine):
    """Bind GRPO-Guard's mean-bias stage to shared replay execution."""

    def __init__(
        self,
        replay_adapter: FlowTrajectoryReplayAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        initial_policy_revision: str,
        clip_range: float = 1.0e-4,
        advantage_clip_max: float = 5.0,
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
            algorithm=GRPOGuardStageAlgorithm(
                clip_range=clip_range,
                advantage_clip_max=advantage_clip_max,
            ),
            initial_policy_revision=initial_policy_revision,
            state_schema=GRPO_GUARD_ENGINE_STATE_SCHEMA,
            display_name="GRPO-Guard",
            anchor_schema="worldfoundry-grpo-guard-anchor",
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
        assert isinstance(algorithm, GRPOGuardStageAlgorithm)
        return algorithm.clip_range

    @property
    def advantage_clip_max(self) -> float:
        algorithm = self.algorithm
        assert isinstance(algorithm, GRPOGuardStageAlgorithm)
        return algorithm.advantage_clip_max

    def train_step(self, *, anchor_id: str) -> GRPOGuardStepResult:
        result: FlowPolicyStepResult = super().train_step(anchor_id=anchor_id)
        return GRPOGuardStepResult(
            loss=result.loss,
            policy_loss=result.policy_loss,
            reference_kl=result.reference_kl,
            metrics=result.metrics,
            trajectory_complete=result.trajectory_complete,
            sample_count=result.sample_count,
            token_count=result.token_count,
            replay_microbatches=result.replay_microbatches,
        )


def build_native_grpo_guard_engine(
    algorithm: GRPOGuardAlgorithmSpec,
    replay_adapter: FlowTrajectoryReplayAdapter,
    optimizer: torch.optim.Optimizer,
    **kwargs: object,
) -> NativeGRPOGuardEngine:
    """Inject GRPO-Guard objective fields into the shared policy engine."""

    if not isinstance(algorithm, GRPOGuardAlgorithmSpec):
        raise TypeError("GRPO-Guard engine factory requires GRPOGuardAlgorithmSpec")
    return NativeGRPOGuardEngine(
        replay_adapter,
        optimizer,
        clip_range=algorithm.clip_range,
        advantage_clip_max=algorithm.advantage_clip_max,
        **kwargs,
    )


__all__ = [
    "GRPO_GUARD_ENGINE_STATE_SCHEMA",
    "GRPOGuardStepResult",
    "NativeGRPOGuardEngine",
    "build_native_grpo_guard_engine",
]
