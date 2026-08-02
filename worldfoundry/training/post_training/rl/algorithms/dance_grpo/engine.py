"""Native DANCE engine binding over the shared flow-policy learner."""

from __future__ import annotations

import torch

from worldfoundry.training.recipes.post_training.algorithms.dance_grpo import (
    DanceGRPOAlgorithmSpec,
)

from ....shared.distributed import PostTrainingParallelContext
from ...contracts import FlowTrajectoryReplayAdapter
from ..flow_grpo.engine import NativeFlowGRPOEngine
from ..flow_policy.engine import NativeFlowPolicyEngine
from .algorithm import DanceGRPOStageAlgorithm

DANCE_GRPO_ENGINE_STATE_SCHEMA = "worldfoundry-dance-grpo-engine"


class NativeDanceGRPOEngine(NativeFlowGRPOEngine):
    """Use DANCE's masked clipped objective with shared replay ownership."""

    def __init__(
        self,
        replay_adapter: FlowTrajectoryReplayAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        initial_policy_revision: str,
        clip_range: float = 1.0e-4,
        update_timestep_fraction: float = 0.6,
        max_grad_norm: float = 1.0,
        updates_per_trajectory: int = 1,
        reference_replay_adapter: FlowTrajectoryReplayAdapter | None = None,
        reference_kl_weight: float = 0.0,
        parallel_context: PostTrainingParallelContext | None = None,
        replay_microbatch_size: int | None = None,
    ) -> None:
        NativeFlowPolicyEngine.__init__(
            self,
            replay_adapter,
            optimizer,
            algorithm=DanceGRPOStageAlgorithm(
                clip_range=clip_range,
                update_timestep_fraction=update_timestep_fraction,
            ),
            initial_policy_revision=initial_policy_revision,
            state_schema=DANCE_GRPO_ENGINE_STATE_SCHEMA,
            display_name="DANCE",
            anchor_schema="worldfoundry-dance-grpo-anchor",
            max_grad_norm=max_grad_norm,
            updates_per_trajectory=updates_per_trajectory,
            reference_replay_adapter=reference_replay_adapter,
            reference_kl_weight=reference_kl_weight,
            parallel_context=parallel_context,
            replay_microbatch_size=replay_microbatch_size,
        )

    @property
    def update_timestep_fraction(self) -> float:
        algorithm = self.algorithm
        assert isinstance(algorithm, DanceGRPOStageAlgorithm)
        return algorithm.update_timestep_fraction


def build_native_dance_grpo_engine(
    algorithm: DanceGRPOAlgorithmSpec,
    replay_adapter: FlowTrajectoryReplayAdapter,
    optimizer: torch.optim.Optimizer,
    **kwargs: object,
) -> NativeDanceGRPOEngine:
    if not isinstance(algorithm, DanceGRPOAlgorithmSpec):
        raise TypeError("DANCE engine factory requires DanceGRPOAlgorithmSpec")
    return NativeDanceGRPOEngine(
        replay_adapter,
        optimizer,
        clip_range=algorithm.clip_range,
        update_timestep_fraction=algorithm.update_timestep_fraction,
        **kwargs,
    )


__all__ = [
    "DANCE_GRPO_ENGINE_STATE_SCHEMA",
    "NativeDanceGRPOEngine",
    "build_native_dance_grpo_engine",
]
