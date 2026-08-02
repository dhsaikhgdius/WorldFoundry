"""Native MixGRPO engine binding over the shared flow-policy learner."""

from __future__ import annotations

import torch

from worldfoundry.training.recipes.post_training.algorithms.mix_grpo import (
    MixGRPOAlgorithmSpec,
)

from ....shared.distributed import PostTrainingParallelContext
from ...contracts import FlowTrajectoryReplayAdapter
from ..flow_grpo.engine import NativeFlowGRPOEngine
from ..flow_policy.engine import NativeFlowPolicyEngine
from .algorithm import MixGRPOStageAlgorithm

MIX_GRPO_ENGINE_STATE_SCHEMA = "worldfoundry-mix-grpo-engine"


class NativeMixGRPOEngine(NativeFlowGRPOEngine):
    """Use MixGRPO advantages with the shared clipped replay runtime."""

    def __init__(
        self,
        replay_adapter: FlowTrajectoryReplayAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        initial_policy_revision: str,
        clip_range: float = 1.0e-4,
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
            algorithm=MixGRPOStageAlgorithm(clip_range=clip_range),
            initial_policy_revision=initial_policy_revision,
            state_schema=MIX_GRPO_ENGINE_STATE_SCHEMA,
            display_name="MixGRPO",
            anchor_schema="worldfoundry-mix-grpo-anchor",
            max_grad_norm=max_grad_norm,
            updates_per_trajectory=updates_per_trajectory,
            reference_replay_adapter=reference_replay_adapter,
            reference_kl_weight=reference_kl_weight,
            parallel_context=parallel_context,
            replay_microbatch_size=replay_microbatch_size,
        )


def build_native_mix_grpo_engine(
    algorithm: MixGRPOAlgorithmSpec,
    replay_adapter: FlowTrajectoryReplayAdapter,
    optimizer: torch.optim.Optimizer,
    **kwargs: object,
) -> NativeMixGRPOEngine:
    if not isinstance(algorithm, MixGRPOAlgorithmSpec):
        raise TypeError("MixGRPO engine factory requires MixGRPOAlgorithmSpec")
    return NativeMixGRPOEngine(
        replay_adapter,
        optimizer,
        clip_range=algorithm.clip_range,
        **kwargs,
    )


__all__ = [
    "MIX_GRPO_ENGINE_STATE_SCHEMA",
    "NativeMixGRPOEngine",
    "build_native_mix_grpo_engine",
]
