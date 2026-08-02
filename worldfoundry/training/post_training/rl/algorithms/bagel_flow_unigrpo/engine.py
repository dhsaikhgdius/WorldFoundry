"""Bagel Flow-UniGRPO binding over shared flow-policy execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from worldfoundry.training.recipes.post_training.algorithms.bagel_flow_unigrpo import (
    BagelFlowUniGRPOAlgorithmSpec,
)

from ....shared.distributed import PostTrainingParallelContext
from ...contracts import FlowTrajectoryReplayAdapter
from ..flow_policy.engine import FlowPolicyStepResult, NativeFlowPolicyEngine
from .algorithm import BagelFlowUniGRPOStageAlgorithm

BAGEL_FLOW_UNIGRPO_ENGINE_STATE_SCHEMA = "worldfoundry-bagel-flow-unigrpo-engine"


@dataclass(frozen=True, slots=True)
class BagelFlowUniGRPOStepResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_kl: torch.Tensor | None
    metrics: Mapping[str, object]
    trajectory_complete: bool
    sample_count: int
    token_count: int
    replay_microbatches: int


class NativeBagelFlowUniGRPOEngine(NativeFlowPolicyEngine):
    """Bind velocity-regularized Flow-GRPO to shared replay execution."""

    def __init__(
        self,
        replay_adapter: FlowTrajectoryReplayAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        initial_policy_revision: str,
        reference_replay_adapter: FlowTrajectoryReplayAdapter,
        clip_range: float = 1.0e-4,
        velocity_mse_weight: float = 1.0,
        ratio_norm: bool = False,
        grad_reweight: bool = False,
        max_grad_norm: float = 1.0,
        updates_per_trajectory: int = 1,
        parallel_context: PostTrainingParallelContext | None = None,
        replay_microbatch_size: int | None = None,
    ) -> None:
        super().__init__(
            replay_adapter,
            optimizer,
            algorithm=BagelFlowUniGRPOStageAlgorithm(
                clip_range=clip_range,
                velocity_mse_weight=velocity_mse_weight,
                ratio_norm=ratio_norm,
                grad_reweight=grad_reweight,
            ),
            initial_policy_revision=initial_policy_revision,
            state_schema=BAGEL_FLOW_UNIGRPO_ENGINE_STATE_SCHEMA,
            display_name="Bagel Flow-UniGRPO",
            anchor_schema="worldfoundry-bagel-flow-unigrpo-anchor",
            max_grad_norm=max_grad_norm,
            updates_per_trajectory=updates_per_trajectory,
            reference_replay_adapter=reference_replay_adapter,
            reference_kl_weight=0.0,
            parallel_context=parallel_context,
            replay_microbatch_size=replay_microbatch_size,
        )

    def train_step(self, *, anchor_id: str) -> BagelFlowUniGRPOStepResult:
        result: FlowPolicyStepResult = super().train_step(anchor_id=anchor_id)
        return BagelFlowUniGRPOStepResult(
            loss=result.loss,
            policy_loss=result.policy_loss,
            reference_kl=result.reference_kl,
            metrics=result.metrics,
            trajectory_complete=result.trajectory_complete,
            sample_count=result.sample_count,
            token_count=result.token_count,
            replay_microbatches=result.replay_microbatches,
        )


def build_native_bagel_flow_unigrpo_engine(
    algorithm: BagelFlowUniGRPOAlgorithmSpec,
    replay_adapter: FlowTrajectoryReplayAdapter,
    optimizer: torch.optim.Optimizer,
    **kwargs: object,
) -> NativeBagelFlowUniGRPOEngine:
    if not isinstance(algorithm, BagelFlowUniGRPOAlgorithmSpec):
        raise TypeError("Bagel Flow-UniGRPO engine factory requires BagelFlowUniGRPOAlgorithmSpec")
    reference_kl_weight = float(kwargs.pop("reference_kl_weight", 0.0))
    if reference_kl_weight != 0:
        raise ValueError("Bagel Flow-UniGRPO does not use reference KL")
    return NativeBagelFlowUniGRPOEngine(
        replay_adapter,
        optimizer,
        clip_range=algorithm.clip_range,
        velocity_mse_weight=algorithm.velocity_mse_weight,
        ratio_norm=algorithm.ratio_norm,
        grad_reweight=algorithm.grad_reweight,
        **kwargs,
    )


__all__ = [
    "BAGEL_FLOW_UNIGRPO_ENGINE_STATE_SCHEMA",
    "BagelFlowUniGRPOStepResult",
    "NativeBagelFlowUniGRPOEngine",
    "build_native_bagel_flow_unigrpo_engine",
]
