"""Flow-DPPO configuration wrapper over the shared flow-policy learner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from worldfoundry.training.recipes.post_training.algorithms.flow_dppo import (
    FlowDPPOAlgorithmSpec,
)

from ....shared.distributed import PostTrainingParallelContext
from ...contracts import FlowTrajectoryReplayAdapter
from ..flow_policy.engine import FlowPolicyStepResult, NativeFlowPolicyEngine
from .algorithm import FlowDPPOStageAlgorithm

FLOW_DPPO_ENGINE_STATE_SCHEMA = "worldfoundry-flow-dppo-engine"


@dataclass(frozen=True, slots=True)
class FlowDPPOStepResult:
    """One Flow-DPPO update with KL-mask metrics in ``metrics``."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_kl: torch.Tensor | None
    metrics: Mapping[str, object]
    trajectory_complete: bool
    sample_count: int
    token_count: int
    replay_microbatches: int


class NativeFlowDPPOEngine(NativeFlowPolicyEngine):
    """Bind Flow-DPPO's KL-advantage mask to shared replay execution."""

    def __init__(
        self,
        replay_adapter: FlowTrajectoryReplayAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        initial_policy_revision: str,
        kl_mask_threshold: float = 1.0e-5,
        add_kl_coefficient: bool = True,
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
            algorithm=FlowDPPOStageAlgorithm(
                kl_mask_threshold=kl_mask_threshold,
                add_kl_coefficient=add_kl_coefficient,
            ),
            initial_policy_revision=initial_policy_revision,
            state_schema=FLOW_DPPO_ENGINE_STATE_SCHEMA,
            display_name="Flow-DPPO",
            anchor_schema="worldfoundry-flow-dppo-anchor",
            max_grad_norm=max_grad_norm,
            updates_per_trajectory=updates_per_trajectory,
            reference_replay_adapter=reference_replay_adapter,
            reference_kl_weight=reference_kl_weight,
            parallel_context=parallel_context,
            replay_microbatch_size=replay_microbatch_size,
        )

    @property
    def kl_mask_threshold(self) -> float:
        algorithm = self.algorithm
        assert isinstance(algorithm, FlowDPPOStageAlgorithm)
        return algorithm.kl_mask_threshold

    @property
    def add_kl_coefficient(self) -> bool:
        algorithm = self.algorithm
        assert isinstance(algorithm, FlowDPPOStageAlgorithm)
        return algorithm.add_kl_coefficient

    def train_step(self, *, anchor_id: str) -> FlowDPPOStepResult:
        result: FlowPolicyStepResult = super().train_step(anchor_id=anchor_id)
        return FlowDPPOStepResult(
            loss=result.loss,
            policy_loss=result.policy_loss,
            reference_kl=result.reference_kl,
            metrics=result.metrics,
            trajectory_complete=result.trajectory_complete,
            sample_count=result.sample_count,
            token_count=result.token_count,
            replay_microbatches=result.replay_microbatches,
        )


def build_native_flow_dppo_engine(
    algorithm: FlowDPPOAlgorithmSpec,
    replay_adapter: FlowTrajectoryReplayAdapter,
    optimizer: torch.optim.Optimizer,
    **kwargs: object,
) -> NativeFlowDPPOEngine:
    """Inject Flow-DPPO objective fields into the shared policy engine."""

    if not isinstance(algorithm, FlowDPPOAlgorithmSpec):
        raise TypeError("Flow-DPPO engine factory requires FlowDPPOAlgorithmSpec")
    return NativeFlowDPPOEngine(
        replay_adapter,
        optimizer,
        kl_mask_threshold=algorithm.kl_mask_threshold,
        add_kl_coefficient=algorithm.add_kl_coefficient,
        **kwargs,
    )


__all__ = [
    "FLOW_DPPO_ENGINE_STATE_SCHEMA",
    "FlowDPPOStepResult",
    "NativeFlowDPPOEngine",
    "build_native_flow_dppo_engine",
]
