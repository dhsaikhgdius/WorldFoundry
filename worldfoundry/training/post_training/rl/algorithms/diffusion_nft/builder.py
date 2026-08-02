"""Construction of the native DiffusionNFT collection and update stack."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.diffusion_nft import (
    DiffusionNFTAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ....rewards.scalarization import WeightedRewardScalarizer
from ....shared.building import (
    build_post_training_optimizer,
    prediction_module,
    resolve_tensor_dtype,
    validate_post_training_recipe,
)
from ....shared.contracts import FlowPredictionAdapter
from ....shared.distributed import PostTrainingParallelContext
from ....shared.prediction import NativeClassifierFreeGuidance, NativeFlowPredictionAdapter
from ...contracts import FlowRolloutBatch
from .collection import NativeDiffusionNFTTerminalCollector
from .contracts import DiffusionNFTRewardAdapter, OldPolicyRefresh
from .engine import NativeDiffusionNFTEngine
from .session import NativeDiffusionNFTTrainingSession


@dataclass(frozen=True, slots=True)
class NativeDiffusionNFTTrainingStack:
    """Recipe-materialized terminal collection, reward, and optimizer plane."""

    collector: NativeDiffusionNFTTerminalCollector
    reward_adapter: DiffusionNFTRewardAdapter
    scalarizer: WeightedRewardScalarizer
    algorithm_state: NamedStatefulCollection
    optimizer: torch.optim.Optimizer
    engine: NativeDiffusionNFTEngine
    group_size: int
    sigmas: tuple[float, ...]

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": None,
            "ema": None,
            "algorithm_state": self.algorithm_state,
        }

    def build_session(
        self,
        dataloader: Iterable[FlowRolloutBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> NativeDiffusionNFTTrainingSession:
        return NativeDiffusionNFTTrainingSession(
            self.engine,
            dataloader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=save_every_steps,
            asynchronous_checkpoints=asynchronous_checkpoints,
            event_sink=event_sink,
            collector=self.collector,
            reward_adapter=self.reward_adapter,
            scalarizer=self.scalarizer,
        )


def _collection_policy(
    old_policy: FlowPredictionAdapter,
    *,
    guidance_scale: float,
) -> FlowPredictionAdapter:
    if guidance_scale == 1:
        return old_policy
    if not isinstance(old_policy, NativeFlowPredictionAdapter):
        raise TypeError("DiffusionNFT classifier-free collection requires NativeFlowPredictionAdapter")
    return NativeClassifierFreeGuidance(
        old_policy,
        guidance_scale=guidance_scale,
    )


def build_native_diffusion_nft_training_stack(
    recipe: PostTrainingRecipe,
    *,
    policy: FlowPredictionAdapter,
    old_policy: FlowPredictionAdapter,
    initial_old_policy_revision: str,
    reward_adapter: DiffusionNFTRewardAdapter,
    reference_policy: FlowPredictionAdapter | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeDiffusionNFTTrainingStack:
    """Build DiffusionNFT from loaded model roles and a terminal reward adapter."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, DiffusionNFTAlgorithmSpec):
        raise TypeError("DiffusionNFT stack requires a DiffusionNFTAlgorithmSpec recipe")
    if not isinstance(reward_adapter, DiffusionNFTRewardAdapter):
        raise TypeError("reward_adapter must implement DiffusionNFTRewardAdapter")
    algorithm = recipe.algorithm
    expected_reward_ids = tuple(algorithm.reward_model.reward_ids)
    if tuple(reward_adapter.reward_ids) != expected_reward_ids:
        raise ValueError("DiffusionNFT reward adapter ids differ from reward_model.reward_ids")
    policy_module = prediction_module(policy, role="DiffusionNFT policy")
    old_policy_module = prediction_module(old_policy, role="DiffusionNFT old policy")
    if algorithm.reference_mse_weight > 0 and reference_policy is None:
        raise ValueError("positive reference_mse_weight requires a frozen reference_policy")
    if algorithm.reference_mse_weight == 0 and reference_policy is not None:
        raise ValueError("reference_policy is unused when reference_mse_weight is zero")
    if reference_policy is not None:
        reference_module = prediction_module(
            reference_policy,
            role="DiffusionNFT reference policy",
        )
        if any(parameter.requires_grad for parameter in reference_module.parameters()):
            raise ValueError("DiffusionNFT reference policy parameters must be frozen")

    optimizer = build_post_training_optimizer(
        recipe.optimizer,
        policy_module,
        fused=fused_adamw,
        role="DiffusionNFT policy",
    )
    scalarizer = WeightedRewardScalarizer(
        algorithm.reward_weights,
        calibration_mean=algorithm.reward_model.calibration_mean,
        calibration_std=algorithm.reward_model.calibration_std,
        normalization_epsilon=algorithm.reward_model.normalization_epsilon,
    )
    collection = algorithm.collection
    collector = NativeDiffusionNFTTerminalCollector(
        _collection_policy(
            old_policy,
            guidance_scale=collection.guidance_scale,
        ),
        sigmas=collection.sigmas,
        group_size=collection.group_size,
        latent_dtype=resolve_tensor_dtype(collection.latent_dtype),
        forward_batch_size=collection.forward_batch_size,
    )
    engine = NativeDiffusionNFTEngine(
        policy,
        old_policy,
        optimizer,
        initial_old_policy_revision=initial_old_policy_revision,
        beta=algorithm.beta,
        advantage_clip_max=algorithm.advantage_clip_max,
        advantage_epsilon=algorithm.advantage_epsilon,
        advantage_normalization=algorithm.advantage_normalization,
        advantage_mode=algorithm.advantage_mode,
        reference_policy=reference_policy,
        reference_mse_weight=algorithm.reference_mse_weight,
        reconstruction_mae_floor=algorithm.reconstruction_mae_floor,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        old_policy_refresh=OldPolicyRefresh(
            schedule=algorithm.old_policy_refresh.decay,
            update_interval=algorithm.old_policy_refresh.interval,
        ),
        parallel_context=parallel_context,
    )
    algorithm_state = NamedStatefulCollection(
        {
            "old_policy": old_policy_module,
            "reward_scalarizer": scalarizer,
        }
    )
    return NativeDiffusionNFTTrainingStack(
        collector=collector,
        reward_adapter=reward_adapter,
        scalarizer=scalarizer,
        algorithm_state=algorithm_state,
        optimizer=optimizer,
        engine=engine,
        group_size=collection.group_size,
        sigmas=collection.sigmas,
    )


__all__ = [
    "NativeDiffusionNFTTrainingStack",
    "build_native_diffusion_nft_training_stack",
]
