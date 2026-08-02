"""Construction of the native DDRL rollout, reward, and update stack."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.recipes.post_training.algorithms.ddrl import (
    DDRLAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ....rewards.scalarization import WeightedRewardScalarizer
from ....shared.building import (
    build_post_training_optimizer,
    validate_post_training_recipe,
)
from ....shared.distributed import PostTrainingParallelContext
from .contracts import (
    DDRLDataRegularizerAdapter,
    DDRLReplayAdapter,
    DDRLRewardAdapter,
    DDRLRolloutAdapter,
)
from .engine import NativeDDRLEngine
from .session import NativeDDRLTrainingSession


@dataclass(frozen=True, slots=True)
class NativeDDRLTrainingStack:
    """Loaded adapters, scalarization, optimizer, and exact-resume engine."""

    rollout_adapter: DDRLRolloutAdapter
    replay_adapter: DDRLReplayAdapter
    reward_adapter: DDRLRewardAdapter
    data_regularizer: DDRLDataRegularizerAdapter | None
    scalarizer: WeightedRewardScalarizer
    optimizer: torch.optim.Optimizer
    engine: NativeDDRLEngine
    train_on: tuple[int, ...]

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": None,
            "ema": None,
            "algorithm_state": self.scalarizer,
        }

    def build_session(
        self,
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> NativeDDRLTrainingSession:
        return NativeDDRLTrainingSession(
            rollout_adapter=self.rollout_adapter,
            reward_adapter=self.reward_adapter,
            scalarizer=self.scalarizer,
            engine=self.engine,
            progress=progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=save_every_steps,
            asynchronous_checkpoints=asynchronous_checkpoints,
            event_sink=event_sink,
            expected_train_on=self.train_on,
        )


def build_native_ddrl_training_stack(
    recipe: PostTrainingRecipe,
    *,
    rollout_adapter: DDRLRolloutAdapter,
    replay_adapter: DDRLReplayAdapter,
    reward_adapter: DDRLRewardAdapter,
    data_regularizer: DDRLDataRegularizerAdapter | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeDDRLTrainingStack:
    """Build DDRL from loaded collection, replay, reward, and data seams."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, DDRLAlgorithmSpec):
        raise TypeError("DDRL stack requires a DDRLAlgorithmSpec recipe")
    if not isinstance(rollout_adapter, DDRLRolloutAdapter):
        raise TypeError("rollout_adapter must implement DDRLRolloutAdapter")
    if not isinstance(replay_adapter, DDRLReplayAdapter):
        raise TypeError("replay_adapter must implement DDRLReplayAdapter")
    if not isinstance(reward_adapter, DDRLRewardAdapter):
        raise TypeError("reward_adapter must implement DDRLRewardAdapter")
    policy_module = replay_adapter.module
    if not isinstance(policy_module, torch.nn.Module):
        raise TypeError("replay_adapter.module must be an nn.Module")

    algorithm = recipe.algorithm
    if tuple(reward_adapter.reward_ids) != tuple(algorithm.reward_model.reward_ids):
        raise ValueError("DDRL reward adapter ids differ from reward_model.reward_ids")
    if algorithm.data_beta > 0 and not isinstance(
        data_regularizer,
        DDRLDataRegularizerAdapter,
    ):
        raise TypeError("positive DDRL data_beta requires a data_regularizer adapter")
    if algorithm.data_beta == 0 and data_regularizer is not None:
        raise ValueError("DDRL data_regularizer is unused when data_beta is zero")

    optimizer = build_post_training_optimizer(
        recipe.optimizer,
        policy_module,
        fused=fused_adamw,
        role="DDRL policy",
    )
    scalarizer = WeightedRewardScalarizer(
        {reward_id: algorithm.reward_weights[reward_id] for reward_id in algorithm.reward_model.reward_ids},
        calibration_mean=algorithm.reward_model.calibration_mean,
        calibration_std=algorithm.reward_model.calibration_std,
        normalization_epsilon=algorithm.reward_model.normalization_epsilon,
    )
    engine = NativeDDRLEngine(
        replay_adapter,
        optimizer,
        clip_range=algorithm.clip_range,
        loss_scale=algorithm.loss_scale,
        advantage_epsilon=algorithm.advantage_epsilon,
        advantage_normalization=algorithm.advantage_normalization,
        advantage_clip_min=algorithm.advantage_clip_min,
        advantage_clip_max=algorithm.advantage_clip_max,
        exponential_advantage=algorithm.exponential_advantage,
        kl_beta=algorithm.kl_beta,
        data_beta=algorithm.data_beta,
        data_regularizer=data_regularizer,
        data_on_first_step_only=algorithm.data_on_first_step_only,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        parallel_context=parallel_context,
    )
    return NativeDDRLTrainingStack(
        rollout_adapter=rollout_adapter,
        replay_adapter=replay_adapter,
        reward_adapter=reward_adapter,
        data_regularizer=data_regularizer,
        scalarizer=scalarizer,
        optimizer=optimizer,
        engine=engine,
        train_on=algorithm.train_on,
    )


__all__ = ["NativeDDRLTrainingStack", "build_native_ddrl_training_stack"]
