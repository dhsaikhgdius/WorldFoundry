"""Recipe-bound construction of native token PPO training stacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.token_ppo import (
    TokenPPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ....rewards.scalarization import WeightedRewardScalarizer
from ....shared.building import (
    build_post_training_optimizer,
    validate_post_training_recipe,
)
from ....shared.distributed import PostTrainingParallelContext
from .contracts import (
    TokenPPOReplayAdapter,
    TokenPPORolloutAdapter,
    TokenPPOTerminalRewardAdapter,
)
from .engine import NativeTokenPPOEngine, TokenPPOStepResult
from .session import NativeTokenPPOTrainingSession


@dataclass(frozen=True, slots=True)
class NativeTokenPPOTrainingStack:
    """Complete actor-critic optimizer stack before run-state materialization."""

    rollout_adapter: TokenPPORolloutAdapter
    replay_adapter: TokenPPOReplayAdapter
    reward_adapter: TokenPPOTerminalRewardAdapter
    scalarizer: WeightedRewardScalarizer
    optimizer: torch.optim.Optimizer
    engine: NativeTokenPPOEngine
    sampling_temperature: float

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        algorithm_state: object = self.scalarizer
        if callable(getattr(self.rollout_adapter, "state_dict", None)) and callable(
            getattr(self.rollout_adapter, "load_state_dict", None)
        ):
            algorithm_state = NamedStatefulCollection(
                {
                    "reward_scalarizer": self.scalarizer,
                    "rollout": self.rollout_adapter,
                }
            )
        return {
            "lr_scheduler": None,
            "ema": None,
            "algorithm_state": algorithm_state,
        }

    def build_session(
        self,
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        step_sink: Callable[[TokenPPOStepResult], None] | None = None,
    ) -> NativeTokenPPOTrainingSession:
        return NativeTokenPPOTrainingSession(
            rollout_adapter=self.rollout_adapter,
            reward_adapter=self.reward_adapter,
            scalarizer=self.scalarizer,
            engine=self.engine,
            progress=progress,
            sampling_temperature=self.sampling_temperature,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=save_every_steps,
            asynchronous_checkpoints=asynchronous_checkpoints,
            step_sink=step_sink,
        )


def build_native_token_ppo_training_stack(
    recipe: PostTrainingRecipe,
    *,
    rollout_adapter: TokenPPORolloutAdapter,
    replay_adapter: TokenPPOReplayAdapter,
    reward_adapter: TokenPPOTerminalRewardAdapter,
    initial_policy_revision: str,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeTokenPPOTrainingStack:
    """Build classic PPO without routing through a GRPO implementation."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, TokenPPOAlgorithmSpec):
        raise TypeError("token PPO stack requires a TokenPPOAlgorithmSpec recipe")
    if not isinstance(rollout_adapter, TokenPPORolloutAdapter):
        raise TypeError("rollout_adapter must implement TokenPPORolloutAdapter")
    if not isinstance(replay_adapter, TokenPPOReplayAdapter):
        raise TypeError("replay_adapter must implement TokenPPOReplayAdapter")
    if not isinstance(reward_adapter, TokenPPOTerminalRewardAdapter):
        raise TypeError("reward_adapter must implement TokenPPOTerminalRewardAdapter")
    if not isinstance(replay_adapter.module, torch.nn.Module):
        raise TypeError("replay_adapter.module must be an nn.Module")
    reward_ids = tuple(reward_adapter.reward_ids)
    if len(reward_ids) != len(set(reward_ids)) or any(not value.strip() for value in reward_ids):
        raise ValueError("reward_adapter.reward_ids must be unique non-empty strings")
    if set(reward_ids) != set(recipe.algorithm.reward_weights):
        raise ValueError("PPO reward adapter ids differ from reward_weights")

    optimizer = build_post_training_optimizer(
        recipe.optimizer,
        replay_adapter.module,
        fused=fused_adamw,
        role="token PPO actor-critic",
    )
    scalarizer = WeightedRewardScalarizer(recipe.algorithm.reward_weights)
    engine = NativeTokenPPOEngine(
        replay_adapter,
        optimizer,
        initial_policy_revision=initial_policy_revision,
        update_epochs=recipe.algorithm.update_epochs,
        update_partitions=recipe.algorithm.update_partitions,
        replay_microbatch_size=recipe.algorithm.replay_microbatch_size,
        clip_range=recipe.algorithm.clip_range,
        clip_range_high=recipe.algorithm.clip_range_high,
        clip_schedule=recipe.algorithm.clip_schedule,
        clip_schedule_steps=recipe.algorithm.clip_schedule_steps,
        value_clip_range=recipe.algorithm.value_clip_range,
        vf_coef=recipe.algorithm.vf_coef,
        gamma=recipe.algorithm.gamma,
        gae_lambda=recipe.algorithm.gae_lambda,
        reduction=recipe.algorithm.reduction,
        horizon=recipe.algorithm.horizon,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        parallel_context=parallel_context,
    )
    return NativeTokenPPOTrainingStack(
        rollout_adapter=rollout_adapter,
        replay_adapter=replay_adapter,
        reward_adapter=reward_adapter,
        scalarizer=scalarizer,
        optimizer=optimizer,
        engine=engine,
        sampling_temperature=recipe.algorithm.sampling_temperature,
    )


__all__ = [
    "NativeTokenPPOTrainingStack",
    "build_native_token_ppo_training_stack",
]
