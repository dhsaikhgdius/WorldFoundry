"""Construction of complete native autoregressive policy-training stacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.token_policy import (
    TokenCPPOAlgorithmSpec,
    TokenDPPOAlgorithmSpec,
    TokenDRPOAlgorithmSpec,
    TokenGRPOAlgorithmSpec,
    TokenGSPOAlgorithmSpec,
    TokenPolicyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ....rewards.scalarization import WeightedRewardScalarizer
from ....shared.building import (
    build_post_training_optimizer,
    validate_post_training_recipe,
)
from ....shared.distributed import PostTrainingParallelContext
from .contracts import (
    TokenPolicyReplayAdapter,
    TokenPolicyRolloutAdapter,
    TokenTrajectoryRewardAdapter,
)
from .engine import NativeTokenPolicyEngine, TokenPolicyStepResult
from .runtime import build_token_policy_stage
from .session import NativeTokenPolicyTrainingSession
from .stages import TokenPolicyStage


@dataclass(frozen=True, slots=True)
class NativeTokenPolicyTrainingStack:
    """Recipe-bound rollout, reward, replay, optimizer, and session factory."""

    rollout_adapter: TokenPolicyRolloutAdapter
    replay_adapter: TokenPolicyReplayAdapter
    reward_adapter: TokenTrajectoryRewardAdapter
    scalarizer: WeightedRewardScalarizer
    optimizer: torch.optim.Optimizer
    engine: NativeTokenPolicyEngine
    group_size: int
    advantage_epsilon: float
    advantage_normalization: str
    advantage_clip_max: float | None
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
        step_sink: Callable[[TokenPolicyStepResult], None] | None = None,
    ) -> NativeTokenPolicyTrainingSession:
        return NativeTokenPolicyTrainingSession(
            rollout_adapter=self.rollout_adapter,
            reward_adapter=self.reward_adapter,
            scalarizer=self.scalarizer,
            engine=self.engine,
            progress=progress,
            group_size=self.group_size,
            advantage_epsilon=self.advantage_epsilon,
            advantage_normalization=self.advantage_normalization,
            advantage_clip_max=self.advantage_clip_max,
            sampling_temperature=self.sampling_temperature,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=save_every_steps,
            asynchronous_checkpoints=asynchronous_checkpoints,
            step_sink=step_sink,
        )


def _stage_from_spec(algorithm: TokenPolicyAlgorithmSpec) -> TokenPolicyStage:
    if isinstance(algorithm, TokenGRPOAlgorithmSpec):
        settings = {
            "clip_range": algorithm.clip_range,
            "clip_range_high": algorithm.clip_range_high,
            "clip_schedule": algorithm.clip_schedule,
            "clip_schedule_steps": algorithm.clip_schedule_steps,
            "reduction": algorithm.reduction,
            "horizon": algorithm.horizon,
        }
    elif isinstance(algorithm, TokenGSPOAlgorithmSpec):
        settings = {
            "clip_range": algorithm.clip_range,
            "clip_range_high": algorithm.clip_range_high,
            "clip_schedule": algorithm.clip_schedule,
            "clip_schedule_steps": algorithm.clip_schedule_steps,
        }
    elif isinstance(algorithm, TokenDPPOAlgorithmSpec):
        settings = {
            "delta": algorithm.delta,
            "reduction": algorithm.reduction,
            "horizon": algorithm.horizon,
        }
    elif isinstance(algorithm, TokenDRPOAlgorithmSpec):
        settings = {
            "epsilon": algorithm.epsilon,
            "mu_weighted": algorithm.mu_weighted,
            "reduction": algorithm.reduction,
            "horizon": algorithm.horizon,
        }
    elif isinstance(algorithm, TokenCPPOAlgorithmSpec):
        settings = {
            "delta": algorithm.delta,
            "w_min": algorithm.w_min,
            "delta_b": algorithm.delta_b,
            "reduction": algorithm.reduction,
            "horizon": algorithm.horizon,
        }
    else:
        raise TypeError(f"unsupported token-policy spec: {type(algorithm).__name__}")
    return build_token_policy_stage(
        algorithm.type.removeprefix("token-"),
        settings,
    )


def build_native_token_policy_training_stack(
    recipe: PostTrainingRecipe,
    *,
    rollout_adapter: TokenPolicyRolloutAdapter,
    replay_adapter: TokenPolicyReplayAdapter,
    reward_adapter: TokenTrajectoryRewardAdapter,
    initial_policy_revision: str,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeTokenPolicyTrainingStack:
    """Build a token-policy learner around model-owned rollout/replay adapters."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, TokenPolicyAlgorithmSpec):
        raise TypeError("token-policy stack requires a TokenPolicyAlgorithmSpec recipe")
    if not isinstance(rollout_adapter, TokenPolicyRolloutAdapter):
        raise TypeError("rollout_adapter must implement TokenPolicyRolloutAdapter")
    if not isinstance(replay_adapter, TokenPolicyReplayAdapter):
        raise TypeError("replay_adapter must implement TokenPolicyReplayAdapter")
    if not isinstance(reward_adapter, TokenTrajectoryRewardAdapter):
        raise TypeError("reward_adapter must implement TokenTrajectoryRewardAdapter")
    policy_module = replay_adapter.module
    if not isinstance(policy_module, torch.nn.Module):
        raise TypeError("replay_adapter.module must be an nn.Module")

    algorithm = recipe.algorithm
    reward_ids = tuple(reward_adapter.reward_ids)
    if (
        not reward_ids
        or any(not isinstance(value, str) or not value.strip() for value in reward_ids)
        or len(reward_ids) != len(set(reward_ids))
    ):
        raise ValueError("reward_adapter.reward_ids must be unique non-empty strings")
    if set(reward_ids) != set(algorithm.reward_weights):
        raise ValueError("token-policy reward adapter ids differ from reward_weights")

    optimizer = build_post_training_optimizer(
        recipe.optimizer,
        policy_module,
        fused=fused_adamw,
        role=f"{algorithm.type} policy",
    )
    scalarizer = WeightedRewardScalarizer(algorithm.reward_weights)
    engine = NativeTokenPolicyEngine(
        replay_adapter,
        optimizer,
        algorithm=_stage_from_spec(algorithm),
        initial_policy_revision=initial_policy_revision,
        old_log_prob_source=algorithm.old_log_prob_source,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        updates_per_trajectory=algorithm.updates_per_trajectory,
        first_update_log_ratio_tolerance=(algorithm.first_update_log_ratio_tolerance),
        parallel_context=parallel_context,
        replay_microbatch_size=algorithm.replay_microbatch_size,
    )
    return NativeTokenPolicyTrainingStack(
        rollout_adapter=rollout_adapter,
        replay_adapter=replay_adapter,
        reward_adapter=reward_adapter,
        scalarizer=scalarizer,
        optimizer=optimizer,
        engine=engine,
        group_size=algorithm.group_size,
        advantage_epsilon=algorithm.advantage_epsilon,
        advantage_normalization=algorithm.advantage_normalization,
        advantage_clip_max=algorithm.advantage_clip_max,
        sampling_temperature=algorithm.sampling_temperature,
    )


__all__ = [
    "NativeTokenPolicyTrainingStack",
    "build_native_token_policy_training_stack",
]
