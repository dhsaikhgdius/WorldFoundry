"""Rollout, terminal reward, PPO update, and checkpoint lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress

from ....rewards.scalarization import (
    RewardScalarizationResult,
    WeightedRewardScalarizer,
)
from .contracts import (
    PackedTokenPPOTrajectory,
    TokenPPORolloutAdapter,
    TokenPPORolloutRequest,
    TokenPPOTerminalRewardAdapter,
)
from .engine import NativeTokenPPOEngine, TokenPPOAnchor, TokenPPOStepResult


@dataclass(frozen=True, slots=True)
class TokenPPOIterationResult:
    """One rollout, its frozen GAE anchor, and all PPO partition updates."""

    trajectory: PackedTokenPPOTrajectory
    rewards: RewardScalarizationResult
    anchor: TokenPPOAnchor
    updates: tuple[TokenPPOStepResult, ...]


class NativeTokenPPOTrainingSession:
    """Compose model-owned actor-critic adapters with the native PPO engine."""

    def __init__(
        self,
        *,
        rollout_adapter: TokenPPORolloutAdapter,
        reward_adapter: TokenPPOTerminalRewardAdapter,
        scalarizer: WeightedRewardScalarizer,
        engine: NativeTokenPPOEngine,
        progress: TrainingProgress,
        sampling_temperature: float = 1.0,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        step_sink: Callable[[TokenPPOStepResult], None] | None = None,
    ) -> None:
        if not isinstance(rollout_adapter, TokenPPORolloutAdapter):
            raise TypeError("rollout_adapter must implement TokenPPORolloutAdapter")
        if not isinstance(reward_adapter, TokenPPOTerminalRewardAdapter):
            raise TypeError("reward_adapter must implement TokenPPOTerminalRewardAdapter")
        if not isinstance(scalarizer, WeightedRewardScalarizer):
            raise TypeError("scalarizer must be WeightedRewardScalarizer")
        if not isinstance(engine, NativeTokenPPOEngine):
            raise TypeError("engine must be NativeTokenPPOEngine")
        if not isinstance(progress, TrainingProgress) or progress.optimizer_steps != engine.global_step:
            raise ValueError("PPO progress must match engine global_step")
        temperature = float(sampling_temperature)
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")
        if isinstance(save_every_steps, bool) or int(save_every_steps) < 0:
            raise ValueError("save_every_steps must be non-negative")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError("checkpoint cadence requires checkpoint state and checkpointer")
        self.rollout_adapter = rollout_adapter
        self.reward_adapter = reward_adapter
        self.scalarizer = scalarizer
        self.engine = engine
        self.progress = progress
        self.sampling_temperature = temperature
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.save_every_steps = int(save_every_steps)
        self.asynchronous_checkpoints = bool(asynchronous_checkpoints)
        self.step_sink = step_sink
        self._pending: list[PendingTrainingCheckpoint] = []

    def wait_for_checkpoints(self) -> None:
        for pending in self._pending:
            pending.wait()
        self._pending.clear()

    @staticmethod
    def _validate_rollout(
        request: TokenPPORolloutRequest,
        trajectory: PackedTokenPPOTrajectory,
    ) -> None:
        if not isinstance(trajectory, PackedTokenPPOTrajectory):
            raise TypeError("PPO rollout must return PackedTokenPPOTrajectory")
        if trajectory.sample_ids != request.sample_ids:
            raise ValueError("PPO rollout changed sample order")
        if trajectory.policy_revision != request.policy_revision:
            raise ValueError("PPO rollout changed the policy revision")
        if trajectory.sampling_temperature != request.sampling_temperature:
            raise ValueError("PPO rollout changed the sampling temperature")

    def train_iteration(
        self,
        request: TokenPPORolloutRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> TokenPPOIterationResult:
        if not isinstance(request, TokenPPORolloutRequest):
            raise TypeError("request must be TokenPPORolloutRequest")
        if request.policy_revision != self.engine.current_policy_revision:
            raise ValueError("rollout request revision differs from the active PPO policy")
        if request.sampling_temperature != self.sampling_temperature:
            raise ValueError("rollout request sampling temperature differs from the recipe")
        if self.engine.global_step % self.engine.updates_per_trajectory:
            raise RuntimeError("PPO iteration started outside a completed trajectory boundary")

        initial_step = self.engine.global_step
        trajectory = self.rollout_adapter.rollout(request, generator=generator)
        self._validate_rollout(request, trajectory)
        reward_values = self.reward_adapter.score(trajectory)
        if set(reward_values) != set(self.reward_adapter.reward_ids):
            raise ValueError("PPO terminal reward ids differ from the adapter declaration")
        for reward_id, values in reward_values.items():
            if (
                not isinstance(values, torch.Tensor)
                or tuple(values.shape) != (trajectory.batch_size,)
                or not values.is_floating_point()
                or not bool(torch.isfinite(values).all())
            ):
                raise ValueError(f"PPO terminal reward {reward_id!r} must be finite with shape [B]")
        rewards = self.scalarizer.scalarize(reward_values)
        anchor = self.engine.prepare_trajectory(trajectory, rewards.scalar_rewards)
        updates: list[TokenPPOStepResult] = []
        while self.engine.has_active_trajectory:
            update = self.engine.train_step()
            updates.append(update)
            self.progress.record_step(
                microbatches=update.replay_microbatches,
                samples=update.sample_count,
                latent_tokens=update.token_count,
            )
            if self.step_sink is not None:
                self.step_sink(update)
        if len(updates) != self.engine.updates_per_trajectory:
            raise RuntimeError("PPO did not execute every configured update partition and epoch")
        if self.progress.optimizer_steps != self.engine.global_step:
            raise RuntimeError("PPO progress failed to commit with the engine")

        crossed_checkpoint_boundary = (
            self.save_every_steps > 0
            and self.progress.optimizer_steps // self.save_every_steps > initial_step // self.save_every_steps
        )
        if crossed_checkpoint_boundary:
            assert self.checkpointer is not None and self.checkpoint_state is not None
            artifact = self.checkpointer.save(
                self.checkpoint_state,
                asynchronous=self.asynchronous_checkpoints,
            )
            if isinstance(artifact, PendingTrainingCheckpoint):
                self._pending.append(artifact)
        return TokenPPOIterationResult(
            trajectory=trajectory,
            rewards=rewards,
            anchor=anchor,
            updates=tuple(updates),
        )


__all__ = ["NativeTokenPPOTrainingSession", "TokenPPOIterationResult"]
