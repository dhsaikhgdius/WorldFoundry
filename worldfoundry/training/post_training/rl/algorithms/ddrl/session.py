"""Rollout, reward, update, and checkpoint session for DDRL."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress

from ....rewards.scalarization import (
    RewardScalarizationResult,
    WeightedRewardScalarizer,
)
from .contracts import (
    DDRLRewardAdapter,
    DDRLRolloutAdapter,
    DDRLRolloutBatch,
    DDRLTrajectory,
)
from .engine import DDRLStepResult, NativeDDRLEngine


@dataclass(frozen=True, slots=True)
class DDRLIterationResult:
    trajectory: DDRLTrajectory
    rewards: RewardScalarizationResult
    update: DDRLStepResult


class NativeDDRLTrainingSession:
    """Collect and optimize one complete selected-step trajectory at a time."""

    def __init__(
        self,
        *,
        rollout_adapter: DDRLRolloutAdapter,
        reward_adapter: DDRLRewardAdapter,
        scalarizer: WeightedRewardScalarizer,
        engine: NativeDDRLEngine,
        progress: TrainingProgress,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
        expected_train_on: tuple[int, ...] | None = None,
    ) -> None:
        if not isinstance(rollout_adapter, DDRLRolloutAdapter):
            raise TypeError("rollout_adapter must implement DDRLRolloutAdapter")
        if not isinstance(reward_adapter, DDRLRewardAdapter):
            raise TypeError("reward_adapter must implement DDRLRewardAdapter")
        if not isinstance(scalarizer, WeightedRewardScalarizer):
            raise TypeError("scalarizer must be WeightedRewardScalarizer")
        if tuple(reward_adapter.reward_ids) != tuple(scalarizer.weights):
            raise ValueError("DDRL reward adapter ids must match scalarizer weights")
        if not isinstance(engine, NativeDDRLEngine):
            raise TypeError("engine must be NativeDDRLEngine")
        if not isinstance(progress, TrainingProgress) or progress.optimizer_steps != engine.global_step:
            raise ValueError("DDRL progress must match engine global step")
        if isinstance(save_every_steps, bool) or int(save_every_steps) < 0:
            raise ValueError("save_every_steps must be non-negative")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError("checkpoint cadence requires checkpoint_state and checkpointer")
        if expected_train_on is not None:
            if any(isinstance(step, bool) or not isinstance(step, int) for step in expected_train_on):
                raise TypeError("expected_train_on must contain integer indices")
            expected_train_on = tuple(expected_train_on)
            if (
                not expected_train_on
                or expected_train_on[0] < 0
                or expected_train_on != tuple(sorted(set(expected_train_on)))
            ):
                raise ValueError("expected_train_on must be non-empty, non-negative, strictly increasing, and unique")
        self.rollout_adapter = rollout_adapter
        self.reward_adapter = reward_adapter
        self.scalarizer = scalarizer
        self.engine = engine
        self.progress = progress
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.save_every_steps = int(save_every_steps)
        self.asynchronous_checkpoints = bool(asynchronous_checkpoints)
        self.event_sink = event_sink
        self.expected_train_on = expected_train_on

    def _checkpoint_if_due(self) -> None:
        if not self.save_every_steps or self.progress.optimizer_steps % self.save_every_steps:
            return
        assert self.checkpointer is not None and self.checkpoint_state is not None
        artifact = self.checkpointer.save(
            self.checkpoint_state,
            asynchronous=self.asynchronous_checkpoints,
        )
        if isinstance(artifact, PendingTrainingCheckpoint):
            artifact.wait()

    def train_iteration(
        self,
        batch: DDRLRolloutBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> DDRLIterationResult:
        if not isinstance(batch, DDRLRolloutBatch):
            raise TypeError("batch must be DDRLRolloutBatch")
        trajectory = self.rollout_adapter.collect(batch, generator=generator)
        if not isinstance(trajectory, DDRLTrajectory):
            raise TypeError("DDRL rollout adapter must return DDRLTrajectory")
        if trajectory.sample_ids != batch.sample_ids or trajectory.group_ids != batch.group_ids:
            raise ValueError("DDRL trajectory sample/group ordering differs from its rollout batch")
        if self.expected_train_on is not None and trajectory.train_on != self.expected_train_on:
            raise ValueError("DDRL trajectory train_on differs from the configured recipe indices")
        components = self.reward_adapter.score(trajectory)
        rewards = self.scalarizer.scalarize(components)
        if not isinstance(rewards.scalar_rewards, torch.Tensor):
            raise TypeError("DDRL scalar rewards must be a torch.Tensor")
        update = self.engine.train_trajectory(
            trajectory,
            rewards.scalar_rewards,
            generator=generator,
        )
        latent_tokens = (
            trajectory.batch_size
            * trajectory.step_count
            * prod(int(size) for size in trajectory.next_latents.shape[3:])
        )
        self.progress.record_step(
            microbatches=trajectory.step_count,
            samples=trajectory.batch_size,
            latent_tokens=latent_tokens,
        )
        if self.progress.optimizer_steps != self.engine.global_step:
            raise RuntimeError("DDRL progress failed to commit with the engine")
        if self.event_sink is not None:
            self.event_sink(
                {
                    "schema": "worldfoundry-ddrl-step-event",
                    "global_step": self.engine.global_step,
                    "trajectory_id": trajectory.trajectory_id,
                    "train_on": list(trajectory.train_on),
                    "loss": float(update.loss.item()),
                    "policy_loss": float(update.policy_loss.item()),
                    "ratio_mean": float(update.ratios.mean().item()),
                }
            )
        self._checkpoint_if_due()
        return DDRLIterationResult(
            trajectory=trajectory,
            rewards=rewards,
            update=update,
        )


__all__ = ["DDRLIterationResult", "NativeDDRLTrainingSession"]
