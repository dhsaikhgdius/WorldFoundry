"""Rollout, reward, replay, and checkpoint lifecycle for token policies."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
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
    PackedTokenTrajectory,
    TokenPolicyRolloutAdapter,
    TokenRolloutRequest,
    TokenTrajectoryRewardAdapter,
    TokenTrajectoryRewards,
)
from .engine import NativeTokenPolicyEngine, TokenPolicyStepResult
from .packing import select_packed_token_trajectory


@dataclass(frozen=True, slots=True)
class TokenPolicyIterationResult:
    """One rollout and every optimizer update anchored to it."""

    trajectory: PackedTokenTrajectory
    rewards: RewardScalarizationResult
    updates: tuple[TokenPolicyStepResult, ...]


class NativeTokenPolicyTrainingSession:
    """Compose model-specific rollout with the shared token-policy learner."""

    def __init__(
        self,
        *,
        rollout_adapter: TokenPolicyRolloutAdapter,
        reward_adapter: TokenTrajectoryRewardAdapter,
        scalarizer: WeightedRewardScalarizer,
        engine: NativeTokenPolicyEngine,
        progress: TrainingProgress,
        group_size: int = 2,
        advantage_epsilon: float = 1.0e-8,
        advantage_normalization: str = "group-population-variance",
        advantage_clip_max: float | None = None,
        sampling_temperature: float = 1.0,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        step_sink: Callable[[TokenPolicyStepResult], None] | None = None,
    ) -> None:
        if not isinstance(rollout_adapter, TokenPolicyRolloutAdapter):
            raise TypeError("rollout_adapter must implement TokenPolicyRolloutAdapter")
        if not isinstance(reward_adapter, TokenTrajectoryRewardAdapter):
            raise TypeError("reward_adapter must implement TokenTrajectoryRewardAdapter")
        if not isinstance(scalarizer, WeightedRewardScalarizer):
            raise TypeError("scalarizer must be WeightedRewardScalarizer")
        if not isinstance(engine, NativeTokenPolicyEngine):
            raise TypeError("engine must be NativeTokenPolicyEngine")
        if not isinstance(progress, TrainingProgress) or progress.optimizer_steps != engine.global_step:
            raise ValueError("token-policy progress must match engine global step")
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
            raise ValueError("group_size must be an integer of at least two")
        if isinstance(save_every_steps, bool) or int(save_every_steps) < 0:
            raise ValueError("save_every_steps must be non-negative")
        temperature = float(sampling_temperature)
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError("checkpoint cadence requires checkpoint_state and checkpointer")
        self.rollout_adapter = rollout_adapter
        self.reward_adapter = reward_adapter
        self.scalarizer = scalarizer
        self.engine = engine
        self.progress = progress
        self.group_size = group_size
        self.advantage_epsilon = float(advantage_epsilon)
        self.advantage_normalization = advantage_normalization
        self.advantage_clip_max = advantage_clip_max
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
        request: TokenRolloutRequest,
        trajectory: PackedTokenTrajectory,
    ) -> None:
        if not isinstance(trajectory, PackedTokenTrajectory):
            raise TypeError("token rollout must return PackedTokenTrajectory")
        excluded = set(trajectory.excluded_sample_ids)
        expected_samples = tuple(
            sample_id for sample_id in request.sample_ids if sample_id not in excluded
        )
        expected_groups = tuple(
            group_id
            for sample_id, group_id in zip(
                request.sample_ids,
                request.group_ids,
                strict=True,
            )
            if sample_id not in excluded
        )
        if set(request.sample_ids) != set(expected_samples) | excluded:
            raise ValueError("token rollout reported unknown excluded samples")
        if trajectory.sample_ids != expected_samples or trajectory.group_ids != expected_groups:
            raise ValueError("token rollout changed the order of trainable samples")
        if trajectory.policy_revision != request.policy_revision:
            raise ValueError("token rollout changed the requested policy revision")
        if trajectory.sampling_temperature != request.sampling_temperature:
            raise ValueError("token rollout changed the requested sampling temperature")

    @staticmethod
    def _select_valid_rewards(
        trajectory: PackedTokenTrajectory,
        scored: TokenTrajectoryRewards,
    ) -> tuple[PackedTokenTrajectory, Mapping[str, torch.Tensor]]:
        values = dict(scored.values)
        valid = dict(scored.valid)
        reference = next(iter(values.values()))
        if any(tuple(value.shape) != (trajectory.batch_size,) for value in values.values()):
            raise ValueError("token reward values must match trajectory batch size")
        joint = torch.ones(
            trajectory.batch_size,
            device=reference.device,
            dtype=torch.bool,
        )
        for reward_id, value in values.items():
            joint &= valid[reward_id].to(device=joint.device)
            joint &= torch.isfinite(value.to(device=joint.device))
        keep_rows = joint.tolist()
        group_counts = Counter(
            group_id
            for group_id, keep in zip(trajectory.group_ids, keep_rows, strict=True)
            if keep
        )
        selected = tuple(
            index
            for index, (group_id, keep) in enumerate(
                zip(trajectory.group_ids, keep_rows, strict=True)
            )
            if keep and group_counts[group_id] >= 2
        )
        if not selected:
            raise ValueError("token rewards left no group with at least two valid trajectories")
        if len(selected) == trajectory.batch_size:
            return trajectory, values
        filtered = select_packed_token_trajectory(trajectory, selected)
        return filtered, {
            reward_id: value.index_select(
                0,
                torch.tensor(selected, device=value.device, dtype=torch.long),
            )
            for reward_id, value in values.items()
        }

    def train_iteration(
        self,
        request: TokenRolloutRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> TokenPolicyIterationResult:
        if not isinstance(request, TokenRolloutRequest):
            raise TypeError("request must be TokenRolloutRequest")
        if self.engine.is_poisoned:
            raise RuntimeError("token-policy engine is poisoned; restore complete training state into a fresh session")
        group_counts = Counter(request.group_ids)
        invalid_groups = sorted(group_id for group_id, count in group_counts.items() if count != self.group_size)
        if invalid_groups:
            raise ValueError(f"token-policy groups must each contain {self.group_size} samples: {invalid_groups}")
        if request.policy_revision != self.engine.current_policy_revision:
            raise ValueError("rollout request policy revision differs from the active engine")
        if request.sampling_temperature != self.sampling_temperature:
            raise ValueError("rollout request sampling temperature differs from the recipe")
        if self.engine.global_step % self.engine.updates_per_trajectory:
            raise RuntimeError("token-policy iteration started outside a trajectory boundary")
        initial_optimizer_step = self.engine.global_step
        trajectory = self.rollout_adapter.rollout(request, generator=generator)
        self._validate_rollout(request, trajectory)
        scored = self.reward_adapter.score(trajectory)
        if isinstance(scored, TokenTrajectoryRewards):
            trajectory, reward_values = self._select_valid_rewards(trajectory, scored)
        else:
            reward_values = scored
        rewards = self.scalarizer.scalarize(reward_values)
        anchor_id = self.engine.prepare_trajectory(
            trajectory,
            rewards.scalar_rewards,
            advantage_epsilon=self.advantage_epsilon,
            advantage_normalization=self.advantage_normalization,
            advantage_clip_max=self.advantage_clip_max,
        )
        updates: list[TokenPolicyStepResult] = []
        while self.engine.has_active_trajectory:
            result = self.engine.train_step(anchor_id=anchor_id)
            updates.append(result)
            if result.optimizer_committed:
                self.progress.record_step(
                    microbatches=result.replay_microbatches,
                    samples=result.sample_count,
                    latent_tokens=result.token_count,
                )
            if self.step_sink is not None:
                self.step_sink(result)
        if sum(result.sample_count for result in updates) != trajectory.batch_size:
            raise RuntimeError("token-policy optimizer partitions did not consume every sample exactly once")
        if sum(result.token_count for result in updates) != trajectory.token_count:
            raise RuntimeError("token-policy optimizer partitions did not consume every response token exactly once")
        if self.progress.optimizer_steps != self.engine.global_step:
            raise RuntimeError("token-policy progress failed to commit with the engine")
        crossed_checkpoint_boundary = (
            self.save_every_steps > 0
            and self.progress.optimizer_steps // self.save_every_steps > initial_optimizer_step // self.save_every_steps
        )
        if crossed_checkpoint_boundary:
            assert self.checkpointer is not None and self.checkpoint_state is not None
            artifact = self.checkpointer.save(
                self.checkpoint_state,
                asynchronous=self.asynchronous_checkpoints,
            )
            if isinstance(artifact, PendingTrainingCheckpoint):
                self._pending.append(artifact)
        return TokenPolicyIterationResult(
            trajectory=trajectory,
            rewards=rewards,
            updates=tuple(updates),
        )


__all__ = ["NativeTokenPolicyTrainingSession", "TokenPolicyIterationResult"]
