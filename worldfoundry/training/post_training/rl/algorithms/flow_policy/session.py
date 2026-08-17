"""Shared rollout, reward, replay, and update session for flow policies."""

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
from ...contracts import (
    FlowRolloutBatch,
    FlowTrajectory,
    FlowTrajectorySamplingAdapter,
    TrajectoryRewardAdapter,
)
from ...rollout_strategies.contracts import FlowSDEIndexResolver
from ...rollout_strategies.sparse_sde_steps import FlowSDEIndexSchedule
from .engine import FlowPolicyStepResult, NativeFlowPolicyEngine


@dataclass(frozen=True, slots=True)
class FlowPolicyIterationResult:
    """A completed rollout and every policy update anchored to it."""

    trajectory: FlowTrajectory
    rewards: RewardScalarizationResult
    updates: tuple[FlowPolicyStepResult, ...]


class NativeFlowPolicyTrainingSession:
    """Compose the algorithm-neutral flow-policy training lifecycle."""

    engine_type: type[NativeFlowPolicyEngine] = NativeFlowPolicyEngine
    iteration_result_type: type[FlowPolicyIterationResult] = FlowPolicyIterationResult
    event_schema = "worldfoundry-flow-policy-step-event"
    event_metric_names: tuple[str, ...] = ("approx_kl",)

    def __init__(
        self,
        *,
        sampler: FlowTrajectorySamplingAdapter,
        reward_adapter: TrajectoryRewardAdapter,
        scalarizer: WeightedRewardScalarizer,
        engine: NativeFlowPolicyEngine,
        progress: TrainingProgress,
        sde_step_indices: tuple[int, ...] | None = None,
        sde_index_schedule: FlowSDEIndexResolver | None = None,
        old_log_prob_source: str = "rollout",
        advantage_epsilon: float = 1.0e-8,
        advantage_normalization: str = "group-population-variance",
        advantage_clip_max: float | None = None,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(sampler, FlowTrajectorySamplingAdapter):
            raise TypeError("sampler must implement FlowTrajectorySamplingAdapter")
        if not isinstance(reward_adapter, TrajectoryRewardAdapter):
            raise TypeError("reward_adapter must implement TrajectoryRewardAdapter")
        if not isinstance(scalarizer, WeightedRewardScalarizer):
            raise TypeError("scalarizer must be WeightedRewardScalarizer")
        if not isinstance(engine, self.engine_type):
            raise TypeError(f"engine must be {self.engine_type.__name__}")
        if not isinstance(progress, TrainingProgress) or progress.optimizer_steps != engine.global_step:
            raise ValueError(f"{engine.display_name} progress must match engine global step")
        if sde_index_schedule is not None and sde_step_indices is not None:
            raise ValueError("provide sde_index_schedule or static sde_step_indices, not both")
        if sde_index_schedule is None:
            if sde_step_indices is None:
                raise ValueError(f"{engine.display_name} requires an SDE index schedule")
            sde_index_schedule = FlowSDEIndexSchedule(
                transition_count=max(int(index) for index in sde_step_indices) + 1,
                static_indices=sde_step_indices,
            )
        if not isinstance(sde_index_schedule, FlowSDEIndexResolver):
            raise TypeError("sde_index_schedule must implement FlowSDEIndexResolver")
        if old_log_prob_source not in {"rollout", "replay"}:
            raise ValueError("old_log_prob_source must be 'rollout' or 'replay'")
        if isinstance(save_every_steps, bool) or int(save_every_steps) < 0:
            raise ValueError("save_every_steps must be non-negative")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError("checkpoint cadence requires checkpoint_state and checkpointer")
        self.sampler = sampler
        self.reward_adapter = reward_adapter
        self.scalarizer = scalarizer
        self.engine = engine
        self.progress = progress
        self.sde_index_schedule = sde_index_schedule
        self.old_log_prob_source = old_log_prob_source
        self.advantage_epsilon = float(advantage_epsilon)
        self.advantage_normalization = advantage_normalization
        self.advantage_clip_max = advantage_clip_max
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.save_every_steps = int(save_every_steps)
        self.asynchronous_checkpoints = bool(asynchronous_checkpoints)
        self.event_sink = event_sink
        self._pending: list[PendingTrainingCheckpoint] = []

    def _prepare_engine_trajectory(
        self,
        trajectory: FlowTrajectory,
        components: Mapping[str, object],
        rewards: RewardScalarizationResult,
        *,
        generator: torch.Generator | None,
    ) -> tuple[FlowTrajectory, str]:
        del components, generator
        anchor = self.engine.prepare_trajectory(
            trajectory,
            rewards.scalar_rewards,
            old_log_prob_source=self.old_log_prob_source,
            advantage_epsilon=self.advantage_epsilon,
            advantage_normalization=self.advantage_normalization,
            advantage_clip_max=self.advantage_clip_max,
        )
        return trajectory, anchor

    def wait_for_checkpoints(self) -> None:
        for pending in self._pending:
            pending.wait()
        self._pending.clear()

    def train_iteration(
        self,
        batch: FlowRolloutBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> FlowPolicyIterationResult:
        if not isinstance(batch, FlowRolloutBatch):
            raise TypeError("batch must be FlowRolloutBatch")
        if batch.policy_revision != self.engine.current_policy_revision:
            raise ValueError("rollout batch policy revision differs from the active engine")
        if self.engine.global_step % self.engine.updates_per_trajectory:
            raise RuntimeError(f"{self.engine.display_name} iteration started outside a trajectory boundary")
        rollout_id = self.engine.global_step // self.engine.updates_per_trajectory
        sde_step_indices = self.sde_index_schedule.resolve(rollout_id)
        initial_optimizer_step = self.engine.global_step
        trajectory = self.sampler.sample(
            batch.initial_latents,
            batch.sigmas,
            sample_ids=batch.sample_ids,
            group_ids=batch.group_ids,
            conditioning=batch.conditioning,
            policy_revision=batch.policy_revision,
            sde_step_indices=sde_step_indices,
            generator=generator,
            metadata=batch.metadata,
        )
        components = self.reward_adapter.score(trajectory)
        rewards = self.scalarizer.scalarize(components)
        trajectory, anchor = self._prepare_engine_trajectory(
            trajectory,
            components,
            rewards,
            generator=generator,
        )
        updates: list[FlowPolicyStepResult] = []
        while self.engine.has_active_trajectory:
            result = self.engine.train_step(anchor_id=anchor)
            self.progress.record_step(
                microbatches=result.replay_microbatches,
                samples=result.sample_count,
                latent_tokens=result.token_count,
            )
            updates.append(result)
            if self.event_sink is not None:
                event: dict[str, object] = {
                    "schema": self.event_schema,
                    "global_step": self.engine.global_step,
                    "policy_revision": self.engine.current_policy_revision,
                    "rollout_id": rollout_id,
                    "sde_step_indices": list(sde_step_indices),
                    "policy_loss": float(result.policy_loss.item()),
                }
                for name in self.event_metric_names:
                    value = result.metrics.get(name)
                    if value is not None:
                        event[name] = float(value.item())
                self.event_sink(event)
        selected_transitions = (
            trajectory.batch_size * len(trajectory.step_indices)
            if trajectory.update_step_mask is None
            else int(trajectory.update_step_mask.sum().item())
        )
        expected_token_count = selected_transitions * prod(
            int(size) for size in trajectory.latents.shape[3:]
        )
        if sum(result.sample_count for result in updates) != trajectory.batch_size:
            raise RuntimeError("flow-policy optimizer partitions did not consume every sample exactly once")
        if sum(result.token_count for result in updates) != expected_token_count:
            raise RuntimeError("flow-policy optimizer partitions did not consume every latent token exactly once")
        if self.progress.optimizer_steps != self.engine.global_step:
            raise RuntimeError(f"{self.engine.display_name} progress failed to commit with the engine")
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
        return self.iteration_result_type(
            trajectory=trajectory,
            rewards=rewards,
            updates=tuple(updates),
        )


__all__ = ["FlowPolicyIterationResult", "NativeFlowPolicyTrainingSession"]
