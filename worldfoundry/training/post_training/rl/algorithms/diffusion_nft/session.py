"""Synchronous rollout-to-update session for native DiffusionNFT."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import prod

import torch

from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress

from ....rewards.scalarization import (
    RewardScalarizationResult,
    WeightedRewardScalarizer,
)
from ...contracts import FlowRolloutBatch
from .collection import NativeDiffusionNFTTerminalCollector
from .contracts import DiffusionNFTRewardAdapter, DiffusionNFTRollout
from .engine import DiffusionNFTStepResult, NativeDiffusionNFTEngine


def _latent_tokens(tensor: torch.Tensor) -> int:
    if tensor.ndim < 2:
        raise ValueError("clean latent tensor must include batch and channel/feature dimensions")
    return int(tensor.shape[0]) * prod(int(size) for size in tensor.shape[2:])


@dataclass(frozen=True, slots=True)
class DiffusionNFTRunSummary:
    initial_step: int
    final_step: int
    iterations: int
    old_policy_refreshes: int
    final_loss: float
    final_policy_loss: float


@dataclass(frozen=True, slots=True)
class DiffusionNFTIterationResult:
    rollout: DiffusionNFTRollout
    update: DiffusionNFTStepResult
    reward_components: Mapping[str, object] | None
    scalarization: RewardScalarizationResult | None


class NativeDiffusionNFTTrainingSession:
    """Consume each collected terminal-latent rollout exactly once."""

    def __init__(
        self,
        engine: NativeDiffusionNFTEngine,
        dataloader: Iterable[DiffusionNFTRollout | FlowRolloutBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
        collector: NativeDiffusionNFTTerminalCollector | None = None,
        reward_adapter: DiffusionNFTRewardAdapter | None = None,
        scalarizer: WeightedRewardScalarizer | None = None,
    ) -> None:
        if not isinstance(engine, NativeDiffusionNFTEngine):
            raise TypeError("engine must be NativeDiffusionNFTEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("DiffusionNFT progress and engine global step differ")
        if isinstance(save_every_steps, bool) or int(save_every_steps) < 0:
            raise ValueError("save_every_steps must be non-negative")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError("checkpoint cadence requires checkpoint_state and checkpointer")
        pipeline_components = (collector, reward_adapter, scalarizer)
        if any(value is not None for value in pipeline_components) and not all(
            value is not None for value in pipeline_components
        ):
            raise ValueError("DiffusionNFT terminal collection requires collector, reward_adapter, and scalarizer")
        if collector is not None and not isinstance(
            collector,
            NativeDiffusionNFTTerminalCollector,
        ):
            raise TypeError("collector must be NativeDiffusionNFTTerminalCollector")
        if reward_adapter is not None and not isinstance(
            reward_adapter,
            DiffusionNFTRewardAdapter,
        ):
            raise TypeError("reward_adapter must implement DiffusionNFTRewardAdapter")
        if scalarizer is not None and not isinstance(scalarizer, WeightedRewardScalarizer):
            raise TypeError("scalarizer must be WeightedRewardScalarizer")
        self.engine = engine
        self.dataloader = dataloader
        self.progress = progress
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.save_every_steps = int(save_every_steps)
        self.asynchronous_checkpoints = bool(asynchronous_checkpoints)
        self.event_sink = event_sink
        self.collector = collector
        self.reward_adapter = reward_adapter
        self.scalarizer = scalarizer
        self._pending: list[PendingTrainingCheckpoint] = []

    def _emit(self, payload: Mapping[str, object]) -> None:
        if self.event_sink is not None:
            self.event_sink(payload)

    def _checkpoint_if_due(self) -> None:
        if not self.save_every_steps or self.progress.optimizer_steps % self.save_every_steps:
            return
        assert self.checkpointer is not None and self.checkpoint_state is not None
        artifact = self.checkpointer.save(
            self.checkpoint_state,
            asynchronous=self.asynchronous_checkpoints,
        )
        if isinstance(artifact, PendingTrainingCheckpoint):
            self._pending.append(artifact)

    def wait_for_checkpoints(self) -> None:
        for pending in self._pending:
            pending.wait()
        self._pending.clear()

    def _prepare_rollout(
        self,
        batch: DiffusionNFTRollout | FlowRolloutBatch,
    ) -> tuple[
        DiffusionNFTRollout,
        Mapping[str, object] | None,
        RewardScalarizationResult | None,
    ]:
        if isinstance(batch, DiffusionNFTRollout):
            if self.collector is not None:
                raise TypeError("a collection-enabled DiffusionNFT session requires FlowRolloutBatch values")
            return batch, None, None
        if not isinstance(batch, FlowRolloutBatch):
            raise TypeError("DiffusionNFT dataloader must emit DiffusionNFTRollout or FlowRolloutBatch values")
        if batch.policy_revision != self.engine.current_collection_policy_revision:
            raise ValueError("DiffusionNFT collection batch was created by a stale behavior policy")
        if self.collector is None or self.reward_adapter is None or self.scalarizer is None:
            raise TypeError("FlowRolloutBatch requires the DiffusionNFT collection and reward pipeline")
        collection_id = canonical_sha256(
            {
                "schema": "worldfoundry-diffusion-nft-collection",
                "optimizer_step": self.engine.global_step,
                "policy_revision": batch.policy_revision,
                "sample_ids": batch.sample_ids,
            }
        )
        terminal = self.collector.collect(batch, collection_id=collection_id)
        components = self.reward_adapter.score(terminal)
        scalarized = self.scalarizer.scalarize(components)
        if not isinstance(scalarized.scalar_rewards, torch.Tensor):
            raise TypeError("DiffusionNFT scalar rewards must be a torch.Tensor")
        return terminal.with_rewards(scalarized.scalar_rewards), components, scalarized

    def train_iteration(
        self,
        batch: DiffusionNFTRollout | FlowRolloutBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> DiffusionNFTIterationResult:
        """Collect, score, scalarize, and commit one non-replayable update."""

        rollout, components, scalarization = self._prepare_rollout(batch)
        result = self.engine.train_step(rollout, generator=generator)
        self.progress.record_step(
            microbatches=1,
            samples=rollout.batch_size,
            latent_tokens=_latent_tokens(rollout.clean_latents),
        )
        if self.progress.optimizer_steps != self.engine.global_step:
            raise RuntimeError("DiffusionNFT progress failed to commit with the engine")
        self._emit(
            {
                "schema": "worldfoundry-diffusion-nft-step-event",
                "global_step": self.engine.global_step,
                "collection_id": rollout.collection_id,
                "policy_revision": rollout.policy_revision,
                "next_collection_policy_revision": (self.engine.current_collection_policy_revision),
                "loss": float(result.loss.item()),
                "policy_loss": float(result.policy_loss.item()),
                "old_policy_refreshed": result.old_policy_refreshed,
            }
        )
        self._checkpoint_if_due()
        return DiffusionNFTIterationResult(
            rollout=rollout,
            update=result,
            reward_components=components,
            scalarization=scalarization,
        )

    def run(
        self,
        *,
        max_steps: int,
        generator: torch.Generator | None = None,
    ) -> DiffusionNFTRunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        initial_step = self.engine.global_step
        iterator = iter(self.dataloader)
        final_result: DiffusionNFTStepResult | None = None
        try:
            for _ in range(int(max_steps)):
                try:
                    batch = next(iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        "DiffusionNFT dataloader exhausted before max_steps; collected rollouts cannot be replayed"
                    ) from error
                iteration = self.train_iteration(batch, generator=generator)
                final_result = iteration.update
        finally:
            self.wait_for_checkpoints()
        assert final_result is not None
        return DiffusionNFTRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=int(max_steps),
            old_policy_refreshes=self.engine.old_policy_refreshes,
            final_loss=float(final_result.loss.item()),
            final_policy_loss=float(final_result.policy_loss.item()),
        )


__all__ = [
    "DiffusionNFTIterationResult",
    "DiffusionNFTRunSummary",
    "NativeDiffusionNFTTrainingSession",
]
