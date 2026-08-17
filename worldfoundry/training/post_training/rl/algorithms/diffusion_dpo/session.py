"""Synchronous paired-data training session for Diffusion-DPO."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress

from ....shared.batching import latent_token_count as _latent_tokens
from .contracts import DiffusionDPOBatch
from .engine import DiffusionDPOStepResult, NativeDiffusionDPOEngine


@dataclass(frozen=True, slots=True)
class DiffusionDPORunSummary:
    initial_step: int
    final_step: int
    iterations: int
    final_loss: float
    final_preference_accuracy: float


class NativeDiffusionDPOTrainingSession:
    """Consume paired clean-latent batches and commit exact step boundaries."""

    def __init__(
        self,
        engine: NativeDiffusionDPOEngine,
        dataloader: Iterable[DiffusionDPOBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(engine, NativeDiffusionDPOEngine):
            raise TypeError("engine must be NativeDiffusionDPOEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("Diffusion-DPO progress and engine global step differ")
        if isinstance(save_every_steps, bool) or int(save_every_steps) < 0:
            raise ValueError("save_every_steps must be non-negative")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError("checkpoint cadence requires checkpoint_state and checkpointer")
        self.engine = engine
        self.dataloader = dataloader
        self.progress = progress
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.save_every_steps = int(save_every_steps)
        self.asynchronous_checkpoints = bool(asynchronous_checkpoints)
        self.event_sink = event_sink
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

    def run(
        self,
        *,
        max_steps: int,
        generator: torch.Generator | None = None,
    ) -> DiffusionDPORunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        initial_step = self.engine.global_step
        iterator = iter(self.dataloader)
        final_result: DiffusionDPOStepResult | None = None
        try:
            for _ in range(int(max_steps)):
                try:
                    batch = next(iterator)
                except StopIteration as error:
                    raise RuntimeError("Diffusion-DPO dataloader exhausted before max_steps") from error
                if not isinstance(batch, DiffusionDPOBatch):
                    raise TypeError("Diffusion-DPO dataloader must emit DiffusionDPOBatch values")
                result = self.engine.train_step(batch, generator=generator)
                self.progress.record_step(
                    microbatches=1,
                    samples=batch.batch_size,
                    latent_tokens=_latent_tokens(batch.clean_latents),
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("Diffusion-DPO progress failed to commit with the engine")
                final_result = result
                self._emit(
                    {
                        "schema": "worldfoundry-diffusion-dpo-step-event",
                        "global_step": self.engine.global_step,
                        "batch_id": batch.batch_id,
                        "pair_count": batch.pair_count,
                        "loss": float(result.loss.item()),
                        "preference_accuracy": float(result.preference_accuracy.item()),
                    }
                )
                self._checkpoint_if_due()
        finally:
            self.wait_for_checkpoints()
        assert final_result is not None
        return DiffusionDPORunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=int(max_steps),
            final_loss=float(final_result.loss.item()),
            final_preference_accuracy=float(final_result.preference_accuracy.item()),
        )


__all__ = ["DiffusionDPORunSummary", "NativeDiffusionDPOTrainingSession"]
