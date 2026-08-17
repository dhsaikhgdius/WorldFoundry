"""Exact-boundary execution session for native Data-Forcing Distillation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.post_training.shared.batching import latent_token_count as _latent_tokens

from .contracts import DFDTrainingBatch
from .engine import DFDTrainResult, NativeDFDTrainEngine


@dataclass(frozen=True, slots=True)
class DFDRunSummary:
    initial_step: int
    final_step: int
    iterations: int
    student_optimizer_steps: int
    fake_score_optimizer_steps: int
    discriminator_optimizer_steps: int
    final_phase: str
    final_loss: float


class NativeDFDTrainingSession:
    def __init__(
        self,
        engine: NativeDFDTrainEngine,
        dataloader: Iterable[DFDTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
    ) -> None:
        if not isinstance(engine, NativeDFDTrainEngine):
            raise TypeError("engine must be NativeDFDTrainEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("DFD progress and engine global step differ")
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
        self._pending: list[PendingTrainingCheckpoint] = []

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
    ) -> DFDRunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        initial_step = self.engine.global_step
        iterator = iter(self.dataloader)
        final_result: DFDTrainResult | None = None
        try:
            for _ in range(int(max_steps)):
                batches: list[DFDTrainingBatch] = []
                for _ in range(self.engine.gradient_accumulation_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(self.dataloader)
                        try:
                            batch = next(iterator)
                        except StopIteration as error:
                            raise RuntimeError("DFD dataloader is empty") from error
                    if not isinstance(batch, DFDTrainingBatch):
                        raise TypeError("DFD dataloader value must be DFDTrainingBatch")
                    batches.append(batch)
                final_result = self.engine.train_step(tuple(batches), generator=generator)
                if not all(isinstance(batch.real_latents, torch.Tensor) for batch in batches):
                    raise TypeError("DFD real_latents must be torch tensors")
                self.progress.record_step(
                    microbatches=len(batches),
                    samples=sum(batch.batch_size for batch in batches),
                    latent_tokens=sum(
                        _latent_tokens(batch.real_latents)
                        for batch in batches
                    ),
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("DFD progress failed to commit with the engine")
                self._checkpoint_if_due()
        finally:
            self.wait_for_checkpoints()
        assert final_result is not None
        return DFDRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=int(max_steps),
            student_optimizer_steps=self.engine.student_optimizer_steps,
            fake_score_optimizer_steps=self.engine.fake_score_optimizer_steps,
            discriminator_optimizer_steps=self.engine.discriminator_optimizer_steps,
            final_phase=final_result.phase,
            final_loss=float(final_result.loss.item()),
        )


__all__ = ["DFDRunSummary", "NativeDFDTrainingSession"]
