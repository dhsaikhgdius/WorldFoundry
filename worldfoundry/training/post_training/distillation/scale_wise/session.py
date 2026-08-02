"""Exact-boundary execution session for native scale-wise distillation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress

from .contracts import ScaleWiseTrainingBatch
from .engine import NativeScaleWiseTrainEngine, ScaleWiseTrainResult


def _latent_tokens(batch: ScaleWiseTrainingBatch) -> int:
    current = batch.current_latents
    if not isinstance(current, torch.Tensor):
        raise TypeError("current_latents must be a torch.Tensor")
    return int(current.shape[0]) * prod(int(size) for size in current.shape[2:])


@dataclass(frozen=True, slots=True)
class ScaleWiseRunSummary:
    initial_step: int
    final_step: int
    iterations: int
    student_optimizer_steps: int
    fake_score_optimizer_steps: int
    final_student_loss: float
    final_fake_score_loss: float


class NativeScaleWiseTrainingSession:
    """Consume fresh fake batches before the student batch at each interval."""

    def __init__(
        self,
        engine: NativeScaleWiseTrainEngine,
        dataloader: Iterable[ScaleWiseTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
    ) -> None:
        if not isinstance(engine, NativeScaleWiseTrainEngine):
            raise TypeError("engine must be NativeScaleWiseTrainEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("scale-wise progress and engine global step differ")
        if isinstance(save_every_steps, bool) or int(save_every_steps) < 0:
            raise ValueError("save_every_steps must be non-negative")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError("checkpoint cadence requires state and checkpointer")
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
    ) -> ScaleWiseRunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        initial_step = self.engine.global_step
        iterator = iter(self.dataloader)
        final_result: ScaleWiseTrainResult | None = None

        def next_batch() -> ScaleWiseTrainingBatch:
            nonlocal iterator
            try:
                value = next(iterator)
            except StopIteration:
                iterator = iter(self.dataloader)
                try:
                    value = next(iterator)
                except StopIteration as error:
                    raise RuntimeError("scale-wise dataloader is empty") from error
            if not isinstance(value, ScaleWiseTrainingBatch):
                raise TypeError("scale-wise dataloader must emit typed batches")
            return value

        try:
            for _ in range(int(max_steps)):
                fake_count = (
                    self.engine.fake_updates_per_iteration
                    * self.engine.gradient_accumulation_steps
                )
                fake_batches = tuple(next_batch() for _ in range(fake_count))
                student_batches = tuple(
                    next_batch()
                    for _ in range(self.engine.gradient_accumulation_steps)
                )
                consumed = (*fake_batches, *student_batches)
                final_result = self.engine.train_step(
                    student_batches,
                    fake_score_batches=fake_batches,
                    generator=generator,
                )
                self.progress.record_step(
                    microbatches=len(consumed),
                    samples=sum(batch.batch_size for batch in consumed),
                    latent_tokens=sum(_latent_tokens(batch) for batch in consumed),
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError(
                        "scale-wise progress failed to commit with engine"
                    )
                self._checkpoint_if_due()
        finally:
            self.wait_for_checkpoints()
        assert final_result is not None
        return ScaleWiseRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=int(max_steps),
            student_optimizer_steps=self.engine.student_optimizer_steps,
            fake_score_optimizer_steps=self.engine.fake_score_optimizer_steps,
            final_student_loss=float(final_result.student_loss.item()),
            final_fake_score_loss=float(final_result.fake_score_loss.item()),
        )


__all__ = ["NativeScaleWiseTrainingSession", "ScaleWiseRunSummary"]
