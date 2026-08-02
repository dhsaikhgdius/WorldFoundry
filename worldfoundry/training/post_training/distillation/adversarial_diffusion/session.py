"""Exact-boundary execution session for native ADD training."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress

from .contracts import ADDTrainingBatch
from .engine import ADDTrainResult, NativeADDTrainEngine


def _latent_tokens(batch: ADDTrainingBatch) -> int:
    latents = batch.clean_latents
    if not isinstance(latents, torch.Tensor) or latents.ndim < 2:
        raise TypeError("ADD clean_latents must be a tensor with batch and feature dimensions")
    return int(latents.shape[0]) * prod(int(size) for size in latents.shape[2:])


@dataclass(frozen=True, slots=True)
class ADDRunSummary:
    initial_step: int
    final_step: int
    iterations: int
    student_optimizer_steps: int
    discriminator_optimizer_steps: int
    final_generator_loss: float
    final_discriminator_loss: float


class NativeADDTrainingSession:
    """Drive complete discriminator-to-student iterations and safe checkpoints."""

    def __init__(
        self,
        engine: NativeADDTrainEngine,
        dataloader: Iterable[ADDTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
    ) -> None:
        if not isinstance(engine, NativeADDTrainEngine):
            raise TypeError("engine must be NativeADDTrainEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("ADD progress and engine global step differ")
        if isinstance(save_every_steps, bool) or not isinstance(save_every_steps, int):
            raise TypeError("save_every_steps must be an integer")
        if save_every_steps < 0:
            raise ValueError("save_every_steps must be non-negative")
        if not isinstance(asynchronous_checkpoints, bool):
            raise TypeError("asynchronous_checkpoints must be a bool")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError("checkpoint cadence requires checkpoint_state and checkpointer")
        self.engine = engine
        self.dataloader = dataloader
        self.progress = progress
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.save_every_steps = save_every_steps
        self.asynchronous_checkpoints = asynchronous_checkpoints
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
    ) -> ADDRunSummary:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("max_steps must be an integer")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None")
        initial_step = self.engine.global_step
        iterator = iter(self.dataloader)
        final_result: ADDTrainResult | None = None

        def next_batch() -> ADDTrainingBatch:
            nonlocal iterator
            try:
                value = next(iterator)
            except StopIteration:
                iterator = iter(self.dataloader)
                try:
                    value = next(iterator)
                except StopIteration as error:
                    raise RuntimeError("ADD dataloader is empty") from error
            if not isinstance(value, ADDTrainingBatch):
                raise TypeError("ADD dataloader must emit ADDTrainingBatch values")
            return value

        try:
            for _ in range(max_steps):
                batches = tuple(next_batch() for _ in range(self.engine.gradient_accumulation_steps))
                samples = sum(batch.batch_size for batch in batches)
                latent_tokens = sum(_latent_tokens(batch) for batch in batches)
                final_result = self.engine.train_step(batches, generator=generator)
                self.progress.record_step(
                    microbatches=len(batches),
                    samples=samples,
                    latent_tokens=latent_tokens,
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("ADD progress failed to commit with the engine")
                self._checkpoint_if_due()
        finally:
            self.wait_for_checkpoints()

        assert final_result is not None
        return ADDRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=max_steps,
            student_optimizer_steps=self.engine.student_optimizer_steps,
            discriminator_optimizer_steps=self.engine.discriminator_optimizer_steps,
            final_generator_loss=float(final_result.generator_loss.item()),
            final_discriminator_loss=float(final_result.discriminator_loss.item()),
        )


__all__ = ["ADDRunSummary", "NativeADDTrainingSession"]
