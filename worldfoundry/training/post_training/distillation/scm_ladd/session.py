"""Synchronous run session for native sCM-LADD distillation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import prod

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress

from .contracts import SCMLADDTrainingBatch
from .engine import NativeSCMLADDTrainEngine, SCMLADDTrainResult


def _latent_tokens(tensor: torch.Tensor) -> int:
    if tensor.ndim < 2:
        raise ValueError("latent tensor must include batch and channel/feature dimensions")
    return int(tensor.shape[0]) * prod(int(size) for size in tensor.shape[2:])


@dataclass(frozen=True, slots=True)
class SCMLADDRunSummary:
    initial_step: int
    final_step: int
    updates: int
    student_optimizer_steps: int
    discriminator_optimizer_steps: int
    final_generator_loss: float | None
    final_discriminator_loss: float | None


class NativeSCMLADDTrainingSession:
    """Drive alternating generator/discriminator updates from a latent loader."""

    def __init__(
        self,
        engine: NativeSCMLADDTrainEngine,
        dataloader: Iterable[SCMLADDTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(engine, NativeSCMLADDTrainEngine):
            raise TypeError("engine must be NativeSCMLADDTrainEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("SCM-LADD progress and engine global step differ")
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
        boundary_every_steps: int = 0,
        boundary_sink: Callable[[int, int], None] | None = None,
    ) -> SCMLADDRunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if isinstance(boundary_every_steps, bool) or int(boundary_every_steps) < 0:
            raise ValueError("boundary_every_steps must be non-negative")
        if boundary_every_steps and boundary_sink is None:
            raise ValueError("boundary cadence requires boundary_sink")
        initial_step = self.engine.global_step
        iterator = iter(self.dataloader)
        final_generator: SCMLADDTrainResult | None = None
        final_discriminator: SCMLADDTrainResult | None = None
        try:
            for _ in range(int(max_steps)):
                batches: list[SCMLADDTrainingBatch] = []
                for _ in range(self.engine.gradient_accumulation_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(self.dataloader)
                        try:
                            batch = next(iterator)
                        except StopIteration as error:
                            raise RuntimeError("SCM-LADD dataloader is empty") from error
                    if not isinstance(batch, SCMLADDTrainingBatch):
                        raise TypeError("SCM-LADD dataloader must emit SCMLADDTrainingBatch values")
                    batches.append(batch)
                previous_step = self.engine.global_step
                result = self.engine.train_step(tuple(batches), generator=generator)
                self.progress.record_step(
                    microbatches=len(batches),
                    samples=sum(batch.batch_size for batch in batches),
                    latent_tokens=sum(_latent_tokens(batch.clean_latents) for batch in batches),
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("SCM-LADD progress failed to commit with the engine")
                if result.phase == "generator":
                    final_generator = result
                else:
                    final_discriminator = result
                if self.event_sink is not None:
                    self.event_sink(
                        {
                            "schema": "worldfoundry-scm-ladd-step-event",
                            "global_step": self.engine.global_step,
                            "phase": result.phase,
                            "microbatches": len(batches),
                            "samples": sum(batch.batch_size for batch in batches),
                            "loss": float(result.loss.item()),
                        }
                    )
                self._checkpoint_if_due()
                if boundary_every_steps and self.engine.global_step // int(boundary_every_steps) > (
                    previous_step // int(boundary_every_steps)
                ):
                    assert boundary_sink is not None
                    boundary_sink(previous_step, self.engine.global_step)
        finally:
            self.wait_for_checkpoints()
        return SCMLADDRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            updates=int(max_steps),
            student_optimizer_steps=self.engine.student_optimizer_steps,
            discriminator_optimizer_steps=self.engine.discriminator_optimizer_steps,
            final_generator_loss=(None if final_generator is None else float(final_generator.loss.item())),
            final_discriminator_loss=(
                None if final_discriminator is None else float(final_discriminator.loss.item())
            ),
        )


__all__ = ["NativeSCMLADDTrainingSession", "SCMLADDRunSummary"]
