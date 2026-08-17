"""Synchronous execution session for native DMD training."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.post_training.shared.batching import latent_token_count as _latent_tokens

from .contracts import DMDTrainingBatch
from .engine import DMDTrainResult


@dataclass(frozen=True, slots=True)
class DMDRunSummary:
    initial_step: int
    final_step: int
    iterations: int
    student_optimizer_steps: int
    fake_score_optimizer_steps: int
    final_generator_loss: float
    final_fake_score_loss: float


@runtime_checkable
class DMDTrainingEngine(Protocol):
    """Execution surface shared by DMD and state-owning DMD composites."""

    global_step: int
    student_optimizer_steps: int
    fake_score_optimizer_steps: int
    gradient_accumulation_steps: int
    generator_update_interval: int

    def generator_update_due(self) -> bool: ...

    def train_step(
        self,
        batch: DMDTrainingBatch | tuple[DMDTrainingBatch, ...],
        *,
        fake_score_batch: DMDTrainingBatch | tuple[DMDTrainingBatch, ...] | None = None,
        generator: torch.Generator | None = None,
    ) -> DMDTrainResult: ...


class NativeDMDTrainingSession:
    """Drive a stateful latent loader through the native DMD engine."""

    step_event_schema = "worldfoundry-dmd-step-event"

    def __init__(
        self,
        engine: DMDTrainingEngine,
        dataloader: Iterable[DMDTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        fresh_fake_score_batches: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(engine, DMDTrainingEngine):
            raise TypeError("engine must implement DMDTrainingEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("DMD progress and engine global step differ")
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
        self.fresh_fake_score_batches = bool(fresh_fake_score_batches)
        self.event_sink = event_sink
        self._pending: list[PendingTrainingCheckpoint] = []
        self._iterator: Iterator[DMDTrainingBatch] | None = None

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
        boundary_every_steps: int = 0,
        boundary_sink: Callable[[int, int], None] | None = None,
    ) -> DMDRunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if isinstance(boundary_every_steps, bool) or int(boundary_every_steps) < 0:
            raise ValueError("boundary_every_steps must be non-negative")
        boundary_cadence = int(boundary_every_steps)
        if (boundary_sink is None) != (boundary_cadence == 0):
            raise ValueError("boundary_sink and a positive boundary_every_steps must be configured together")
        initial_step = self.engine.global_step
        final_result: DMDTrainResult | None = None

        def next_microbatch() -> DMDTrainingBatch:
            if self._iterator is None:
                self._iterator = iter(self.dataloader)
            try:
                value = next(self._iterator)
            except StopIteration:
                self._iterator = iter(self.dataloader)
                try:
                    value = next(self._iterator)
                except StopIteration as error:
                    raise RuntimeError("DMD dataloader is empty") from error
            if not isinstance(value, DMDTrainingBatch):
                raise TypeError("DMD dataloader must emit DMDTrainingBatch values")
            return value

        try:
            for _ in range(int(max_steps)):
                batches = [next_microbatch() for _ in range(self.engine.gradient_accumulation_steps)]
                generator_due = self.engine.generator_update_due()
                fake_batches = (
                    [next_microbatch() for _ in range(self.engine.gradient_accumulation_steps)]
                    if self.fresh_fake_score_batches and generator_due
                    else batches
                )
                consumed = batches + fake_batches if fake_batches is not batches else batches
                previous_step = self.engine.global_step
                result = self.engine.train_step(
                    tuple(batches),
                    fake_score_batch=(tuple(fake_batches) if fake_batches is not batches else None),
                    generator=generator,
                )
                self.progress.record_step(
                    microbatches=len(consumed),
                    samples=sum(batch.batch_size for batch in consumed),
                    latent_tokens=sum(_latent_tokens(batch.clean_latents) for batch in consumed),
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("DMD progress failed to commit with the engine")
                final_result = result
                self._emit(
                    {
                        "schema": self.step_event_schema,
                        "global_step": self.engine.global_step,
                        "microbatches": len(consumed),
                        "generator_microbatches": len(batches) if generator_due else 0,
                        "fake_score_microbatches": len(fake_batches),
                        "samples": sum(batch.batch_size for batch in consumed),
                        "generator_updated": result.generator_updated,
                        "generator_loss": float(result.generator_loss.item()),
                        "fake_score_loss": float(result.fake_score_loss.item()),
                    }
                )
                self._checkpoint_if_due()
                if boundary_cadence and self.engine.global_step // boundary_cadence > (
                    previous_step // boundary_cadence
                ):
                    # A pending checkpoint may be gathering the same live
                    # parameters. Finish its immutable commit before another
                    # collective full-state export starts.
                    self.wait_for_checkpoints()
                    assert boundary_sink is not None
                    boundary_sink(previous_step, self.engine.global_step)
        finally:
            self.wait_for_checkpoints()
        assert final_result is not None
        return DMDRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=int(max_steps),
            student_optimizer_steps=self.engine.student_optimizer_steps,
            fake_score_optimizer_steps=self.engine.fake_score_optimizer_steps,
            final_generator_loss=float(final_result.generator_loss.item()),
            final_fake_score_loss=float(final_result.fake_score_loss.item()),
        )


__all__ = ["DMDRunSummary", "DMDTrainingEngine", "NativeDMDTrainingSession"]
