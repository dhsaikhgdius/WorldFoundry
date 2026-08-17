"""Synchronous exact-boundary session for native SiD."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.post_training.shared.batching import latent_token_count as _latent_tokens

from .contracts import SIDTrainingBatch
from .engine import NativeSIDTrainEngine, SIDTrainResult


@dataclass(frozen=True, slots=True)
class SIDRunSummary:
    initial_step: int
    final_step: int
    iterations: int
    student_optimizer_steps: int
    fake_score_optimizer_steps: int
    final_generator_loss: float
    final_fake_score_loss: float


class NativeSIDTrainingSession:
    def __init__(
        self,
        engine: NativeSIDTrainEngine,
        dataloader: Iterable[SIDTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(engine, NativeSIDTrainEngine):
            raise TypeError("engine must be NativeSIDTrainEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("SiD progress and engine global step differ")
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
    ) -> SIDRunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if isinstance(boundary_every_steps, bool) or int(boundary_every_steps) < 0:
            raise ValueError("boundary_every_steps must be non-negative")
        if (boundary_sink is None) != (int(boundary_every_steps) == 0):
            raise ValueError("boundary_sink and a positive boundary_every_steps must be configured together")
        initial_step = self.engine.global_step
        iterator = iter(self.dataloader)
        final_result: SIDTrainResult | None = None

        def next_microbatches(*, role: str) -> list[SIDTrainingBatch]:
            nonlocal iterator
            batches: list[SIDTrainingBatch] = []
            for _ in range(self.engine.gradient_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(self.dataloader)
                    try:
                        batch = next(iterator)
                    except StopIteration as error:
                        raise RuntimeError("SiD dataloader is empty") from error
                if not isinstance(batch, SIDTrainingBatch):
                    raise TypeError(f"SiD {role} dataloader value must be SIDTrainingBatch")
                batches.append(batch)
            return batches

        try:
            for _ in range(int(max_steps)):
                fake_batches = next_microbatches(role="fake-score")
                generator_batches = next_microbatches(role="generator")
                previous_step = self.engine.global_step
                final_result = self.engine.train_step(
                    tuple(fake_batches),
                    tuple(generator_batches),
                    generator=generator,
                )
                consumed_batches = (*fake_batches, *generator_batches)
                samples = sum(batch.batch_size for batch in consumed_batches)
                self.progress.record_step(
                    microbatches=len(consumed_batches),
                    samples=samples,
                    latent_tokens=sum(
                        _latent_tokens(batch.latent_template) for batch in consumed_batches
                    ),
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("SiD progress failed to commit with the engine")
                if self.event_sink is not None:
                    self.event_sink(
                        {
                            "schema": "worldfoundry-sid-step-event",
                            "global_step": self.engine.global_step,
                            "target_index": final_result.target_index,
                            "microbatches_per_role": len(fake_batches),
                            "samples": samples,
                            "generator_loss": float(final_result.generator_loss.item()),
                            "fake_score_loss": float(final_result.fake_score_loss.item()),
                        }
                    )
                self._checkpoint_if_due()
                if boundary_every_steps and self.engine.global_step // int(
                    boundary_every_steps
                ) > previous_step // int(boundary_every_steps):
                    assert boundary_sink is not None
                    boundary_sink(previous_step, self.engine.global_step)
        finally:
            self.wait_for_checkpoints()
        assert final_result is not None
        return SIDRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=int(max_steps),
            student_optimizer_steps=self.engine.student_optimizer_steps,
            fake_score_optimizer_steps=self.engine.fake_score_optimizer_steps,
            final_generator_loss=float(final_result.generator_loss.item()),
            final_fake_score_loss=float(final_result.fake_score_loss.item()),
        )


__all__ = ["NativeSIDTrainingSession", "SIDRunSummary"]
