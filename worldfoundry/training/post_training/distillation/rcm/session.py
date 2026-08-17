"""Synchronous training session for native rCM engines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.post_training.shared.batching import latent_token_count as _latent_tokens

from .contracts import RCMTrainingBatch
from .engine import NativeRCMTrainEngine, RCMTrainResult


@dataclass(frozen=True, slots=True)
class RCMRunSummary:
    initial_step: int
    final_step: int
    updates: int
    student_optimizer_steps: int
    fake_score_optimizer_steps: int
    final_student_loss: float | None
    final_fake_score_loss: float | None


class NativeRCMTrainingSession:
    """Consume one fresh batch group per committed rCM phase."""

    def __init__(
        self,
        engine: NativeRCMTrainEngine,
        dataloader: Iterable[RCMTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(engine, NativeRCMTrainEngine):
            raise TypeError("engine must be NativeRCMTrainEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("rCM progress and engine global step differ")
        if isinstance(save_every_steps, bool) or int(save_every_steps) < 0:
            raise ValueError("save_every_steps must be non-negative")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError("checkpoint cadence requires checkpoint_state and checkpointer")
        if not isinstance(asynchronous_checkpoints, bool):
            raise TypeError("asynchronous_checkpoints must be a bool")
        self.engine = engine
        self.dataloader = dataloader
        self.progress = progress
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.save_every_steps = int(save_every_steps)
        self.asynchronous_checkpoints = asynchronous_checkpoints
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

    def _next_microbatches(
        self,
        iterator: object,
    ) -> tuple[tuple[RCMTrainingBatch, ...], object]:
        batches: list[RCMTrainingBatch] = []
        active = iterator
        for _ in range(self.engine.gradient_accumulation_steps):
            try:
                batch = next(active)  # type: ignore[arg-type]
            except StopIteration:
                active = iter(self.dataloader)
                try:
                    batch = next(active)
                except StopIteration as error:
                    raise RuntimeError("rCM dataloader is empty") from error
            if not isinstance(batch, RCMTrainingBatch):
                raise TypeError("rCM dataloader must emit RCMTrainingBatch values")
            batches.append(batch)
        return tuple(batches), active

    def run(
        self,
        *,
        max_steps: int,
        generator: torch.Generator | None = None,
        boundary_every_steps: int = 0,
        boundary_sink: Callable[[int, int], None] | None = None,
    ) -> RCMRunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        if isinstance(boundary_every_steps, bool) or int(boundary_every_steps) < 0:
            raise ValueError("boundary_every_steps must be non-negative")
        if (boundary_sink is None) != (int(boundary_every_steps) == 0):
            raise ValueError("boundary_sink and a positive boundary_every_steps must be configured together")
        initial_step = self.engine.global_step
        iterator: object = iter(self.dataloader)
        final_student: RCMTrainResult | None = None
        final_fake: RCMTrainResult | None = None
        try:
            for _ in range(int(max_steps)):
                batches, iterator = self._next_microbatches(iterator)
                previous_step = self.engine.global_step
                result = self.engine.train_step(batches, generator=generator)
                self.progress.record_step(
                    microbatches=len(batches),
                    samples=sum(batch.batch_size for batch in batches),
                    latent_tokens=sum(_latent_tokens(batch.clean_latents) for batch in batches),
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("rCM progress failed to commit with the engine")
                if result.phase == "student":
                    final_student = result
                else:
                    final_fake = result
                if self.event_sink is not None:
                    self.event_sink(
                        {
                            "schema": "worldfoundry-rcm-step-event",
                            "global_step": self.engine.global_step,
                            "phase": result.phase,
                            "microbatches": len(batches),
                            "samples": sum(batch.batch_size for batch in batches),
                            "loss": float(result.loss.item()),
                        }
                    )
                self._checkpoint_if_due()
                if boundary_every_steps and self.engine.global_step // boundary_every_steps > (
                    previous_step // boundary_every_steps
                ):
                    assert boundary_sink is not None
                    boundary_sink(previous_step, self.engine.global_step)
        finally:
            self.wait_for_checkpoints()
        return RCMRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            updates=int(max_steps),
            student_optimizer_steps=self.engine.student_optimizer_steps,
            fake_score_optimizer_steps=self.engine.fake_score_optimizer_steps,
            final_student_loss=(
                None if final_student is None else float(final_student.loss.item())
            ),
            final_fake_score_loss=(
                None if final_fake is None else float(final_fake.loss.item())
            ),
        )


__all__ = ["NativeRCMTrainingSession", "RCMRunSummary"]
