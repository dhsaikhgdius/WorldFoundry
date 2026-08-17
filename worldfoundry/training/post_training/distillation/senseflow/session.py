"""Synchronous run session for native SenseFlow distillation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.post_training.shared.batching import latent_token_count as _latent_tokens

from .contracts import SenseFlowTrainingBatch
from .engine import NativeSenseFlowTrainEngine, SenseFlowTrainResult


@dataclass(frozen=True, slots=True)
class SenseFlowRunSummary:
    initial_step: int
    final_step: int
    iterations: int
    student_optimizer_steps: int
    fake_score_optimizer_steps: int
    discriminator_optimizer_steps: int
    ida_updates: int
    final_generator_loss: float
    final_fake_score_loss: float
    final_discriminator_loss: float


class NativeSenseFlowTrainingSession:
    """Drive atomic generator→IDA→fake-score→discriminator iterations."""

    def __init__(
        self,
        engine: NativeSenseFlowTrainEngine,
        dataloader: Iterable[SenseFlowTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(engine, NativeSenseFlowTrainEngine):
            raise TypeError("engine must be NativeSenseFlowTrainEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("SenseFlow progress and engine global step differ")
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

    def run(self, *, max_steps: int) -> SenseFlowRunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        initial_step = self.engine.global_step
        iterator = iter(self.dataloader)
        final_result: SenseFlowTrainResult | None = None
        try:
            for _ in range(int(max_steps)):
                batches: list[SenseFlowTrainingBatch] = []
                for _ in range(self.engine.gradient_accumulation_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(self.dataloader)
                        try:
                            batch = next(iterator)
                        except StopIteration as error:
                            raise RuntimeError("SenseFlow dataloader is empty") from error
                    if not isinstance(batch, SenseFlowTrainingBatch):
                        raise TypeError(
                            "SenseFlow dataloader must emit SenseFlowTrainingBatch values"
                        )
                    batches.append(batch)
                final_result = self.engine.train_step(tuple(batches))
                samples = sum(batch.batch_size for batch in batches)
                self.progress.record_step(
                    microbatches=len(batches),
                    samples=samples,
                    latent_tokens=sum(_latent_tokens(batch.real_latents) for batch in batches),
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("SenseFlow progress failed to commit with the engine")
                if self.event_sink is not None:
                    self.event_sink(
                        {
                            "schema": "worldfoundry-senseflow-step-event",
                            "global_step": self.engine.global_step,
                            "generator_updated": final_result.generator_updated,
                            "microbatches": len(batches),
                            "samples": samples,
                            "generator_loss": float(final_result.generator_loss.item()),
                            "fake_score_loss": float(final_result.fake_score_loss.item()),
                            "discriminator_loss": float(
                                final_result.discriminator_loss.item()
                            ),
                        }
                    )
                self._checkpoint_if_due()
        finally:
            self.wait_for_checkpoints()
        assert final_result is not None
        return SenseFlowRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=int(max_steps),
            student_optimizer_steps=self.engine.student_optimizer_steps,
            fake_score_optimizer_steps=self.engine.fake_score_optimizer_steps,
            discriminator_optimizer_steps=self.engine.discriminator_optimizer_steps,
            ida_updates=self.engine.ida_updates,
            final_generator_loss=float(final_result.generator_loss.item()),
            final_fake_score_loss=float(final_result.fake_score_loss.item()),
            final_discriminator_loss=float(final_result.discriminator_loss.item()),
        )


__all__ = [
    "NativeSenseFlowTrainingSession",
    "SenseFlowRunSummary",
]
