"""Synchronous sessions for native AnyFlow training."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod

import torch

from worldfoundry.training.checkpoint.checkpointer import (
    PendingTrainingCheckpoint,
    TrainingCheckpointer,
)
from worldfoundry.training.checkpoint.state import TrainingProgress

from ...shared.session import NativeSingleOptimizerTrainingSession
from .contracts import AnyFlowTrainingBatch
from .engine import AnyFlowOnPolicyResult, NativeAnyFlowOnPolicyEngine


def _latent_tokens(batch: object) -> int:
    if not isinstance(batch, AnyFlowTrainingBatch):
        raise TypeError("batch must be AnyFlowTrainingBatch")
    value = batch.clean_latents
    if not isinstance(value, torch.Tensor):
        raise TypeError("AnyFlow clean_latents must be a torch.Tensor")
    return int(value.shape[0]) * prod(int(size) for size in value.shape[2:])


class NativeAnyFlowPretrainingSession(NativeSingleOptimizerTrainingSession):
    """Run FAR or bidirectional FlowMap pretraining at arbitrary world size."""

    def __init__(self, engine, dataloader, progress, **kwargs) -> None:
        super().__init__(
            engine,
            dataloader,
            progress,
            batch_type=AnyFlowTrainingBatch,
            batch_size=lambda batch: batch.batch_size,
            latent_tokens=_latent_tokens,
            event_schema="worldfoundry-anyflow-pretrain-step-event",
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class AnyFlowOnPolicyRunSummary:
    initial_step: int
    final_step: int
    iterations: int
    student_optimizer_steps: int
    fake_score_optimizer_steps: int
    final_generator_loss: float
    final_fake_score_loss: float


class NativeAnyFlowOnPolicyTrainingSession:
    """Bind independent fresh generator/fake batches to one atomic iteration."""

    def __init__(
        self,
        engine: NativeAnyFlowOnPolicyEngine,
        dataloader: Iterable[AnyFlowTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
    ) -> None:
        if not isinstance(engine, NativeAnyFlowOnPolicyEngine):
            raise TypeError("engine must be NativeAnyFlowOnPolicyEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("AnyFlow progress and engine global step differ")
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
    ) -> AnyFlowOnPolicyRunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        initial_step = self.engine.global_step
        iterator = iter(self.dataloader)
        final_result: AnyFlowOnPolicyResult | None = None

        def next_batch() -> AnyFlowTrainingBatch:
            nonlocal iterator
            try:
                value = next(iterator)
            except StopIteration:
                iterator = iter(self.dataloader)
                try:
                    value = next(iterator)
                except StopIteration as error:
                    raise RuntimeError("AnyFlow dataloader is empty") from error
            if not isinstance(value, AnyFlowTrainingBatch):
                raise TypeError("AnyFlow dataloader must emit typed batches")
            return value

        try:
            for _ in range(int(max_steps)):
                student_batches = tuple(next_batch() for _ in range(self.engine.gradient_accumulation_steps))
                fake_count = self.engine.discriminator_update_ratio * self.engine.gradient_accumulation_steps
                fake_batches = tuple(next_batch() for _ in range(fake_count))
                final_result = self.engine.train_step(
                    student_batches,
                    fake_score_batches=fake_batches,
                    generator=generator,
                )
                consumed = (*student_batches, *fake_batches)
                self.progress.record_step(
                    microbatches=len(consumed),
                    samples=sum(batch.batch_size for batch in consumed),
                    latent_tokens=sum(_latent_tokens(batch) for batch in consumed),
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("AnyFlow progress failed to commit with the engine")
                self._checkpoint_if_due()
        finally:
            self.wait_for_checkpoints()
        assert final_result is not None
        return AnyFlowOnPolicyRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=int(max_steps),
            student_optimizer_steps=self.engine.student_optimizer_steps,
            fake_score_optimizer_steps=self.engine.fake_score_optimizer_steps,
            final_generator_loss=float(final_result.generator_loss.item()),
            final_fake_score_loss=float(final_result.fake_score_loss.item()),
        )


__all__ = [
    "AnyFlowOnPolicyRunSummary",
    "NativeAnyFlowOnPolicyTrainingSession",
    "NativeAnyFlowPretrainingSession",
]
