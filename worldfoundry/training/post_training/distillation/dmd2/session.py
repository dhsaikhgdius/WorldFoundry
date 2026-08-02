"""Synchronous run session for native DMD2 distillation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import prod

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress

from .contracts import DMD2TrainingBatch
from .engine import DMD2TrainResult, NativeDMD2TrainEngine


def _latent_tokens(tensor: torch.Tensor) -> int:
    if tensor.ndim < 2:
        raise ValueError("latent tensor must include batch and feature dimensions")
    return int(tensor.shape[0]) * prod(int(size) for size in tensor.shape[2:])


@dataclass(frozen=True, slots=True)
class DMD2RunSummary:
    initial_step: int
    final_step: int
    iterations: int
    student_optimizer_steps: int
    guidance_optimizer_steps: int
    final_generator_loss: float
    final_guidance_loss: float


class NativeDMD2TrainingSession:
    """Drive atomic G→guidance iterations and checkpoint only committed state."""

    def __init__(
        self,
        engine: NativeDMD2TrainEngine,
        dataloader: Iterable[DMD2TrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(engine, NativeDMD2TrainEngine):
            raise TypeError("engine must be NativeDMD2TrainEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("DMD2 progress and engine global step differ")
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
    ) -> DMD2RunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        initial_step = self.engine.global_step
        iterator = iter(self.dataloader)
        final_result: DMD2TrainResult | None = None
        try:
            for _ in range(int(max_steps)):
                batches: list[DMD2TrainingBatch] = []
                for _ in range(self.engine.gradient_accumulation_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(self.dataloader)
                        try:
                            batch = next(iterator)
                        except StopIteration as error:
                            raise RuntimeError("DMD2 dataloader is empty") from error
                    if not isinstance(batch, DMD2TrainingBatch):
                        raise TypeError("DMD2 dataloader must emit DMD2TrainingBatch values")
                    batches.append(batch)
                final_result = self.engine.train_step(tuple(batches), generator=generator)
                samples = sum(batch.batch_size for batch in batches)
                self.progress.record_step(
                    microbatches=len(batches),
                    samples=samples,
                    latent_tokens=sum(_latent_tokens(batch.real_latents) for batch in batches),
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("DMD2 progress failed to commit with the engine")
                if self.event_sink is not None:
                    self.event_sink(
                        {
                            "schema": "worldfoundry-dmd2-step-event",
                            "global_step": self.engine.global_step,
                            "generator_updated": final_result.generator_updated,
                            "microbatches": len(batches),
                            "samples": samples,
                            "generator_loss": float(final_result.generator_loss.item()),
                            "guidance_loss": float(final_result.guidance_loss.item()),
                        }
                    )
                self._checkpoint_if_due()
        finally:
            self.wait_for_checkpoints()
        assert final_result is not None
        return DMD2RunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=int(max_steps),
            student_optimizer_steps=self.engine.student_optimizer_steps,
            guidance_optimizer_steps=self.engine.guidance_optimizer_steps,
            final_generator_loss=float(final_result.generator_loss.item()),
            final_guidance_loss=float(final_result.guidance_loss.item()),
        )


__all__ = ["DMD2RunSummary", "NativeDMD2TrainingSession"]
