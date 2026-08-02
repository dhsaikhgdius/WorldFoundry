"""Shared exact-boundary session for one-optimizer post-training engines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress


@runtime_checkable
class SingleOptimizerEngine(Protocol):
    global_step: int
    optimizer_steps: int
    gradient_accumulation_steps: int

    def train_step(self, batch: object) -> object: ...


@dataclass(frozen=True, slots=True)
class SingleOptimizerRunSummary:
    initial_step: int
    final_step: int
    iterations: int
    optimizer_steps: int
    final_loss: float


class NativeSingleOptimizerTrainingSession:
    """Drive a typed loader without duplicating checkpoint/session cadence."""

    def __init__(
        self,
        engine: SingleOptimizerEngine,
        dataloader: Iterable[object],
        progress: TrainingProgress,
        *,
        batch_type: type,
        batch_size: Callable[[object], int],
        latent_tokens: Callable[[object], int],
        event_schema: str,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(engine, SingleOptimizerEngine):
            raise TypeError("engine must implement SingleOptimizerEngine")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError("progress and engine global step differ")
        if not isinstance(batch_type, type):
            raise TypeError("batch_type must be a type")
        if not callable(batch_size) or not callable(latent_tokens):
            raise TypeError("batch accounting callbacks must be callable")
        schema = str(event_schema).strip()
        if not schema:
            raise ValueError("event_schema must be non-empty")
        if isinstance(save_every_steps, bool) or int(save_every_steps) < 0:
            raise ValueError("save_every_steps must be non-negative")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError("checkpoint cadence requires checkpoint_state and checkpointer")
        self.engine = engine
        self.dataloader = dataloader
        self.progress = progress
        self.batch_type = batch_type
        self.batch_size = batch_size
        self.latent_tokens = latent_tokens
        self.event_schema = schema
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

    def run(self, *, max_steps: int) -> SingleOptimizerRunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        initial_step = self.engine.global_step
        iterator = iter(self.dataloader)
        final_loss: torch.Tensor | None = None

        def next_batch() -> object:
            nonlocal iterator
            try:
                value = next(iterator)
            except StopIteration:
                iterator = iter(self.dataloader)
                try:
                    value = next(iterator)
                except StopIteration as error:
                    raise RuntimeError("post-training dataloader is empty") from error
            if not isinstance(value, self.batch_type):
                raise TypeError(
                    f"dataloader must emit {self.batch_type.__name__} values"
                )
            return value

        try:
            for _ in range(int(max_steps)):
                batches = tuple(
                    next_batch()
                    for _ in range(self.engine.gradient_accumulation_steps)
                )
                payload: object = batches[0] if len(batches) == 1 else batches
                result = self.engine.train_step(payload)
                loss = getattr(result, "loss", None)
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise TypeError("single-optimizer result must expose one scalar loss tensor")
                final_loss = loss.detach().float().reshape(())
                samples = sum(self.batch_size(batch) for batch in batches)
                tokens = sum(self.latent_tokens(batch) for batch in batches)
                self.progress.record_step(
                    microbatches=len(batches),
                    samples=samples,
                    latent_tokens=tokens,
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("progress failed to commit with the engine")
                if self.event_sink is not None:
                    self.event_sink(
                        {
                            "schema": self.event_schema,
                            "global_step": self.engine.global_step,
                            "microbatches": len(batches),
                            "samples": samples,
                            "latent_tokens": tokens,
                            "loss": float(final_loss.item()),
                        }
                    )
                self._checkpoint_if_due()
        finally:
            self.wait_for_checkpoints()
        assert final_loss is not None
        return SingleOptimizerRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=int(max_steps),
            optimizer_steps=self.engine.optimizer_steps,
            final_loss=float(final_loss.item()),
        )


__all__ = [
    "NativeSingleOptimizerTrainingSession",
    "SingleOptimizerEngine",
    "SingleOptimizerRunSummary",
]
