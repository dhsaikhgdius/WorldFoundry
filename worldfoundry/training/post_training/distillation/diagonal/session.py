"""Unified synchronous session for native diagonal distillation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress

from ..dmd.contracts import DMDTrainingBatch
from ..dmd.session import NativeDMDTrainingSession
from .engine import NativeDiagonalTrainEngine


class NativeDiagonalTrainingSession(NativeDMDTrainingSession):
    """Reuse the scalable DMD loader/checkpoint cadence with diagonal state."""

    step_event_schema = "worldfoundry-diagonal-step-event"

    def __init__(
        self,
        engine: NativeDiagonalTrainEngine,
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
        if not isinstance(engine, NativeDiagonalTrainEngine):
            raise TypeError("engine must be NativeDiagonalTrainEngine")
        super().__init__(
            engine,
            dataloader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=save_every_steps,
            asynchronous_checkpoints=asynchronous_checkpoints,
            fresh_fake_score_batches=fresh_fake_score_batches,
            event_sink=event_sink,
        )


__all__ = ["NativeDiagonalTrainingSession"]
