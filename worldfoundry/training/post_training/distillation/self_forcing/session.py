"""Self-Forcing session with fresh generator and critic prompt batches."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress

from ..dmd.contracts import DMDTrainingBatch
from ..dmd.engine import NativeDMDTrainEngine
from ..dmd.session import NativeDMDTrainingSession


class NativeSelfForcingTrainingSession(NativeDMDTrainingSession):
    """Use a fresh prompt batch and fresh rollout for each active DMD role."""

    step_event_schema = "worldfoundry-self-forcing-step-event"

    def __init__(
        self,
        engine: NativeDMDTrainEngine,
        dataloader: Iterable[DMDTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        super().__init__(
            engine,
            dataloader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=save_every_steps,
            asynchronous_checkpoints=asynchronous_checkpoints,
            fresh_fake_score_batches=True,
            event_sink=event_sink,
        )


__all__ = ["NativeSelfForcingTrainingSession"]
