"""Reward-Forcing session with fresh generator and fake-score rollouts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress

from ..dmd.session import NativeDMDTrainingSession
from .contracts import RewardForcingTrainingBatch
from .engine import NativeRewardForcingTrainEngine


class NativeRewardForcingTrainingSession(NativeDMDTrainingSession):
    """Consume independent prompt batches for every active Re-DMD role."""

    step_event_schema = "worldfoundry-reward-forcing-step-event"

    def __init__(
        self,
        engine: NativeRewardForcingTrainEngine,
        dataloader: Iterable[RewardForcingTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(engine, NativeRewardForcingTrainEngine):
            raise TypeError("Reward-Forcing session requires NativeRewardForcingTrainEngine")
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


__all__ = ["NativeRewardForcingTrainingSession"]
