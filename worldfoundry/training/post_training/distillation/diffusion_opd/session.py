"""Training session for accumulated homogeneous-domain DiffusionOPD rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress

from .batching import NativeDiffusionOPDDataLoader
from .contracts import DiffusionOPDTrajectory
from .engine import DiffusionOPDTrainResult, NativeDiffusionOPDEngine
from .trajectory import DiffusionOPDTrajectorySampler


@dataclass(frozen=True, slots=True)
class DiffusionOPDIterationResult:
    """Student trajectories and their committed teacher-matching update."""

    trajectories: tuple[DiffusionOPDTrajectory, ...]
    update: DiffusionOPDTrainResult


class NativeDiffusionOPDTrainingSession:
    """Own rollout accumulation, optimizer boundaries, and checkpoint cadence."""

    def __init__(
        self,
        *,
        sampler: DiffusionOPDTrajectorySampler,
        engine: NativeDiffusionOPDEngine,
        dataloader: NativeDiffusionOPDDataLoader,
        progress: TrainingProgress,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
    ) -> None:
        if not isinstance(sampler, DiffusionOPDTrajectorySampler):
            raise TypeError("sampler must be DiffusionOPDTrajectorySampler")
        if not isinstance(engine, NativeDiffusionOPDEngine):
            raise TypeError("engine must be NativeDiffusionOPDEngine")
        if not isinstance(dataloader, NativeDiffusionOPDDataLoader):
            raise TypeError("dataloader must be NativeDiffusionOPDDataLoader")
        if not isinstance(progress, TrainingProgress) or progress.optimizer_steps != engine.global_step:
            raise ValueError("DiffusionOPD progress must match engine global_step")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError("checkpoint cadence requires state and checkpointer")
        self.sampler = sampler
        self.engine = engine
        self.dataloader = dataloader
        self.progress = progress
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.save_every_steps = int(save_every_steps)
        self.asynchronous_checkpoints = bool(asynchronous_checkpoints)
        self._pending: list[PendingTrainingCheckpoint] = []

    def wait_for_checkpoints(self) -> None:
        for pending in self._pending:
            pending.wait()
        self._pending.clear()

    def train_iteration(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> DiffusionOPDIterationResult:
        trajectories = tuple(
            self.sampler.sample(next(self.dataloader), generator=generator)
            for _ in range(self.engine.gradient_accumulation_steps)
        )
        update = self.engine.train_step(trajectories)
        latent_tokens = sum(
            trajectory.batch_size * trajectory.selected_steps * prod(int(size) for size in trajectory.latents.shape[2:])
            for trajectory in trajectories
        )
        self.progress.record_step(
            microbatches=update.microbatches,
            samples=update.sample_count,
            latent_tokens=latent_tokens,
        )
        if self.progress.optimizer_steps != self.engine.global_step:
            raise RuntimeError("DiffusionOPD progress failed to commit with engine")
        if self.save_every_steps and not self.engine.global_step % self.save_every_steps:
            assert self.checkpointer is not None and self.checkpoint_state is not None
            artifact = self.checkpointer.save(
                self.checkpoint_state,
                asynchronous=self.asynchronous_checkpoints,
            )
            if isinstance(artifact, PendingTrainingCheckpoint):
                self._pending.append(artifact)
        return DiffusionOPDIterationResult(
            trajectories=trajectories,
            update=update,
        )


__all__ = [
    "DiffusionOPDIterationResult",
    "NativeDiffusionOPDTrainingSession",
]
