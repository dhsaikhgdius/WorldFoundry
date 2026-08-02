"""Two-stream synchronous session for adaptive video distillation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import prod

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress

from ..dmd.session import DMDRunSummary
from .batching import NativeAdaptiveVideoDataLoader
from .contracts import AdaptiveVideoTrainingBatch
from .engine import NativeAdaptiveVideoTrainEngine


def _latent_tokens(tensor: torch.Tensor) -> int:
    if tensor.ndim < 3:
        raise ValueError("adaptive video latents must have shape [B,F,...]")
    return int(tensor.shape[0]) * prod(int(size) for size in tensor.shape[2:])


class NativeAdaptiveVideoTrainingSession:
    """Advance real-video data only when the generator is scheduled."""

    step_event_schema = "worldfoundry-adaptive-video-step-event"

    def __init__(
        self,
        engine: NativeAdaptiveVideoTrainEngine,
        dataloader: NativeAdaptiveVideoDataLoader,
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(engine, NativeAdaptiveVideoTrainEngine):
            raise TypeError("engine must be NativeAdaptiveVideoTrainEngine")
        if not isinstance(dataloader, NativeAdaptiveVideoDataLoader):
            raise TypeError("dataloader must be NativeAdaptiveVideoDataLoader")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if progress.optimizer_steps != engine.global_step:
            raise ValueError(
                "adaptive video progress and engine global step differ"
            )
        if isinstance(save_every_steps, bool) or int(save_every_steps) < 0:
            raise ValueError("save_every_steps must be non-negative")
        if save_every_steps and (checkpoint_state is None or checkpointer is None):
            raise ValueError(
                "checkpoint cadence requires checkpoint_state and checkpointer"
            )
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
    ) -> DMDRunSummary:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if isinstance(boundary_every_steps, bool) or int(boundary_every_steps) < 0:
            raise ValueError("boundary_every_steps must be non-negative")
        boundary_cadence = int(boundary_every_steps)
        if (boundary_sink is None) != (boundary_cadence == 0):
            raise ValueError(
                "boundary_sink and positive boundary_every_steps must be configured together"
            )
        initial_step = self.engine.global_step
        final_result = None
        try:
            for _ in range(int(max_steps)):
                previous_step = self.engine.global_step
                generator_due = (
                    previous_step % self.engine.generator_update_interval == 0
                )
                batches = []
                generated_samples = 0
                real_samples = 0
                latent_tokens = 0
                for _ in range(self.engine.gradient_accumulation_steps):
                    generated = self.dataloader.next_generated()
                    generated_samples += generated.batch_size
                    if not isinstance(generated.clean_latents, torch.Tensor):
                        raise TypeError(
                            "adaptive generated clean_latents must be a torch.Tensor"
                        )
                    latent_tokens += _latent_tokens(generated.clean_latents)
                    if generator_due:
                        real = self.dataloader.next_real()
                        real_samples += real.batch_size
                        if not isinstance(real.latents, torch.Tensor):
                            raise TypeError(
                                "adaptive real latents must be a torch.Tensor"
                            )
                        latent_tokens += _latent_tokens(real.latents)
                        batches.append(
                            AdaptiveVideoTrainingBatch.combine(generated, real)
                        )
                    else:
                        batches.append(generated)
                result = self.engine.train_step(
                    tuple(batches),
                    generator=generator,
                )
                self.progress.record_step(
                    microbatches=len(batches) + (
                        len(batches) if generator_due else 0
                    ),
                    samples=generated_samples + real_samples,
                    latent_tokens=latent_tokens,
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError(
                        "adaptive video progress failed to commit with the engine"
                    )
                final_result = result
                if self.event_sink is not None:
                    self.event_sink(
                        {
                            "schema": self.step_event_schema,
                            "global_step": self.engine.global_step,
                            "generator_updated": result.generator_updated,
                            "generated_microbatches": len(batches),
                            "real_microbatches": len(batches) if generator_due else 0,
                            "generated_samples": generated_samples,
                            "real_samples": real_samples,
                            "generator_loss": float(result.generator_loss.item()),
                            "fake_score_loss": float(result.fake_score_loss.item()),
                        }
                    )
                self._checkpoint_if_due()
                if boundary_cadence and self.engine.global_step // boundary_cadence > (
                    previous_step // boundary_cadence
                ):
                    self.wait_for_checkpoints()
                    assert boundary_sink is not None
                    boundary_sink(previous_step, self.engine.global_step)
        finally:
            self.wait_for_checkpoints()
        assert final_result is not None
        return DMDRunSummary(
            initial_step=initial_step,
            final_step=self.engine.global_step,
            iterations=int(max_steps),
            student_optimizer_steps=self.engine.student_optimizer_steps,
            fake_score_optimizer_steps=self.engine.fake_score_optimizer_steps,
            final_generator_loss=float(final_result.generator_loss.item()),
            final_fake_score_loss=float(final_result.fake_score_loss.item()),
        )


__all__ = ["NativeAdaptiveVideoTrainingSession"]
