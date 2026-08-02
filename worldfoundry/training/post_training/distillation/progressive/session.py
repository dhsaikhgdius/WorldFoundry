"""Exact-boundary session for progressive distillation stages."""

from __future__ import annotations

from math import prod

import torch

from ...shared.session import (
    NativeSingleOptimizerTrainingSession,
    SingleOptimizerRunSummary,
)
from .contracts import ProgressiveDistillationBatch
from .engine import NativeProgressiveDistillationTrainEngine

ProgressiveDistillationRunSummary = SingleOptimizerRunSummary


def _latent_tokens(batch: object) -> int:
    if not isinstance(batch, ProgressiveDistillationBatch):
        raise TypeError("batch must be ProgressiveDistillationBatch")
    value = batch.clean_latents
    if not isinstance(value, torch.Tensor):
        raise TypeError("progressive clean_latents must be a torch.Tensor")
    return int(value.shape[0]) * prod(int(size) for size in value.shape[2:])


class NativeProgressiveDistillationTrainingSession(
    NativeSingleOptimizerTrainingSession
):
    def __init__(self, engine, dataloader, progress, **kwargs) -> None:
        if not isinstance(engine, NativeProgressiveDistillationTrainEngine):
            raise TypeError(
                "engine must be NativeProgressiveDistillationTrainEngine"
            )
        super().__init__(
            engine,
            dataloader,
            progress,
            batch_type=ProgressiveDistillationBatch,
            batch_size=lambda batch: batch.batch_size,
            latent_tokens=_latent_tokens,
            event_schema="worldfoundry-progressive-distillation-step-event",
            **kwargs,
        )

    def run(self, *, max_steps: int) -> ProgressiveDistillationRunSummary:
        if self.engine.is_complete:
            raise RuntimeError("progressive distillation is already complete")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be positive")
        return super().run(
            max_steps=min(max_steps, self.engine.remaining_optimizer_steps)
        )


__all__ = [
    "NativeProgressiveDistillationTrainingSession",
    "ProgressiveDistillationRunSummary",
]
