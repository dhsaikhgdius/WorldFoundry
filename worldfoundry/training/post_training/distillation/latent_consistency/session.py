"""Shared-cadence session binding for latent consistency batches."""

from __future__ import annotations

from math import prod

import torch

from ...shared.session import NativeSingleOptimizerTrainingSession
from .contracts import LatentConsistencyTrainingBatch


def _latent_tokens(batch: object) -> int:
    if not isinstance(batch, LatentConsistencyTrainingBatch):
        raise TypeError("batch must be LatentConsistencyTrainingBatch")
    value = batch.clean_latents
    if not isinstance(value, torch.Tensor):
        raise TypeError("latent consistency clean_latents must be a torch.Tensor")
    return int(value.shape[0]) * prod(int(size) for size in value.shape[2:])


class NativeLatentConsistencyTrainingSession(NativeSingleOptimizerTrainingSession):
    def __init__(self, engine, dataloader, progress, **kwargs) -> None:
        super().__init__(
            engine,
            dataloader,
            progress,
            batch_type=LatentConsistencyTrainingBatch,
            batch_size=lambda batch: batch.batch_size,
            latent_tokens=_latent_tokens,
            event_schema="worldfoundry-latent-consistency-step-event",
            **kwargs,
        )


__all__ = ["NativeLatentConsistencyTrainingSession"]
