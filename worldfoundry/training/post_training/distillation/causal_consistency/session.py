"""Shared-cadence session binding for causal consistency batches."""

from __future__ import annotations

from math import prod

import torch

from ...shared.session import NativeSingleOptimizerTrainingSession
from .contracts import CausalConsistencyTrainingBatch


def _latent_tokens(batch: object) -> int:
    assert isinstance(batch, CausalConsistencyTrainingBatch)
    value = batch.clean_latents
    if not isinstance(value, torch.Tensor):
        raise TypeError("causal consistency latent must be a torch.Tensor")
    return int(value.shape[0]) * prod(int(size) for size in value.shape[2:])


class NativeCausalConsistencyTrainingSession(
    NativeSingleOptimizerTrainingSession
):
    def __init__(self, engine, dataloader, progress, **kwargs) -> None:
        super().__init__(
            engine,
            dataloader,
            progress,
            batch_type=CausalConsistencyTrainingBatch,
            batch_size=lambda batch: batch.batch_size,
            latent_tokens=_latent_tokens,
            event_schema="worldfoundry-causal-consistency-step-event",
            **kwargs,
        )


__all__ = ["NativeCausalConsistencyTrainingSession"]
