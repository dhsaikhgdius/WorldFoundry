"""Shared-cadence session binding for causal ODE batches."""

from __future__ import annotations

from math import prod

import torch

from ...shared.session import NativeSingleOptimizerTrainingSession
from .contracts import CausalODETrainingBatch


def _latent_tokens(batch: object) -> int:
    assert isinstance(batch, CausalODETrainingBatch)
    value = batch.ode_trajectories
    if not isinstance(value, torch.Tensor):
        raise TypeError("causal ODE trajectory must be a torch.Tensor")
    return int(value.shape[0]) * prod(int(size) for size in value.shape[3:])


class NativeCausalODETrainingSession(NativeSingleOptimizerTrainingSession):
    def __init__(self, engine, dataloader, progress, **kwargs) -> None:
        super().__init__(
            engine,
            dataloader,
            progress,
            batch_type=CausalODETrainingBatch,
            batch_size=lambda batch: batch.batch_size,
            latent_tokens=_latent_tokens,
            event_schema="worldfoundry-causal-ode-step-event",
            **kwargs,
        )


__all__ = ["NativeCausalODETrainingSession"]
