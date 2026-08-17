"""Cached Wan latents prepared for native AnyFlow training."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType

import torch

from worldfoundry.training.api.contracts import TrainingBatch
from worldfoundry.training.post_training.distillation.anyflow.contracts import (
    AnyFlowTrainingBatch,
)
from worldfoundry.training.post_training.shared.batching import (
    batch_shared_conditioning,
)

ANYFLOW_DATA_LOADER_STATE_SCHEMA = "worldfoundry-anyflow-data-loader"


def _to_device(
    value: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> object:
    if not isinstance(value, torch.Tensor):
        return value
    target_dtype = dtype if value.is_floating_point() else value.dtype
    return value.to(device=device, dtype=target_dtype, non_blocking=True)


def anyflow_batch_from_cached(
    batch: TrainingBatch,
    *,
    unconditional_conditioning: Mapping[str, object],
    device: torch.device,
    dtype: torch.dtype,
) -> AnyFlowTrainingBatch:
    """Move one cached latent/context batch into the AnyFlow contract."""

    values = dict(batch.conditions)
    clean_latents = values.pop("clean_latents", None)
    if not isinstance(clean_latents, torch.Tensor):
        raise TypeError("AnyFlow cached batches require clean_latents")
    loss_mask = values.pop("latent_loss_mask", None)
    values.pop("valid_latent_mask", None)
    if isinstance(loss_mask, torch.Tensor) and not bool(torch.all(loss_mask == 1)):
        raise ValueError("AnyFlow training currently requires unpadded cached latents")
    conditioning = {
        name: _to_device(value, device=device, dtype=dtype)
        for name, value in values.items()
    }
    negative = batch_shared_conditioning(
        unconditional_conditioning,
        conditioning,
        batch_size=batch.batch_size,
    )
    return AnyFlowTrainingBatch(
        sample_ids=batch.sample_ids,
        clean_latents=clean_latents.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        ),
        conditioning=conditioning,
        unconditional_conditioning=negative,
    )


class AnyFlowCachedDataLoader(Iterable[AnyFlowTrainingBatch]):
    """Convert a checkpointable cache loader without owning its sampling state."""

    def __init__(
        self,
        source: Iterable[TrainingBatch],
        *,
        unconditional_conditioning: Mapping[str, object],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.source = source
        self.unconditional_conditioning = MappingProxyType(
            dict(unconditional_conditioning)
        )
        self.device = torch.device(device)
        self.dtype = dtype

    def __iter__(self) -> Iterator[AnyFlowTrainingBatch]:
        for batch in self.source:
            yield anyflow_batch_from_cached(
                batch,
                unconditional_conditioning=self.unconditional_conditioning,
                device=self.device,
                dtype=self.dtype,
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": ANYFLOW_DATA_LOADER_STATE_SCHEMA,
            "source": dict(self.source.state_dict()),  # type: ignore[attr-defined]
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if state_dict.get("schema") != ANYFLOW_DATA_LOADER_STATE_SCHEMA:
            raise ValueError("unsupported AnyFlow data-loader state")
        source = state_dict.get("source")
        if not isinstance(source, Mapping):
            raise TypeError("AnyFlow source-loader state must be a mapping")
        self.source.load_state_dict(dict(source))  # type: ignore[attr-defined]


__all__ = [
    "ANYFLOW_DATA_LOADER_STATE_SCHEMA",
    "AnyFlowCachedDataLoader",
    "anyflow_batch_from_cached",
]
