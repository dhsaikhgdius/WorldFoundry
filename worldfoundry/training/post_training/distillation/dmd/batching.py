"""Stateful cached-data bridge for native DMD training."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType

import torch

from worldfoundry.training.api.contracts import PreparedBatch, TrainingBatch, TrainModelAdapter

from ...shared.batching import batch_shared_conditioning
from .contracts import DMDTrainingBatch

DMD_DATA_LOADER_STATE_SCHEMA = "worldfoundry-dmd-data-loader"


def dmd_batch_from_prepared(
    prepared: PreparedBatch,
    *,
    shared_unconditional_conditioning: Mapping[str, object],
) -> DMDTrainingBatch:
    """Convert one model-prepared batch into the explicit two-branch DMD contract."""

    if not isinstance(prepared, PreparedBatch):
        raise TypeError("prepared must be a PreparedBatch")
    if not isinstance(prepared.clean_latents, torch.Tensor):
        raise TypeError("native DMD currently requires one clean latent tensor")
    if prepared.loss_mask is not None and not isinstance(prepared.loss_mask, torch.Tensor):
        raise TypeError("native DMD currently requires one tensor loss_mask")
    unconditional = batch_shared_conditioning(
        shared_unconditional_conditioning,
        prepared.conditioning,
        batch_size=prepared.batch_size,
    )
    return DMDTrainingBatch(
        sample_ids=prepared.sample_ids,
        clean_latents=prepared.clean_latents,
        conditioning=prepared.conditioning,
        unconditional_conditioning=unconditional,
        loss_mask=prepared.loss_mask,
        sample_weights=prepared.sample_weights,
        metadata=prepared.metadata,
    )


class NativeDMDDataLoader(Iterable[DMDTrainingBatch]):
    """Prepare cached batches lazily while forwarding exact source-loader state."""

    def __init__(
        self,
        source: Iterable[TrainingBatch],
        adapter: TrainModelAdapter,
        *,
        shared_unconditional_conditioning: Mapping[str, object],
    ) -> None:
        if not isinstance(adapter, TrainModelAdapter):
            raise TypeError("adapter must implement TrainModelAdapter")
        if not callable(getattr(source, "state_dict", None)) or not callable(getattr(source, "load_state_dict", None)):
            raise TypeError("DMD source loader must expose state_dict/load_state_dict")
        if not isinstance(shared_unconditional_conditioning, Mapping):
            raise TypeError("shared_unconditional_conditioning must be a mapping")
        self.source = source
        self.adapter = adapter
        self.shared_unconditional_conditioning = MappingProxyType(
            {str(key): value for key, value in shared_unconditional_conditioning.items()}
        )

    def __iter__(self) -> Iterator[DMDTrainingBatch]:
        for batch in self.source:
            if not isinstance(batch, TrainingBatch):
                raise TypeError("DMD source loader must emit TrainingBatch values")
            yield dmd_batch_from_prepared(
                self.adapter.prepare_batch(batch),
                shared_unconditional_conditioning=self.shared_unconditional_conditioning,
            )

    def state_dict(self) -> dict[str, object]:
        source_state = self.source.state_dict()  # type: ignore[attr-defined]
        if not isinstance(source_state, Mapping):
            raise TypeError("DMD source loader state_dict must return a mapping")
        return {"schema": DMD_DATA_LOADER_STATE_SCHEMA, "source": dict(source_state)}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {"schema", "source"}:
            raise ValueError("DMD data-loader state fields differ from the active schema")
        if state_dict["schema"] != DMD_DATA_LOADER_STATE_SCHEMA:
            raise ValueError(f"unsupported DMD data-loader state: {state_dict['schema']!r}")
        source_state = state_dict["source"]
        if not isinstance(source_state, Mapping):
            raise TypeError("saved DMD source loader state must be a mapping")
        self.source.load_state_dict(dict(source_state))  # type: ignore[attr-defined]


__all__ = ["DMD_DATA_LOADER_STATE_SCHEMA", "NativeDMDDataLoader", "dmd_batch_from_prepared"]
