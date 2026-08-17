"""Stateful cached-video bridge for Cosmos Predict2.5 DMD2."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence

import torch

from worldfoundry.training.api.contracts import PreparedBatch, TrainingBatch
from worldfoundry.training.models.cosmos import CosmosPredict25TrainAdapter
from worldfoundry.training.post_training.distillation.dmd2.contracts import (
    DMD2TrainingBatch,
)

COSMOS_PREDICT25_DMD2_CONDITIONAL_FRAME_PROBABILITIES = (0.6, 0.2, 0.2)
COSMOS_PREDICT25_DMD2_DATA_LOADER_SCHEMA = "worldfoundry-cosmos-predict25-dmd2-data-loader"


def _probabilities(values: Sequence[float]) -> tuple[float, ...]:
    probabilities = tuple(float(value) for value in values)
    total = sum(probabilities)
    if not probabilities or any(value < 0 for value in probabilities) or total <= 0:
        raise ValueError("conditional-frame probabilities must be non-negative with positive mass")
    return tuple(value / total for value in probabilities)


def cosmos_predict25_dmd2_batch(
    prepared: PreparedBatch,
    *,
    conditional_frame_probabilities: Sequence[float],
    seed: int,
) -> DMD2TrainingBatch:
    """Add the released bidirectional conditioning draw to one cached batch."""

    if not isinstance(prepared.clean_latents, torch.Tensor) or prepared.clean_latents.ndim != 5:
        raise TypeError("Cosmos Predict2.5 DMD2 requires BCTHW clean latents")
    clean = prepared.clean_latents
    negative_context = prepared.conditioning.get("negative_context")
    context = prepared.conditioning.get("context")
    if not isinstance(context, torch.Tensor) or not isinstance(negative_context, torch.Tensor):
        raise TypeError("Cosmos Predict2.5 DMD2 requires positive and negative cached context")
    weights = torch.tensor(
        _probabilities(conditional_frame_probabilities),
        device="cpu",
        dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    counts = torch.multinomial(
        weights,
        prepared.batch_size,
        replacement=True,
        generator=generator,
    ).clamp_max(int(clean.shape[2]))
    frame_ids = torch.arange(int(clean.shape[2]), device=clean.device).reshape(1, -1)
    indicator = (frame_ids < counts.to(clean.device)[:, None]).to(clean.dtype)[:, None, :, None, None]
    mask = indicator.expand(prepared.batch_size, 1, *tuple(int(value) for value in clean.shape[-3:]))
    positive = dict(prepared.conditioning)
    positive.update(
        {
            "condition_latents": clean,
            "condition_indicator": indicator,
            "condition_mask": mask,
        }
    )
    unconditional = dict(positive)
    unconditional["context"] = negative_context
    return DMD2TrainingBatch(
        sample_ids=prepared.sample_ids,
        real_sample_ids=prepared.sample_ids,
        real_latents=clean,
        conditioning=positive,
        unconditional_conditioning=unconditional,
        real_conditioning=positive,
        sample_weights=prepared.sample_weights,
    )


class CosmosPredict25DMD2DataLoader(Iterable[DMD2TrainingBatch]):
    """Prepare DMD2 batches while forwarding the source loader's exact cursor."""

    def __init__(
        self,
        source: Iterable[TrainingBatch],
        adapter: CosmosPredict25TrainAdapter,
        *,
        conditional_frame_probabilities: Sequence[float] = (COSMOS_PREDICT25_DMD2_CONDITIONAL_FRAME_PROBABILITIES),
        seed: int = 42,
    ) -> None:
        if not callable(getattr(source, "state_dict", None)) or not callable(getattr(source, "load_state_dict", None)):
            raise TypeError("Cosmos DMD2 source loader must expose state_dict/load_state_dict")
        if not isinstance(adapter, CosmosPredict25TrainAdapter):
            raise TypeError("Cosmos DMD2 data preparation requires CosmosPredict25TrainAdapter")
        self.source = source
        self.adapter = adapter
        self.conditional_frame_probabilities = _probabilities(conditional_frame_probabilities)
        self.seed = int(seed)
        self.batch_index = 0

    def __iter__(self) -> Iterator[DMD2TrainingBatch]:
        for source_batch in self.source:
            if not isinstance(source_batch, TrainingBatch):
                raise TypeError("Cosmos DMD2 source loader must emit TrainingBatch values")
            batch = cosmos_predict25_dmd2_batch(
                self.adapter.prepare_batch(source_batch),
                conditional_frame_probabilities=self.conditional_frame_probabilities,
                seed=self.seed + self.batch_index,
            )
            self.batch_index += 1
            yield batch

    def state_dict(self) -> dict[str, object]:
        source_state = self.source.state_dict()  # type: ignore[attr-defined]
        if not isinstance(source_state, Mapping):
            raise TypeError("Cosmos DMD2 source loader state must be a mapping")
        return {
            "schema": COSMOS_PREDICT25_DMD2_DATA_LOADER_SCHEMA,
            "source": dict(source_state),
            "seed": self.seed,
            "batch_index": self.batch_index,
            "conditional_frame_probabilities": list(self.conditional_frame_probabilities),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        expected = {
            "schema",
            "source",
            "seed",
            "batch_index",
            "conditional_frame_probabilities",
        }
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected:
            raise ValueError("Cosmos DMD2 data-loader state fields differ")
        if state_dict["schema"] != COSMOS_PREDICT25_DMD2_DATA_LOADER_SCHEMA:
            raise ValueError(f"unsupported Cosmos DMD2 data-loader state: {state_dict['schema']!r}")
        if (
            int(state_dict["seed"]) != self.seed
            or tuple(
                float(value)
                for value in state_dict["conditional_frame_probabilities"]  # type: ignore[union-attr]
            )
            != self.conditional_frame_probabilities
        ):
            raise ValueError("saved Cosmos DMD2 conditioning sampler differs")
        source_state = state_dict["source"]
        if not isinstance(source_state, Mapping):
            raise TypeError("saved Cosmos DMD2 source state must be a mapping")
        batch_index = int(state_dict["batch_index"])
        if batch_index < 0:
            raise ValueError("saved Cosmos DMD2 batch index must be non-negative")
        self.source.load_state_dict(dict(source_state))  # type: ignore[attr-defined]
        self.batch_index = batch_index


__all__ = [
    "COSMOS_PREDICT25_DMD2_CONDITIONAL_FRAME_PROBABILITIES",
    "CosmosPredict25DMD2DataLoader",
    "cosmos_predict25_dmd2_batch",
]
