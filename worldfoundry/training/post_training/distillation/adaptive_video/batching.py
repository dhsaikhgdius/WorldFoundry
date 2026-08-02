"""Two-stream data bridge for adaptive video distillation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

import torch

from worldfoundry.training.api.contracts import (
    PreparedBatch,
    TrainingBatch,
    TrainModelAdapter,
)

from ..dmd.contracts import DMDTrainingBatch
from .contracts import AdaptiveVideoRealBatch, AdaptiveVideoTrainingBatch

ADAPTIVE_VIDEO_DATA_LOADER_STATE_SCHEMA = "worldfoundry-adaptive-video-data-loader"


def adaptive_video_real_batch_from_prepared(
    prepared: PreparedBatch,
) -> AdaptiveVideoRealBatch:
    if not isinstance(prepared, PreparedBatch):
        raise TypeError("prepared must be PreparedBatch")
    if not isinstance(prepared.clean_latents, torch.Tensor):
        raise TypeError("adaptive video real data requires one latent tensor")
    if prepared.loss_mask is not None and not isinstance(
        prepared.loss_mask,
        torch.Tensor,
    ):
        raise TypeError("adaptive video real loss_mask must be one tensor")
    return AdaptiveVideoRealBatch(
        sample_ids=prepared.sample_ids,
        latents=prepared.clean_latents,
        conditioning=prepared.conditioning,
        loss_mask=prepared.loss_mask,
        sample_weights=prepared.sample_weights,
    )


class NativeAdaptiveVideoDataLoader:
    """Own independent prompt and real-video cursors for exact resume."""

    def __init__(
        self,
        generated_source: Iterable[DMDTrainingBatch],
        real_source: Iterable[TrainingBatch],
        real_adapter: TrainModelAdapter,
    ) -> None:
        for name, source in (
            ("generated_source", generated_source),
            ("real_source", real_source),
        ):
            if not callable(getattr(source, "state_dict", None)) or not callable(
                getattr(source, "load_state_dict", None)
            ):
                raise TypeError(
                    f"adaptive video {name} must expose state_dict/load_state_dict"
                )
        if not isinstance(real_adapter, TrainModelAdapter):
            raise TypeError("real_adapter must implement TrainModelAdapter")
        self.generated_source = generated_source
        self.real_source = real_source
        self.real_adapter = real_adapter
        self._generated_iterator: Iterator[DMDTrainingBatch] | None = None
        self._real_iterator: Iterator[TrainingBatch] | None = None

    @staticmethod
    def _next_or_restart(
        source: Iterable[object],
        iterator: Iterator[object] | None,
        *,
        role: str,
    ) -> tuple[object, Iterator[object]]:
        active = iter(source) if iterator is None else iterator
        try:
            return next(active), active
        except StopIteration:
            active = iter(source)
            try:
                return next(active), active
            except StopIteration as error:
                raise RuntimeError(f"adaptive video {role} source is empty") from error

    def next_generated(self) -> DMDTrainingBatch:
        value, iterator = self._next_or_restart(
            self.generated_source,
            self._generated_iterator,
            role="generated prompt",
        )
        self._generated_iterator = iterator  # type: ignore[assignment]
        if not isinstance(value, DMDTrainingBatch):
            raise TypeError(
                "adaptive video generated source must emit DMDTrainingBatch"
            )
        return value

    def next_real(self) -> AdaptiveVideoRealBatch:
        value, iterator = self._next_or_restart(
            self.real_source,
            self._real_iterator,
            role="real video",
        )
        self._real_iterator = iterator  # type: ignore[assignment]
        if not isinstance(value, TrainingBatch):
            raise TypeError(
                "adaptive video real source must emit TrainingBatch"
            )
        return adaptive_video_real_batch_from_prepared(
            self.real_adapter.prepare_batch(value)
        )

    def next_generator_batch(self) -> AdaptiveVideoTrainingBatch:
        return AdaptiveVideoTrainingBatch.combine(
            self.next_generated(),
            self.next_real(),
        )

    def state_dict(self) -> dict[str, object]:
        generated_state = self.generated_source.state_dict()  # type: ignore[attr-defined]
        real_state = self.real_source.state_dict()  # type: ignore[attr-defined]
        if not isinstance(generated_state, Mapping) or not isinstance(
            real_state,
            Mapping,
        ):
            raise TypeError("adaptive video source states must be mappings")
        return {
            "schema": ADAPTIVE_VIDEO_DATA_LOADER_STATE_SCHEMA,
            "generated_source": dict(generated_state),
            "real_source": dict(real_state),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {
            "schema",
            "generated_source",
            "real_source",
        }:
            raise ValueError(
                "adaptive video data-loader state fields differ from the active schema"
            )
        if state_dict["schema"] != ADAPTIVE_VIDEO_DATA_LOADER_STATE_SCHEMA:
            raise ValueError("unsupported adaptive video data-loader state schema")
        generated_state = state_dict["generated_source"]
        real_state = state_dict["real_source"]
        if not isinstance(generated_state, Mapping) or not isinstance(
            real_state,
            Mapping,
        ):
            raise TypeError("saved adaptive video source states must be mappings")
        self.generated_source.load_state_dict(dict(generated_state))  # type: ignore[attr-defined]
        self.real_source.load_state_dict(dict(real_state))  # type: ignore[attr-defined]
        self._generated_iterator = None
        self._real_iterator = None


__all__ = [
    "ADAPTIVE_VIDEO_DATA_LOADER_STATE_SCHEMA",
    "NativeAdaptiveVideoDataLoader",
    "adaptive_video_real_batch_from_prepared",
]
