"""Model-neutral classifier-free guidance for DiffusionOPD roles."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from ...shared.contracts import FlowPredictionAdapter


class BranchClassifierFreeGuidance:
    """Combine the adapter's negative and positive branches at one CFG scale."""

    def __init__(
        self,
        prediction: FlowPredictionAdapter,
        *,
        guidance_scale: float,
    ) -> None:
        if not isinstance(prediction, FlowPredictionAdapter):
            raise TypeError("DiffusionOPD guidance requires FlowPredictionAdapter")
        scale = float(guidance_scale)
        if not isfinite(scale) or scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        self.prediction = prediction
        self.module = prediction.module
        self.checkpoint_identity = getattr(prediction, "checkpoint_identity", None)
        self.guidance_scale = scale

    def predict_velocity(
        self,
        noisy_latents: object,
        sigmas: object,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> object:
        if branch != "positive":
            raise ValueError("guided DiffusionOPD prediction exposes its combined positive branch")
        if self.guidance_scale == 1:
            return self.prediction.predict_velocity(
                noisy_latents,
                sigmas,
                sample_ids=sample_ids,
                conditioning=conditioning,
                training=training,
                branch="positive",
            )
        negative = self.prediction.predict_velocity(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch="negative",
        )
        if self.guidance_scale == 0:
            return negative
        positive = self.prediction.predict_velocity(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch="positive",
        )
        return negative + self.guidance_scale * (positive - negative)

    def predict_clean(
        self,
        noisy_latents: object,
        sigmas: object,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> object:
        from worldfoundry.training.objectives.flow_matching import (
            flow_clean_from_velocity,
        )

        velocity = self.predict_velocity(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )
        return flow_clean_from_velocity(noisy_latents, velocity, sigmas)


__all__ = ["BranchClassifierFreeGuidance"]
