"""Native rectified-flow model boundary for bidirectional rCM."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from ...shared.contracts import FlowPredictionAdapter
from ..consistency.math import batch_coefficients, trigflow_to_rf_time
from .contracts import RCMPrediction


class NativeRFRCMPredictionAdapter:
    """Apply the official unit-scale RF-to-TrigFlow preconditioning.

    Native Wan adapters predict rectified-flow velocity at
    ``s = sin(t) / (cos(t) + sin(t))``.  rCM instead operates on the unscaled
    TrigFlow state.  This boundary owns that conversion so the objective never
    relies on a model-repository trainer or an implicit parameterization.

    NVlabs/rCM's executable scaling asserts ``sigma_data == 1``.  WorldFoundry
    therefore fixes that invariant here instead of exposing a configuration
    field whose non-unit values cannot execute faithfully.
    """

    def __init__(self, prediction: FlowPredictionAdapter) -> None:
        if not isinstance(prediction, FlowPredictionAdapter):
            raise TypeError("prediction must implement FlowPredictionAdapter")
        if not isinstance(prediction.module, nn.Module):
            raise TypeError("prediction.module must be an nn.Module")
        self.prediction = prediction
        self.module = prediction.module

    def predict(
        self,
        noisy_latents: torch.Tensor,
        trig_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> RCMPrediction:
        if not isinstance(noisy_latents, torch.Tensor) or noisy_latents.ndim < 2:
            raise TypeError("rCM noisy_latents must be a [B,...] tensor")
        if not noisy_latents.is_floating_point():
            raise TypeError("rCM noisy_latents must be floating point")
        if not isinstance(trig_timesteps, torch.Tensor):
            trig_timesteps = torch.as_tensor(
                trig_timesteps,
                device=noisy_latents.device,
                dtype=torch.float32,
            )
        trig_timesteps = trig_timesteps.to(
            device=noisy_latents.device,
            dtype=torch.float32,
        )
        if trig_timesteps.ndim == 0:
            trig_timesteps = trig_timesteps.expand(noisy_latents.shape[0])
        elif trig_timesteps.numel() == noisy_latents.shape[0]:
            trig_timesteps = trig_timesteps.reshape(noisy_latents.shape[0])
        else:
            raise ValueError("rCM requires one TrigFlow timestep per sample")
        if len(sample_ids) != noisy_latents.shape[0]:
            raise ValueError("sample_ids must match the rCM batch dimension")
        rf_timesteps = trigflow_to_rf_time(trig_timesteps)
        cosine = batch_coefficients(torch.cos(trig_timesteps), noisy_latents)
        sine = batch_coefficients(torch.sin(trig_timesteps), noisy_latents)
        scale = cosine + sine
        scaled_latents = noisy_latents / scale
        rf_velocity = self.prediction.predict_velocity(
            scaled_latents,
            rf_timesteps,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )
        if not isinstance(rf_velocity, torch.Tensor) or rf_velocity.shape != noisy_latents.shape:
            raise ValueError("native RF velocity must match rCM noisy_latents")
        rf_time = batch_coefficients(rf_timesteps, noisy_latents)
        clean = scaled_latents - rf_time * rf_velocity
        # Algebraically equal to (cos(t) * x_t - x0) / sin(t), including the
        # finite t=0 limit where the quotient form is undefined.
        trig_velocity = ((cosine - sine) * noisy_latents + rf_velocity) / scale
        return RCMPrediction(clean_latents=clean, velocity=trig_velocity)


__all__ = ["NativeRFRCMPredictionAdapter"]
