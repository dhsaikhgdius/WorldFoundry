"""Shared AnyFlow DMD and fresh fake-score regression mathematics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import AnyFlowMapConfig
from .contracts import AnyFlowScoreAdapter, AnyFlowTrainingBatch
from .math import (
    anyflow_distribution_gradient,
    anyflow_dmd_proxy_loss,
    anyflow_real_guidance,
    flowmap_interpolate,
    flowmap_step,
    sample_logit_normal_time,
    shift_flowmap_time,
)


@dataclass(frozen=True, slots=True)
class AnyFlowDMDResult:
    loss: Tensor
    raw_time_mean: Tensor
    normalizer_mean: Tensor


@dataclass(frozen=True, slots=True)
class AnyFlowFakeScoreResult:
    loss: Tensor
    raw_time_mean: Tensor


def _score_prediction(
    adapter: AnyFlowScoreAdapter,
    noisy: Tensor,
    model_timesteps: Tensor,
    batch: AnyFlowTrainingBatch,
    *,
    training: bool,
    branch: str,
) -> Tensor:
    conditioning = batch.conditioning if branch == "positive" else batch.unconditional_conditioning
    value = adapter.predict_velocity(
        noisy,
        model_timesteps,
        sample_ids=batch.sample_ids,
        conditioning=conditioning,
        training=training,
        branch=branch,
    )
    if not isinstance(value, Tensor) or value.shape != noisy.shape:
        raise ValueError("AnyFlow score velocity must preserve the latent shape")
    return value


def _score_times(
    normalized: Tensor,
    reference: Tensor,
    flow_map: AnyFlowMapConfig,
    *,
    minimum: float,
    maximum: float,
) -> tuple[Tensor, Tensor]:
    model_scalar = (
        shift_flowmap_time(normalized, flow_map.timestep_shift) * float(flow_map.num_train_timesteps)
    ).clamp(min=float(minimum), max=float(maximum))
    shifted = model_scalar / float(flow_map.num_train_timesteps)
    model = model_scalar[:, None].expand(
        int(reference.shape[0]),
        int(reference.shape[2]),
    )
    return shifted, model


def anyflow_dmd_loss(
    generated: Tensor,
    batch: AnyFlowTrainingBatch,
    real_score: AnyFlowScoreAdapter,
    fake_score: AnyFlowScoreAdapter,
    flow_map: AnyFlowMapConfig,
    *,
    dmd_weight: float,
    real_guidance_scale: float,
    minimum_timestep: float,
    maximum_timestep: float,
    generator: torch.Generator | None,
) -> AnyFlowDMDResult:
    """Sample uniform DMD time and inject the released normalized gradient."""

    if generated.shape != batch.clean_latents.shape:
        raise ValueError("AnyFlow generated latents and DMD batch must align")
    with torch.no_grad():
        raw_time = torch.rand(
            (batch.batch_size,),
            device=generated.device,
            dtype=torch.float32,
            generator=generator,
        )
        shifted, model_time = _score_times(
            raw_time,
            generated,
            flow_map,
            minimum=minimum_timestep,
            maximum=maximum_timestep,
        )
        noise = torch.randn(
            generated.shape,
            device=generated.device,
            dtype=generated.dtype,
            generator=generator,
        )
        noisy = flowmap_interpolate(generated, noise, shifted).detach()
        fake_velocity = _score_prediction(
            fake_score,
            noisy,
            model_time,
            batch,
            training=False,
            branch="positive",
        )
        fake_clean = flowmap_step(
            noisy,
            fake_velocity,
            shifted,
            torch.zeros_like(shifted),
        )
        real_conditional_velocity = _score_prediction(
            real_score,
            noisy,
            model_time,
            batch,
            training=False,
            branch="positive",
        )
        real_conditional = flowmap_step(
            noisy,
            real_conditional_velocity,
            shifted,
            torch.zeros_like(shifted),
        )
        real_unconditional_velocity = _score_prediction(
            real_score,
            noisy,
            model_time,
            batch,
            training=False,
            branch="negative",
        )
        real_unconditional = flowmap_step(
            noisy,
            real_unconditional_velocity,
            shifted,
            torch.zeros_like(shifted),
        )
        guided_real = anyflow_real_guidance(
            real_conditional,
            real_unconditional,
            real_guidance_scale,
        )
        distribution_gradient, normalizer = anyflow_distribution_gradient(
            generated,
            fake_clean,
            guided_real,
        )
    loss = anyflow_dmd_proxy_loss(generated, distribution_gradient) * float(dmd_weight)
    return AnyFlowDMDResult(
        loss=loss,
        raw_time_mean=raw_time.detach().mean(),
        normalizer_mean=normalizer.detach().float().mean(),
    )


def anyflow_fake_score_loss(
    generated: Tensor,
    batch: AnyFlowTrainingBatch,
    fake_score: AnyFlowScoreAdapter,
    flow_map: AnyFlowMapConfig,
    *,
    logit_mean: float,
    logit_std: float,
    minimum_timestep: float,
    maximum_timestep: float,
    generator: torch.Generator | None,
) -> AnyFlowFakeScoreResult:
    """Fit the fake score to a fresh generated distribution sample."""

    if generated.shape != batch.clean_latents.shape:
        raise ValueError("AnyFlow generated latents and fake-score batch must align")
    raw_time = sample_logit_normal_time(
        batch.batch_size,
        device=generated.device,
        mean=logit_mean,
        std=logit_std,
        generator=generator,
    )
    shifted, model_time = _score_times(
        raw_time,
        generated,
        flow_map,
        minimum=minimum_timestep,
        maximum=maximum_timestep,
    )
    noise = torch.randn(
        generated.shape,
        device=generated.device,
        dtype=generated.dtype,
        generator=generator,
    )
    noisy = flowmap_interpolate(generated, noise, shifted)
    predicted = _score_prediction(
        fake_score,
        noisy,
        model_time,
        batch,
        training=True,
        branch="positive",
    )
    loss = (predicted.float() - (noise - generated).float()).square().flatten(1).mean()
    return AnyFlowFakeScoreResult(
        loss=loss,
        raw_time_mean=raw_time.detach().mean(),
    )


__all__ = [
    "AnyFlowDMDResult",
    "AnyFlowFakeScoreResult",
    "anyflow_dmd_loss",
    "anyflow_fake_score_loss",
]
