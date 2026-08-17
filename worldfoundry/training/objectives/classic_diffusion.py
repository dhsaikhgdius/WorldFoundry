"""Discrete DDPM corruption and regression for native latent-diffusion models.

The existing flow-matching objective is intentionally not stretched to cover
classic alpha-cumprod diffusion.  This module supplies the small amount of
model-independent math shared by LVDM and DynamiCrafter while their adapters
continue to own latent encoding and conditioning.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite, prod
from typing import Literal

import torch
from torch import Tensor

from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainStepResult
from worldfoundry.training.objectives.flow_matching import flow_matching_denominator

ClassicPredictionType = Literal["epsilon", "v_prediction", "sample"]
ClassicLossType = Literal["l1", "l2"]
ConditioningBuilder = Callable[[PreparedBatch, Tensor, Tensor, torch.Generator | None], Mapping[str, object]]


def lvdm_linear_beta_schedule(
    num_train_timesteps: int,
    *,
    beta_start: float,
    beta_end: float,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return LVDM's squared linear interpolation between beta square roots."""

    return torch.linspace(
        float(beta_start) ** 0.5,
        float(beta_end) ** 0.5,
        int(num_train_timesteps),
        dtype=torch.float64,
        device=device,
    ).square()


def rescale_betas_to_zero_terminal_snr(betas: Tensor) -> Tensor:
    """Apply the zero-terminal-SNR transform used by DynamiCrafter."""

    alpha_bar_sqrt = torch.cumprod(1.0 - betas, dim=0).sqrt()
    first = alpha_bar_sqrt[0].clone()
    last = alpha_bar_sqrt[-1].clone()
    shifted = (alpha_bar_sqrt - last) * first / (first - last)
    alpha_bar = shifted.square()
    alphas = torch.cat((alpha_bar[:1], alpha_bar[1:] / alpha_bar[:-1]))
    return 1.0 - alphas


def extract_schedule(values: Tensor, timesteps: Tensor, reference: Tensor) -> Tensor:
    """Gather one scalar schedule value per sample and append visual axes."""

    gathered = values.to(device=timesteps.device).gather(0, timesteps)
    return gathered.reshape((int(reference.shape[0]),) + (1,) * (reference.ndim - 1))


def add_ddpm_noise(clean: Tensor, noise: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Return ``sqrt(alpha_bar) * clean + sqrt(1-alpha_bar) * noise``."""

    return alpha.to(dtype=clean.dtype) * clean + sigma.to(dtype=clean.dtype) * noise


def velocity_target(clean: Tensor, noise: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Return the v-prediction target ``alpha * noise - sigma * clean``."""

    return alpha.to(dtype=clean.dtype) * noise - sigma.to(dtype=clean.dtype) * clean


def dynamic_latent_scale(
    timesteps: Tensor,
    reference: Tensor,
    *,
    final_scale: float,
    transition_steps: int,
) -> Tensor:
    """Return DynamiCrafter's 1-to-base-scale prefix followed by a plateau."""

    progress = timesteps.to(torch.float32) / float(max(transition_steps - 1, 1))
    progress = progress.clamp(0.0, 1.0)
    scales = 1.0 + progress * (float(final_scale) - 1.0)
    return scales.reshape((int(reference.shape[0]),) + (1,) * (reference.ndim - 1))


@dataclass(frozen=True, slots=True)
class ClassicDiffusionConfig:
    """Behavior-bearing discrete diffusion settings."""

    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: ClassicPredictionType = "epsilon"
    loss_type: ClassicLossType = "l2"
    zero_terminal_snr: bool = False
    dynamic_rescale_final: float | None = None
    dynamic_rescale_transition_steps: int = 400

    def __post_init__(self) -> None:
        if isinstance(self.num_train_timesteps, bool) or int(self.num_train_timesteps) < 2:
            raise ValueError("num_train_timesteps must be at least two")
        start = float(self.beta_start)
        end = float(self.beta_end)
        if not (isfinite(start) and isfinite(end) and 0.0 < start < end < 1.0):
            raise ValueError("beta range must satisfy 0 < beta_start < beta_end < 1")
        if self.prediction_type not in {"epsilon", "v_prediction", "sample"}:
            raise ValueError("prediction_type must be epsilon, v_prediction, or sample")
        if self.loss_type not in {"l1", "l2"}:
            raise ValueError("loss_type must be l1 or l2")
        if not isinstance(self.zero_terminal_snr, bool):
            raise TypeError("zero_terminal_snr must be bool")
        final = self.dynamic_rescale_final
        if final is not None:
            final = float(final)
            if not isfinite(final) or final <= 0.0:
                raise ValueError("dynamic_rescale_final must be finite and positive")
            if isinstance(self.dynamic_rescale_transition_steps, bool) or int(
                self.dynamic_rescale_transition_steps
            ) <= 0:
                raise ValueError("dynamic_rescale_transition_steps must be positive")
        object.__setattr__(self, "num_train_timesteps", int(self.num_train_timesteps))
        object.__setattr__(self, "beta_start", start)
        object.__setattr__(self, "beta_end", end)
        object.__setattr__(self, "dynamic_rescale_final", final)
        object.__setattr__(
            self,
            "dynamic_rescale_transition_steps",
            int(self.dynamic_rescale_transition_steps),
        )


@dataclass(frozen=True, slots=True)
class DiffusionRegressionLoss:
    loss: Tensor
    numerator: Tensor
    denominator: Tensor


def _expanded_weight(
    target: Tensor,
    *,
    loss_mask: object | None,
    sample_weights: object | None,
) -> Tensor:
    weights = torch.ones_like(target, dtype=torch.float32)
    if loss_mask is not None:
        if not isinstance(loss_mask, Tensor):
            raise TypeError("loss_mask must be a tensor")
        mask = loss_mask
        if mask.ndim + 1 == target.ndim and int(mask.shape[0]) == int(target.shape[0]):
            mask = mask.unsqueeze(1)
        weights = weights * torch.broadcast_to(mask, target.shape).to(device=target.device, dtype=torch.float32)
    if sample_weights is not None:
        if not isinstance(sample_weights, Tensor):
            raise TypeError("sample_weights must be a tensor")
        view = (int(target.shape[0]),) + (1,) * (target.ndim - 1)
        weights = weights * sample_weights.to(device=target.device, dtype=torch.float32).reshape(view)
    return weights


def diffusion_regression_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    loss_type: ClassicLossType,
    loss_mask: object | None = None,
    sample_weights: object | None = None,
) -> DiffusionRegressionLoss:
    """Reduce the official elementwise L1/L2 objective in FP32."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    difference = prediction.float() - target.float()
    elementwise = difference.abs() if loss_type == "l1" else difference.square()
    weights = _expanded_weight(target, loss_mask=loss_mask, sample_weights=sample_weights)
    denominator = weights.sum()
    if not bool(denominator.detach() > 0):
        raise ValueError("diffusion regression has no positive-weight elements")
    numerator = (elementwise * weights).sum()
    return DiffusionRegressionLoss(
        loss=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
    )


class ClassicDiffusionObjective:
    """Uniform-timestep DDPM objective reusable across latent video families."""

    def __init__(
        self,
        config: ClassicDiffusionConfig,
        *,
        conditioning_builder: ConditioningBuilder | None = None,
    ) -> None:
        if not isinstance(config, ClassicDiffusionConfig):
            raise TypeError("config must be ClassicDiffusionConfig")
        self.config = config
        self.prediction_type = config.prediction_type
        self.conditioning_builder = conditioning_builder
        betas = lvdm_linear_beta_schedule(
            config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )
        if config.zero_terminal_snr:
            betas = rescale_betas_to_zero_terminal_snr(betas)
        alpha_cumprods = torch.cumprod(1.0 - betas, dim=0).to(torch.float32)
        self.alpha_cumprods = alpha_cumprods
        self.alphas = alpha_cumprods.sqrt()
        self.sigmas = (1.0 - alpha_cumprods).clamp_min(0.0).sqrt()
        # Per-device copies of the schedule tables so corrupt() does not pay a
        # host-to-device transfer on every training step.
        self._device_schedules: dict[torch.device, tuple[Tensor, Tensor]] = {
            self.alphas.device: (self.alphas, self.sigmas)
        }

    def _schedules_for_device(self, device: torch.device) -> tuple[Tensor, Tensor]:
        cached = self._device_schedules.get(device)
        if cached is None:
            cached = (self.alphas.to(device=device), self.sigmas.to(device=device))
            self._device_schedules[device] = cached
        return cached

    def corrupt(self, batch: PreparedBatch, *, generator: object | None = None) -> ObjectiveBatch:
        clean = batch.clean_latents
        if isinstance(clean, Mapping) or not isinstance(clean, Tensor):
            raise TypeError("classic video diffusion requires one clean latent tensor")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None")
        timesteps = torch.randint(
            self.config.num_train_timesteps,
            (batch.batch_size,),
            device=clean.device,
            generator=generator,
        )
        objective_clean = clean
        if self.config.dynamic_rescale_final is not None:
            objective_clean = clean * dynamic_latent_scale(
                timesteps,
                clean,
                final_scale=self.config.dynamic_rescale_final,
                transition_steps=self.config.dynamic_rescale_transition_steps,
            ).to(dtype=clean.dtype)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        device_alphas, device_sigmas = self._schedules_for_device(clean.device)
        alpha = extract_schedule(device_alphas, timesteps, clean)
        sigma = extract_schedule(device_sigmas, timesteps, clean)
        noisy = add_ddpm_noise(objective_clean, noise, alpha, sigma)
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = velocity_target(objective_clean, noise, alpha, sigma)
        else:
            target = objective_clean

        conditioning = (
            dict(batch.conditioning)
            if self.conditioning_builder is None
            # Conditioning frames come from the original encoded video.  In
            # DynamiCrafter the dynamic scale is applied only to the diffusion
            # x_start, after hybrid conditioning has already been constructed.
            else dict(self.conditioning_builder(batch, timesteps, clean, generator))
        )
        metadata = dict(batch.metadata)
        metadata.update(
            {
                "objective": "classic_diffusion",
                "prediction_type": self.prediction_type,
                "loss_type": self.config.loss_type,
            }
        )
        return ObjectiveBatch(
            sample_ids=batch.sample_ids,
            model_input=noisy,
            target=target,
            sigmas=device_sigmas.gather(0, timesteps),
            timesteps=timesteps,
            conditioning=conditioning,
            noise=noise,
            loss_mask=batch.loss_mask,
            sample_weights=batch.sample_weights,
            metadata=metadata,
        )

    def compute_loss(self, prediction: object, batch: ObjectiveBatch) -> TrainStepResult:
        if isinstance(prediction, Mapping) or not isinstance(prediction, Tensor):
            raise TypeError("classic video diffusion prediction must be one tensor")
        target = batch.target
        if isinstance(target, Mapping) or not isinstance(target, Tensor):
            raise TypeError("classic video diffusion target must be one tensor")
        reduced = diffusion_regression_loss(
            prediction,
            target,
            loss_type=self.config.loss_type,
            loss_mask=batch.loss_mask,
            sample_weights=batch.sample_weights,
        )
        latent_tokens = int(target.shape[0]) * prod(int(size) for size in target.shape[2:])
        name = f"classic_diffusion/{self.config.loss_type}"
        return TrainStepResult(
            loss=reduced.loss,
            losses={"classic_diffusion": reduced.loss, name: reduced.loss},
            metrics={
                "loss_numerator": reduced.numerator.detach(),
                "loss_denominator": reduced.denominator.detach(),
                "sigma_mean": batch.sigmas.float().mean().detach(),
                "sigma_min": batch.sigmas.float().min().detach(),
                "sigma_max": batch.sigmas.float().max().detach(),
            },
            sample_count=batch.batch_size,
            latent_token_count=latent_tokens,
            diagnostics={
                "prediction_type": self.prediction_type,
                "loss_type": self.config.loss_type,
                "zero_terminal_snr": self.config.zero_terminal_snr,
                "dynamic_rescale_final": self.config.dynamic_rescale_final,
            },
        )

    def prepared_loss_denominator(self, batch: PreparedBatch) -> Tensor:
        clean = batch.clean_latents
        if isinstance(clean, Mapping) or not isinstance(clean, Tensor):
            raise TypeError("classic video diffusion requires one clean latent tensor")
        return flow_matching_denominator(
            clean,
            loss_mask=batch.loss_mask,
            sample_weights=batch.sample_weights,
        )


def lvdm_short_objective() -> ClassicDiffusionObjective:
    """Build the released four-frame unconditional LVDM objective."""

    return ClassicDiffusionObjective(
        ClassicDiffusionConfig(
            num_train_timesteps=1000,
            beta_start=0.0015,
            beta_end=0.0155,
            prediction_type="epsilon",
            loss_type="l1",
        )
    )


__all__ = [
    "ClassicDiffusionConfig",
    "ClassicDiffusionObjective",
    "DiffusionRegressionLoss",
    "add_ddpm_noise",
    "diffusion_regression_loss",
    "dynamic_latent_scale",
    "extract_schedule",
    "lvdm_linear_beta_schedule",
    "lvdm_short_objective",
    "rescale_betas_to_zero_terminal_snr",
    "velocity_target",
]
