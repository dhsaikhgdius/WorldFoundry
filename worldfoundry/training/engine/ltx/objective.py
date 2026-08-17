"""LTX flow corruption with the author trainer's token-aware sigma sampler."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, prod

import torch

from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainStepResult
from worldfoundry.training.objectives.flow_matching import (
    FlowMatchingObjective,
    flow_interpolate,
    flow_velocity_target,
)


@dataclass(frozen=True, slots=True)
class LTXTimestepSamplingConfig:
    """Parameters released by Lightricks for LTX video fine-tuning."""

    mode: str = "shifted-logit-normal"
    standard_deviation: float = 1.0
    stretch: bool = True
    epsilon: float = 1.0e-3
    uniform_probability: float = 0.1
    minimum_tokens: int = 1024
    maximum_tokens: int = 4096
    minimum_shift: float = 0.95
    maximum_shift: float = 2.05

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower().replace("_", "-")
        if mode not in {"uniform", "shifted-logit-normal"}:
            raise ValueError(f"unsupported LTX timestep sampler: {mode!r}")
        deviation = float(self.standard_deviation)
        epsilon = float(self.epsilon)
        probability = float(self.uniform_probability)
        minimum_tokens = int(self.minimum_tokens)
        maximum_tokens = int(self.maximum_tokens)
        minimum_shift = float(self.minimum_shift)
        maximum_shift = float(self.maximum_shift)
        if not isfinite(deviation) or deviation <= 0.0:
            raise ValueError("LTX timestep standard_deviation must be positive")
        if not 0.0 < epsilon < 1.0:
            raise ValueError("LTX timestep epsilon must be in (0, 1)")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("LTX timestep uniform_probability must be in [0, 1]")
        if minimum_tokens <= 0 or maximum_tokens <= minimum_tokens:
            raise ValueError("LTX timestep token anchors must be increasing and positive")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "standard_deviation", deviation)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "uniform_probability", probability)
        object.__setattr__(self, "minimum_tokens", minimum_tokens)
        object.__setattr__(self, "maximum_tokens", maximum_tokens)
        object.__setattr__(self, "minimum_shift", minimum_shift)
        object.__setattr__(self, "maximum_shift", maximum_shift)

    @property
    def timestep_sampler(self) -> str:
        return self.mode

    @property
    def flow_shift(self) -> float:
        return 1.0

    @property
    def num_train_timesteps(self) -> None:
        return None

    def shift_for_sequence_length(self, sequence_length: int) -> float:
        slope = (self.maximum_shift - self.minimum_shift) / (self.maximum_tokens - self.minimum_tokens)
        intercept = self.minimum_shift - slope * self.minimum_tokens
        return slope * int(sequence_length) + intercept


def sample_ltx_sigmas(
    batch_size: int,
    sequence_length: int,
    *,
    config: LTXTimestepSamplingConfig,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one corruption level per video using the released LTX formulas."""

    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("LTX sigma sampling requires positive batch and sequence lengths")
    if config.mode == "uniform":
        return torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)

    shift = config.shift_for_sequence_length(sequence_length)
    normal = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
    logit_normal = torch.sigmoid(normal * config.standard_deviation + shift)
    if not config.stretch:
        return logit_normal

    upper = torch.sigmoid(
        torch.tensor(
            shift + 3.0902 * config.standard_deviation,
            device=device,
            dtype=torch.float32,
        )
    )
    lower = torch.sigmoid(
        torch.tensor(
            shift - 2.5758 * config.standard_deviation,
            device=device,
            dtype=torch.float32,
        )
    )
    raw = (logit_normal - lower) / (upper - lower)
    stretched = torch.where(raw >= config.epsilon, raw, 2.0 * config.epsilon - raw).clamp(0.0, 1.0)
    uniform = (1.0 - config.epsilon) * torch.rand(
        batch_size, device=device, dtype=torch.float32, generator=generator
    ) + config.epsilon
    choose = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
    return torch.where(choose > config.uniform_probability, stretched, uniform)


class LTXFlowMatchingObjective(FlowMatchingObjective):
    """Velocity flow matching with LTX's sequence-length-dependent sigmas."""

    def __init__(self, config: LTXTimestepSamplingConfig | None = None) -> None:
        self.config = config or LTXTimestepSamplingConfig()

    @staticmethod
    def _active_sample_weights(
        reference: torch.Tensor,
        loss_mask: object | None,
        sample_weights: object | None,
    ) -> torch.Tensor:
        batch_size = int(reference.shape[0])
        if sample_weights is None:
            weights = torch.ones(batch_size, device=reference.device, dtype=torch.float32)
        else:
            if not isinstance(sample_weights, torch.Tensor):
                raise TypeError("LTX sample weights must be a tensor")
            weights = sample_weights.to(device=reference.device, dtype=torch.float32)
        if loss_mask is not None:
            if not isinstance(loss_mask, torch.Tensor):
                raise TypeError("LTX loss mask must be a tensor")
            active = (loss_mask.to(device=reference.device) > 0).reshape(batch_size, -1).any(dim=1)
            weights = weights * active
        return weights

    def corrupt(
        self,
        batch: PreparedBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ObjectiveBatch:
        clean = batch.clean_latents
        if not isinstance(clean, torch.Tensor) or clean.ndim != 5:
            raise TypeError("LTX flow matching expects one BCTHW clean latent tensor")
        sequence_length = prod(int(size) for size in clean.shape[2:])
        sigmas = sample_ltx_sigmas(
            batch.batch_size,
            sequence_length,
            config=self.config,
            device=clean.device,
            generator=generator,
        )
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        model_input = flow_interpolate(clean, noise, sigmas.to(dtype=clean.dtype))
        target = flow_velocity_target(clean, noise)
        return ObjectiveBatch(
            sample_ids=batch.sample_ids,
            model_input=model_input,
            target=target,
            sigmas=sigmas,
            timesteps=sigmas,
            conditioning=batch.conditioning,
            noise=noise,
            loss_mask=batch.loss_mask,
            sample_weights=batch.sample_weights,
            metadata={"prediction_type": self.prediction_type},
        )

    @staticmethod
    def _sample_reduction(
        prediction: torch.Tensor,
        target: torch.Tensor,
        loss_mask: object | None,
        sample_weights: object | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        squared = (prediction.float() - target.float()).square()
        batch_size = int(squared.shape[0])
        if loss_mask is None:
            per_sample = squared.reshape(batch_size, -1).mean(dim=1)
        else:
            if not isinstance(loss_mask, torch.Tensor):
                raise TypeError("LTX loss mask must be a tensor")
            if loss_mask.ndim + 1 == squared.ndim:
                loss_mask = loss_mask.unsqueeze(1)
            mask = torch.broadcast_to(loss_mask, squared.shape).to(device=squared.device, dtype=torch.float32)
            flat_mask = mask.reshape(batch_size, -1)
            per_denominator = flat_mask.sum(dim=1)
            per_sample = (squared * mask).reshape(batch_size, -1).sum(dim=1) / per_denominator.clamp_min(1.0)
        weights = LTXFlowMatchingObjective._active_sample_weights(
            squared,
            loss_mask,
            sample_weights,
        )
        numerator = (per_sample * weights).sum()
        denominator = weights.sum()
        if not bool(denominator.detach() > 0):
            raise ValueError("LTX loss has no active samples")
        return numerator / denominator, numerator, denominator

    def compute_loss(self, prediction: object, batch: ObjectiveBatch) -> TrainStepResult:
        """Match the author trainer's per-sample masked MSE before batch mean."""

        if not isinstance(prediction, torch.Tensor) or not isinstance(batch.target, torch.Tensor):
            raise TypeError("LTX prediction and target must be tensors")
        loss, numerator, denominator = self._sample_reduction(
            prediction,
            batch.target,
            batch.loss_mask,
            batch.sample_weights,
        )
        return TrainStepResult(
            loss=loss,
            losses={"flow_matching": loss, "flow_matching/main": loss},
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
                "sigma_mean": batch.sigmas.float().mean().detach(),
                "sigma_min": batch.sigmas.float().min().detach(),
                "sigma_max": batch.sigmas.float().max().detach(),
            },
            sample_count=batch.batch_size,
            latent_token_count=int(batch.target.shape[0]) * prod(int(size) for size in batch.target.shape[2:]),
            diagnostics={
                "prediction_type": self.prediction_type,
                "timestep_sampler": self.config.timestep_sampler,
                "flow_shift": self.config.flow_shift,
                "num_train_timesteps": self.config.num_train_timesteps,
                "reduction": "per-sample-masked-mean",
            },
        )

    def loss_denominator(self, batch: ObjectiveBatch) -> torch.Tensor:
        if not isinstance(batch.target, torch.Tensor):
            raise TypeError("LTX target must be a tensor")
        denominator = self._active_sample_weights(
            batch.target,
            batch.loss_mask,
            batch.sample_weights,
        ).sum()
        if not bool(denominator.detach() > 0):
            raise ValueError("LTX loss has no active samples")
        return denominator

    def prepared_loss_denominator(self, batch: PreparedBatch) -> torch.Tensor:
        if not isinstance(batch.clean_latents, torch.Tensor):
            raise TypeError("LTX clean latents must be a tensor")
        denominator = self._active_sample_weights(
            batch.clean_latents,
            batch.loss_mask,
            batch.sample_weights,
        ).sum()
        if not bool(denominator.detach() > 0):
            raise ValueError("LTX loss has no active samples")
        return denominator


__all__ = [
    "LTXFlowMatchingObjective",
    "LTXTimestepSamplingConfig",
    "sample_ltx_sigmas",
]
