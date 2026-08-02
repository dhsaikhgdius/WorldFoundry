"""Teacher-guided latent consistency distillation objective."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from worldfoundry.core.io.integrity import canonical_sha256

from .config import (
    LatentConsistencyConfig,
    LatentConsistencyNoiseSchedule,
    build_latent_consistency_ddim_schedule,
)
from .contracts import (
    LatentConsistencyPredictionAdapter,
    LatentConsistencyRandomInputs,
    LatentConsistencyTrainingBatch,
)
from .math import (
    add_forward_diffusion_noise,
    append_dims,
    boundary_condition_scalings,
    classifier_free_guidance,
    consistency_prediction,
    deterministic_ddim_step,
    gather_schedule_coefficients,
    guidance_scale_embedding,
    latent_consistency_elementwise_loss,
    prediction_to_origin_and_epsilon,
)


@dataclass(frozen=True, slots=True)
class LatentConsistencyLossResult:
    loss: Tensor
    metrics: Mapping[str, object]


def _module(
    adapter: LatentConsistencyPredictionAdapter,
    *,
    role: str,
) -> nn.Module:
    if not isinstance(adapter, LatentConsistencyPredictionAdapter):
        raise TypeError(f"{role} must implement LatentConsistencyPredictionAdapter")
    module = adapter.module
    if not isinstance(module, nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    return module


def _prediction(
    adapter: LatentConsistencyPredictionAdapter,
    noisy_latents: Tensor,
    timesteps: Tensor,
    batch: LatentConsistencyTrainingBatch,
    *,
    guidance_embedding: Tensor | None,
    conditioning: Mapping[str, object],
    training: bool,
    branch: str,
    role: str,
) -> Tensor:
    value = adapter.predict_model_output(
        noisy_latents,
        timesteps,
        guidance_embedding=guidance_embedding,
        sample_ids=batch.sample_ids,
        conditioning=conditioning,
        training=training,
        branch=branch,
    )
    if not isinstance(value, Tensor) or value.shape != noisy_latents.shape:
        raise ValueError(f"{role} model output must match the latent shape")
    if not value.is_floating_point():
        raise TypeError(f"{role} model output must be floating point")
    if value.device != noisy_latents.device:
        raise ValueError(f"{role} model output must remain on the latent device")
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"{role} model output is non-finite")
    return value


class LatentConsistencyObjective:
    """Execute the online, frozen teacher, and frozen EMA target paths."""

    def __init__(
        self,
        *,
        student: LatentConsistencyPredictionAdapter,
        teacher: LatentConsistencyPredictionAdapter,
        ema_target: LatentConsistencyPredictionAdapter,
        noise_schedule: LatentConsistencyNoiseSchedule,
        config: LatentConsistencyConfig,
    ) -> None:
        student_module = _module(student, role="latent consistency student")
        teacher_module = _module(teacher, role="latent consistency teacher")
        target_module = _module(ema_target, role="latent consistency EMA target")
        modules = (student_module, teacher_module, target_module)
        if len({id(module) for module in modules}) != 3:
            raise ValueError("latent consistency roles must be distinct modules")
        parameter_ids = tuple({id(parameter) for parameter in module.parameters()} for module in modules)
        if any(
            parameter_ids[left] & parameter_ids[right]
            for left in range(len(parameter_ids))
            for right in range(left + 1, len(parameter_ids))
        ):
            raise ValueError("latent consistency roles cannot share parameters")
        if not isinstance(noise_schedule, LatentConsistencyNoiseSchedule):
            raise TypeError("noise_schedule must be LatentConsistencyNoiseSchedule")
        if not isinstance(config, LatentConsistencyConfig):
            raise TypeError("config must be LatentConsistencyConfig")

        teacher_module.requires_grad_(False)
        target_module.requires_grad_(False)
        teacher_module.eval()
        target_module.eval()
        self.student = student
        self.teacher = teacher
        self.ema_target = ema_target
        self.student_module = student_module
        self.teacher_module = teacher_module
        self.ema_target_module = target_module
        self.noise_schedule = noise_schedule
        self.config = config
        self.ddim_schedule = build_latent_consistency_ddim_schedule(
            noise_schedule,
            config,
        )
        self._execution_digest = canonical_sha256(
            {
                "config": config.digest,
                "noise_schedule": noise_schedule.digest,
            }
        )
        self._schedule_cache: dict[str, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = {}

    @property
    def config_digest(self) -> str:
        """Digest every value that changes objective execution or sampling."""

        return self._execution_digest

    @property
    def pair_count(self) -> int:
        return self.ddim_schedule.pair_count

    def _schedule_tensors(
        self,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        key = str(device)
        cached = self._schedule_cache.get(key)
        if cached is None:
            alpha_cumprods = torch.tensor(
                self.noise_schedule.alpha_cumprods,
                device=device,
                dtype=torch.float32,
            )
            alpha = alpha_cumprods.sqrt()
            sigma = (1.0 - alpha_cumprods).clamp_min(0.0).sqrt()
            starts = torch.tensor(
                self.ddim_schedule.start_timesteps,
                device=device,
                dtype=torch.int64,
            )
            ends = torch.tensor(
                self.ddim_schedule.end_timesteps,
                device=device,
                dtype=torch.int64,
            )
            previous_alphas = torch.tensor(
                self.ddim_schedule.previous_alpha_cumprods,
                device=device,
                dtype=torch.float32,
            )
            cached = (alpha, sigma, starts, ends, previous_alphas)
            self._schedule_cache[key] = cached
        return cached

    def loss_denominator(self, batch: LatentConsistencyTrainingBatch) -> int:
        if not isinstance(batch, LatentConsistencyTrainingBatch):
            raise TypeError("batch must be LatentConsistencyTrainingBatch")
        clean = batch.clean_latents
        if not isinstance(clean, Tensor):
            raise TypeError("clean_latents must be a torch.Tensor")
        return clean.numel()

    def _validate_random_inputs(
        self,
        clean: Tensor,
        random_inputs: LatentConsistencyRandomInputs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(random_inputs, LatentConsistencyRandomInputs):
            raise TypeError("random_inputs must be LatentConsistencyRandomInputs")
        noise = random_inputs.noise
        indices = random_inputs.timestep_indices
        guidance = random_inputs.guidance_coefficients
        if not isinstance(noise, Tensor) or noise.shape != clean.shape:
            raise ValueError("noise must match clean_latents")
        if noise.device != clean.device or noise.dtype != clean.dtype:
            raise ValueError("noise must share clean_latents device and dtype")
        if not isinstance(indices, Tensor) or indices.shape != (clean.shape[0],):
            raise ValueError("timestep_indices must have shape [B]")
        if indices.device != clean.device or indices.dtype != torch.int64:
            raise ValueError("timestep_indices must be int64 on the latent device")
        if not bool(((indices >= 0) & (indices < self.pair_count)).all()):
            raise ValueError("timestep_indices contain an invalid DDIM pair")
        if not isinstance(guidance, Tensor) or guidance.shape != (clean.shape[0],):
            raise ValueError("guidance_coefficients must have shape [B]")
        if guidance.device != clean.device or not guidance.is_floating_point():
            raise ValueError("guidance_coefficients must be floating on the latent device")
        minimum = self.config.guidance_coefficient_min
        maximum = self.config.guidance_coefficient_max
        if not bool(torch.isfinite(guidance).all()) or not bool(((guidance >= minimum) & (guidance <= maximum)).all()):
            raise ValueError("guidance_coefficients are outside configured bounds")
        return noise, indices, guidance

    def loss(
        self,
        batch: LatentConsistencyTrainingBatch,
        *,
        random_inputs: LatentConsistencyRandomInputs,
    ) -> LatentConsistencyLossResult:
        if not isinstance(batch, LatentConsistencyTrainingBatch):
            raise TypeError("batch must be LatentConsistencyTrainingBatch")
        clean = batch.clean_latents
        if not isinstance(clean, Tensor) or not clean.is_floating_point():
            raise TypeError("clean_latents must be a floating torch.Tensor")
        noise, indices, guidance = self._validate_random_inputs(clean, random_inputs)
        alpha_schedule, sigma_schedule, starts, ends, previous_alphas = self._schedule_tensors(clean.device)
        start_timesteps = starts.gather(0, indices)
        end_timesteps = ends.gather(0, indices)
        start_alpha = gather_schedule_coefficients(
            alpha_schedule,
            start_timesteps,
            clean,
        )
        start_sigma = gather_schedule_coefficients(
            sigma_schedule,
            start_timesteps,
            clean,
        )
        noisy = add_forward_diffusion_noise(
            clean,
            noise,
            start_alpha,
            start_sigma,
        )
        guidance_embedding = guidance_scale_embedding(
            guidance,
            embedding_dim=self.config.guidance_embedding_dim,
            embedding_scale=self.config.guidance_embedding_scale,
            max_period=self.config.guidance_embedding_max_period,
            dtype=clean.dtype,
        )
        start_skip, start_out = boundary_condition_scalings(
            start_timesteps,
            sigma_data=self.config.sigma_data,
            timestep_scaling=self.config.timestep_scaling,
        )
        start_skip = append_dims(start_skip, clean.ndim)
        start_out = append_dims(start_out, clean.ndim)

        student_output = _prediction(
            self.student,
            noisy,
            start_timesteps,
            batch,
            guidance_embedding=guidance_embedding,
            conditioning=batch.conditioning,
            training=True,
            branch="positive",
            role="latent consistency student",
        )
        student_origin, _ = prediction_to_origin_and_epsilon(
            student_output,
            noisy,
            start_alpha,
            start_sigma,
            prediction_type=self.config.prediction_type,
        )
        prediction = consistency_prediction(
            noisy,
            student_origin,
            start_skip,
            start_out,
        )

        self.teacher_module.eval()
        self.ema_target_module.eval()
        with torch.no_grad():
            conditional_output = _prediction(
                self.teacher,
                noisy,
                start_timesteps,
                batch,
                guidance_embedding=None,
                conditioning=batch.conditioning,
                training=False,
                branch="positive",
                role="latent consistency teacher positive branch",
            )
            unconditional_output = _prediction(
                self.teacher,
                noisy,
                start_timesteps,
                batch,
                guidance_embedding=None,
                conditioning=batch.unconditional_conditioning,
                training=False,
                branch="negative",
                role="latent consistency teacher negative branch",
            )
            conditional_origin, conditional_epsilon = prediction_to_origin_and_epsilon(
                conditional_output,
                noisy,
                start_alpha,
                start_sigma,
                prediction_type=self.config.prediction_type,
            )
            unconditional_origin, unconditional_epsilon = prediction_to_origin_and_epsilon(
                unconditional_output,
                noisy,
                start_alpha,
                start_sigma,
                prediction_type=self.config.prediction_type,
            )
            guided_origin = classifier_free_guidance(
                conditional_origin,
                unconditional_origin,
                guidance,
            )
            guided_epsilon = classifier_free_guidance(
                conditional_epsilon,
                unconditional_epsilon,
                guidance,
            )
            previous_alpha = append_dims(
                previous_alphas.gather(0, indices),
                clean.ndim,
            )
            previous_latents = deterministic_ddim_step(
                guided_origin,
                guided_epsilon,
                previous_alpha,
            )
            target_output = _prediction(
                self.ema_target,
                previous_latents,
                end_timesteps,
                batch,
                guidance_embedding=guidance_embedding,
                conditioning=batch.conditioning,
                training=False,
                branch="positive",
                role="latent consistency EMA target",
            )
            end_alpha = gather_schedule_coefficients(
                alpha_schedule,
                end_timesteps,
                previous_latents,
            )
            end_sigma = gather_schedule_coefficients(
                sigma_schedule,
                end_timesteps,
                previous_latents,
            )
            target_origin, _ = prediction_to_origin_and_epsilon(
                target_output,
                previous_latents,
                end_alpha,
                end_sigma,
                prediction_type=self.config.prediction_type,
            )
            end_skip, end_out = boundary_condition_scalings(
                end_timesteps,
                sigma_data=self.config.sigma_data,
                timestep_scaling=self.config.timestep_scaling,
            )
            target = consistency_prediction(
                previous_latents,
                target_origin,
                append_dims(end_skip, clean.ndim),
                append_dims(end_out, clean.ndim),
            ).detach()

        elementwise = latent_consistency_elementwise_loss(
            prediction,
            target,
            loss_type=self.config.loss_type,
            pseudo_huber_c=self.config.pseudo_huber_c,
        )
        loss = elementwise.mean()
        if loss.numel() != 1 or not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("non-finite latent consistency loss")
        per_sample = elementwise.flatten(1).mean(dim=1)
        return LatentConsistencyLossResult(
            loss=loss,
            metrics={
                "loss_denominator": clean.numel(),
                "per_sample_loss": per_sample.detach(),
                "start_timesteps": start_timesteps.detach(),
                "end_timesteps": end_timesteps.detach(),
                "guidance_coefficients": guidance.detach(),
            },
        )


__all__ = ["LatentConsistencyLossResult", "LatentConsistencyObjective"]
