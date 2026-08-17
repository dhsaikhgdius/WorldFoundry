"""Model execution for native SANA-Sprint sCM-LADD objectives."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from worldfoundry.training.recipes.post_training.algorithms.scm_ladd import (
    SCMLADDAlgorithmSpec,
)

from .contracts import (
    SCMLADDDiscriminatorAdapter,
    SCMLADDLossResult,
    SCMLADDTrainingBatch,
    SCMVelocityPrediction,
    TrigFlowPredictionAdapter,
)
from .math import (
    _batch_coefficients,
    _normal_like,
    classifier_free_velocity,
    ladd_discriminator_hinge_loss,
    ladd_generator_hinge_loss,
    sample_trigflow_timesteps,
    scm_adaptive_loss,
    scm_tangent_target,
    trigflow_clean_prediction,
    trigflow_interpolate,
)


def _prediction(
    adapter: TrigFlowPredictionAdapter,
    scaled_noisy: torch.Tensor,
    timesteps: torch.Tensor,
    batch: SCMLADDTrainingBatch,
    *,
    conditioning: Mapping[str, object],
    training: bool,
    guidance_embedding_scale: float,
    return_log_variance: bool,
    branch: str = "positive",
) -> SCMVelocityPrediction:
    result = adapter.predict_velocity(
        scaled_noisy,
        timesteps,
        sample_ids=batch.sample_ids,
        conditioning=conditioning,
        training=training,
        guidance_embedding_scale=guidance_embedding_scale,
        return_log_variance=return_log_variance,
        branch=branch,
    )
    if not isinstance(result, SCMVelocityPrediction):
        raise TypeError("TrigFlow prediction adapters must return SCMVelocityPrediction")
    if not isinstance(result.velocity, torch.Tensor) or result.velocity.shape != scaled_noisy.shape:
        raise ValueError("TrigFlow velocity must match the latent tensor")
    return result


def _repeat_conditioning(
    value: Mapping[str, object],
    *,
    batch_size: int,
    repeats: int,
) -> Mapping[str, object]:
    """Repeat batch-owned conditioning without rolling prompt alignment."""

    def repeat(item: object) -> object:
        if isinstance(item, torch.Tensor) and item.ndim > 0 and item.shape[0] == batch_size:
            return torch.cat([item] * repeats, dim=0)
        if isinstance(item, Mapping):
            return {str(key): repeat(child) for key, child in item.items()}
        if isinstance(item, tuple) and len(item) == batch_size:
            return item * repeats
        if isinstance(item, list) and len(item) == batch_size:
            return item * repeats
        return item

    return {str(key): repeat(item) for key, item in value.items()}


def _with_cfg_scale(
    conditioning: Mapping[str, object],
    guidance_scales: torch.Tensor,
) -> Mapping[str, object]:
    values = dict(conditioning)
    values["cfg_scale"] = guidance_scales
    return values


class NativeSCMLADDLossAdapter:
    """Own the complete model-neutral sCM and LADD objective execution."""

    def __init__(
        self,
        student: TrigFlowPredictionAdapter,
        teacher: TrigFlowPredictionAdapter,
        discriminator: SCMLADDDiscriminatorAdapter,
        config: SCMLADDAlgorithmSpec,
    ) -> None:
        if not isinstance(student, TrigFlowPredictionAdapter):
            raise TypeError("student must implement TrigFlowPredictionAdapter")
        if not isinstance(teacher, TrigFlowPredictionAdapter):
            raise TypeError("teacher must implement TrigFlowPredictionAdapter")
        if not isinstance(discriminator, SCMLADDDiscriminatorAdapter):
            raise TypeError("discriminator must implement SCMLADDDiscriminatorAdapter")
        if not isinstance(config, SCMLADDAlgorithmSpec):
            raise TypeError("config must be SCMLADDAlgorithmSpec")
        self.student = student
        self.teacher = teacher
        self.discriminator = discriminator
        self.config = config

    def loss_denominator(
        self,
        batch: SCMLADDTrainingBatch,
        *,
        role: str,
    ) -> int:
        """Declare the sample-mean reduction before model collectives."""

        if not isinstance(batch, SCMLADDTrainingBatch):
            raise TypeError("batch must be SCMLADDTrainingBatch")
        if role not in {"generator", "discriminator"}:
            raise ValueError("SCM-LADD loss role must be generator or discriminator")
        return batch.batch_size

    def _teacher_path_velocity(
        self,
        scaled_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        batch: SCMLADDTrainingBatch,
        *,
        guidance_scales: torch.Tensor,
        conditioning: Mapping[str, object],
        unconditional_conditioning: Mapping[str, object],
    ) -> torch.Tensor:
        with torch.no_grad():
            conditional = _prediction(
                self.teacher,
                scaled_noisy,
                timesteps,
                batch,
                conditioning=conditioning,
                training=False,
                guidance_embedding_scale=self.config.guidance_embedding_scale,
                return_log_variance=False,
            ).velocity
            unconditional = _prediction(
                self.teacher,
                scaled_noisy,
                timesteps,
                batch,
                conditioning=unconditional_conditioning,
                training=False,
                guidance_embedding_scale=self.config.guidance_embedding_scale,
                return_log_variance=False,
                branch="negative",
            ).velocity
            guided = classifier_free_velocity(conditional, unconditional, guidance_scales)
            return guided * self.config.sigma_data

    def _sample_guidance_scales(
        self,
        reference: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        choices = torch.tensor(
            self.config.teacher_guidance_scales,
            device=reference.device,
            dtype=torch.float32,
        )
        indices = torch.randint(
            0,
            choices.numel(),
            (reference.shape[0],),
            device=reference.device,
            generator=generator,
        )
        return choices[indices]

    def generator_loss(
        self,
        batch: SCMLADDTrainingBatch,
        *,
        training_iteration: int,
        generator: torch.Generator | None = None,
    ) -> SCMLADDLossResult:
        if not isinstance(batch, SCMLADDTrainingBatch):
            raise TypeError("batch must be SCMLADDTrainingBatch")
        clean = batch.clean_latents
        if not isinstance(clean, torch.Tensor):
            raise TypeError("clean_latents must be a torch.Tensor")
        config = self.config
        guidance_scales = self._sample_guidance_scales(clean, generator=generator)
        conditioning = _with_cfg_scale(batch.conditioning, guidance_scales)
        unconditional_conditioning = _with_cfg_scale(
            batch.unconditional_conditioning,
            guidance_scales,
        )
        consistency_t = sample_trigflow_timesteps(
            clean,
            logit_mean=config.generator_logit_mean,
            logit_std=config.generator_logit_std,
            sigma_data=config.sigma_data,
            generator=generator,
        )
        noise = _normal_like(clean, sigma_data=config.sigma_data, generator=generator)
        noisy = trigflow_interpolate(clean, noise, consistency_t)
        scaled_noisy = noisy / config.sigma_data
        teacher_path = self._teacher_path_velocity(
            scaled_noisy,
            consistency_t,
            batch,
            guidance_scales=guidance_scales,
            conditioning=conditioning,
            unconditional_conditioning=unconditional_conditioning,
        )
        coefficient = torch.cos(consistency_t) * torch.sin(consistency_t)
        tangent_x = _batch_coefficients(coefficient, scaled_noisy) * teacher_path / config.sigma_data
        tangent_t = coefficient

        def stopped_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return _prediction(
                self.student,
                x,
                t,
                batch,
                conditioning=conditioning,
                training=True,
                guidance_embedding_scale=config.guidance_embedding_scale,
                return_log_variance=False,
            ).velocity

        with torch.no_grad():
            stopped_velocity, directional_derivative = torch.func.jvp(
                stopped_model,
                (scaled_noisy, consistency_t),
                (tangent_x, tangent_t),
            )
        current = _prediction(
            self.student,
            scaled_noisy,
            consistency_t,
            batch,
            conditioning=conditioning,
            training=True,
            guidance_embedding_scale=config.guidance_embedding_scale,
            return_log_variance=True,
        )
        if not isinstance(current.log_variance, torch.Tensor):
            raise ValueError("SCM student must return learned log_variance")
        if isinstance(training_iteration, bool) or int(training_iteration) <= 0:
            raise ValueError("training_iteration must be a positive integer")
        warmup = min(1.0, float(training_iteration) / float(config.tangent_warmup_steps))
        tangent_target, tangent_norm = scm_tangent_target(
            noisy,
            stopped_velocity.detach(),
            directional_derivative.detach(),
            teacher_path,
            consistency_t,
            sigma_data=config.sigma_data,
            warmup_ratio=warmup,
            normalization_constant=config.tangent_normalization_constant,
        )
        consistency_loss, consistency_no_logvar, consistency_unweighted = scm_adaptive_loss(
            current.velocity,
            stopped_velocity.detach(),
            tangent_target.detach(),
            current.log_variance,
            consistency_t,
            sigma_data=config.sigma_data,
        )

        adversarial_t = sample_trigflow_timesteps(
            clean,
            logit_mean=config.generator_logit_mean,
            logit_std=config.generator_logit_std,
            sigma_data=config.sigma_data,
            max_time_probability=(config.max_time_probability if config.largest_time_enabled else 0.0),
            max_time=config.largest_time,
            generator=generator,
        )
        adversarial_noise = _normal_like(clean, sigma_data=config.sigma_data, generator=generator)
        adversarial_noisy = trigflow_interpolate(clean, adversarial_noise, adversarial_t)
        adversarial_velocity = _prediction(
            self.student,
            adversarial_noisy / config.sigma_data,
            adversarial_t,
            batch,
            conditioning=conditioning,
            training=True,
            guidance_embedding_scale=config.guidance_embedding_scale,
            return_log_variance=False,
        ).velocity
        generated_clean = trigflow_clean_prediction(
            adversarial_noisy,
            adversarial_velocity,
            adversarial_t,
            sigma_data=config.sigma_data,
        )
        discriminator_t = sample_trigflow_timesteps(
            clean,
            logit_mean=config.discriminator_logit_mean,
            logit_std=config.discriminator_logit_std,
            sigma_data=config.sigma_data,
            generator=generator,
        )
        discriminator_noise = _normal_like(clean, sigma_data=config.sigma_data, generator=generator)
        fake_noisy = trigflow_interpolate(generated_clean, discriminator_noise, discriminator_t)
        fake_logits = self.discriminator.predict_logits(
            fake_noisy / config.sigma_data,
            discriminator_t,
            sample_ids=batch.sample_ids,
            conditioning=conditioning,
            training=False,
            head_block_ids=config.discriminator_head_block_ids,
        )
        adversarial_loss = ladd_generator_hinge_loss(fake_logits)
        total = config.consistency_weight * consistency_loss + config.adversarial_weight * adversarial_loss
        denominator = torch.tensor(batch.batch_size, device=total.device, dtype=torch.float32)
        return SCMLADDLossResult(
            loss=total,
            metrics={
                "loss_denominator": denominator,
                "consistency_loss": consistency_loss.detach(),
                "consistency_loss_no_logvar": consistency_no_logvar.detach(),
                "consistency_loss_unweighted": consistency_unweighted.detach(),
                "adversarial_loss": adversarial_loss.detach(),
                "tangent_norm": tangent_norm.detach().mean(),
                "warmup_ratio": torch.tensor(warmup, device=total.device),
                "consistency_timestep_mean": consistency_t.detach().mean(),
                "adversarial_timestep_mean": adversarial_t.detach().mean(),
                "discriminator_timestep_mean": discriminator_t.detach().mean(),
                "guidance_scale_mean": guidance_scales.detach().mean(),
            },
        )

    def discriminator_loss(
        self,
        batch: SCMLADDTrainingBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> SCMLADDLossResult:
        if not isinstance(batch, SCMLADDTrainingBatch):
            raise TypeError("batch must be SCMLADDTrainingBatch")
        clean = batch.clean_latents
        if not isinstance(clean, torch.Tensor):
            raise TypeError("clean_latents must be a torch.Tensor")
        config = self.config
        guidance_scales = self._sample_guidance_scales(clean, generator=generator)
        conditioning = _with_cfg_scale(batch.conditioning, guidance_scales)
        generation_t = sample_trigflow_timesteps(
            clean,
            logit_mean=config.generator_logit_mean,
            logit_std=config.generator_logit_std,
            sigma_data=config.sigma_data,
            max_time_probability=(config.max_time_probability if config.largest_time_enabled else 0.0),
            max_time=config.largest_time,
            generator=generator,
        )
        generation_noise = _normal_like(clean, sigma_data=config.sigma_data, generator=generator)
        generation_noisy = trigflow_interpolate(clean, generation_noise, generation_t)
        with torch.no_grad():
            student_velocity = _prediction(
                self.student,
                generation_noisy / config.sigma_data,
                generation_t,
                batch,
                conditioning=conditioning,
                training=False,
                guidance_embedding_scale=config.guidance_embedding_scale,
                return_log_variance=False,
            ).velocity
            generated_clean = trigflow_clean_prediction(
                generation_noisy,
                student_velocity,
                generation_t,
                sigma_data=config.sigma_data,
            )
        fake_discriminator_t = sample_trigflow_timesteps(
            clean,
            logit_mean=config.discriminator_logit_mean,
            logit_std=config.discriminator_logit_std,
            sigma_data=config.sigma_data,
            generator=generator,
        )
        real_discriminator_t = (
            sample_trigflow_timesteps(
                clean,
                logit_mean=config.discriminator_logit_mean,
                logit_std=config.discriminator_logit_std,
                sigma_data=config.sigma_data,
                generator=generator,
            )
            if config.independent_real_fake_discriminator_times
            else fake_discriminator_t
        )
        fake_noise = _normal_like(clean, sigma_data=config.sigma_data, generator=generator)
        real_noise = _normal_like(clean, sigma_data=config.sigma_data, generator=generator)
        fake_noisy = trigflow_interpolate(generated_clean.detach(), fake_noise, fake_discriminator_t)
        real_noisy = trigflow_interpolate(clean, real_noise, real_discriminator_t)
        fake_sample_ids = batch.sample_ids
        fake_conditioning = conditioning
        if config.misaligned_pairs and batch.batch_size > 1:
            shifted_clean = torch.roll(clean, shifts=1, dims=0)
            shifted_t = sample_trigflow_timesteps(
                clean,
                logit_mean=config.discriminator_logit_mean,
                logit_std=config.discriminator_logit_std,
                sigma_data=config.sigma_data,
                generator=generator,
            )
            shifted_noise = _normal_like(clean, sigma_data=config.sigma_data, generator=generator)
            shifted_noisy = trigflow_interpolate(shifted_clean, shifted_noise, shifted_t)
            fake_noisy = torch.cat([fake_noisy, shifted_noisy], dim=0)
            fake_discriminator_t = torch.cat([fake_discriminator_t, shifted_t], dim=0)
            fake_sample_ids = batch.sample_ids + tuple(f"{sample_id}#misaligned" for sample_id in batch.sample_ids)
            fake_conditioning = _repeat_conditioning(
                conditioning,
                batch_size=batch.batch_size,
                repeats=2,
            )
        fake_logits = self.discriminator.predict_logits(
            fake_noisy / config.sigma_data,
            fake_discriminator_t,
            sample_ids=fake_sample_ids,
            conditioning=fake_conditioning,
            training=True,
            head_block_ids=config.discriminator_head_block_ids,
        )
        real_logits = self.discriminator.predict_logits(
            real_noisy / config.sigma_data,
            real_discriminator_t,
            sample_ids=batch.sample_ids,
            conditioning=conditioning,
            training=True,
            head_block_ids=config.discriminator_head_block_ids,
        )
        loss, real_loss, fake_loss = ladd_discriminator_hinge_loss(real_logits, fake_logits)
        denominator = torch.tensor(batch.batch_size, device=loss.device, dtype=torch.float32)
        return SCMLADDLossResult(
            loss=loss,
            metrics={
                "loss_denominator": denominator,
                "real_hinge_loss": real_loss.detach(),
                "fake_hinge_loss": fake_loss.detach(),
                "generation_timestep_mean": generation_t.detach().mean(),
                "fake_discriminator_timestep_mean": fake_discriminator_t.detach().mean(),
                "real_discriminator_timestep_mean": real_discriminator_t.detach().mean(),
                "guidance_scale_mean": guidance_scales.detach().mean(),
                "misaligned_pairs": torch.tensor(
                    batch.batch_size if config.misaligned_pairs and batch.batch_size > 1 else 0,
                    device=loss.device,
                    dtype=torch.int64,
                ),
            },
        )


__all__ = [
    "NativeSCMLADDLossAdapter",
]
