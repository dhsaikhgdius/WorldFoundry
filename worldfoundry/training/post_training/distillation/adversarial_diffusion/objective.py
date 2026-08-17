"""Complete model execution for Adversarial Diffusion Distillation."""

from __future__ import annotations

import torch
from torch import Tensor

from .adapters import audit_add_model_graph
from .config import ADDConfig, ADDNoiseSchedule
from .contracts import (
    ADDDecoderAdapter,
    ADDDiscriminatorAdapter,
    ADDDiscriminatorOutput,
    ADDLossResult,
    ADDPredictionAdapter,
    ADDTrainingBatch,
)
from .math import (
    add_forward_noise,
    discriminator_hinge_loss_per_sample,
    distillation_weights,
    feature_r1_penalty_per_sample,
    generator_hinge_loss_per_sample,
    pixel_distillation_loss_per_sample,
    sample_student_timesteps,
    sample_teacher_timesteps,
    schedule_coefficients,
)


def _randn_like(reference: Tensor, *, generator: torch.Generator | None) -> Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


class NativeADDLossAdapter:
    """Own student generation, stopped teacher targets, and feature-head GAN loss."""

    def __init__(
        self,
        *,
        student: ADDPredictionAdapter,
        teacher: ADDPredictionAdapter,
        decoder: ADDDecoderAdapter,
        discriminator: ADDDiscriminatorAdapter,
        student_schedule: ADDNoiseSchedule,
        teacher_schedule: ADDNoiseSchedule,
        config: ADDConfig,
    ) -> None:
        audit_add_model_graph(
            student=student,
            teacher=teacher,
            decoder=decoder,
            discriminator=discriminator,
            student_schedule=student_schedule,
            teacher_schedule=teacher_schedule,
            config=config,
        )
        self.student = student
        self.teacher = teacher
        self.decoder = decoder
        self.discriminator = discriminator
        self.student_schedule = student_schedule
        self.teacher_schedule = teacher_schedule
        self.config = config

    def loss_denominator(self, batch: ADDTrainingBatch, *, role: str) -> int:
        if not isinstance(batch, ADDTrainingBatch):
            raise TypeError("batch must be ADDTrainingBatch")
        if role not in {"generator", "discriminator"}:
            raise ValueError("ADD loss role must be generator or discriminator")
        return batch.batch_size

    def _clean_latents(self, batch: ADDTrainingBatch) -> Tensor:
        clean = batch.clean_latents
        if not isinstance(clean, Tensor) or clean.ndim < 2:
            raise TypeError("ADD clean_latents must be a [B,...] tensor")
        if not clean.is_floating_point():
            raise TypeError("ADD clean_latents must be floating point")
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("ADD clean_latents must be finite")
        return clean

    @staticmethod
    def _real_images(batch: ADDTrainingBatch) -> Tensor:
        images = batch.real_images
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise TypeError("ADD real_images must be a [B,C,H,W] tensor")
        if not images.is_floating_point():
            raise TypeError("ADD real_images must be floating point")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("ADD real_images must be finite")
        return images

    def _predict_student(
        self,
        batch: ADDTrainingBatch,
        *,
        generator: torch.Generator | None,
        training: bool,
    ) -> tuple[Tensor, Tensor]:
        clean = self._clean_latents(batch)
        timesteps = sample_student_timesteps(
            clean,
            self.config.student_timesteps,
            generator=generator,
        )
        alpha, sigma = schedule_coefficients(self.student_schedule, timesteps, clean)
        noisy = add_forward_noise(clean, _randn_like(clean, generator=generator), alpha, sigma)
        prediction = self.student.predict_clean(
            noisy,
            timesteps,
            sample_ids=batch.sample_ids,
            conditioning=batch.conditioning,
            training=training,
        )
        if not isinstance(prediction, Tensor) or prediction.shape != clean.shape:
            raise ValueError("ADD student clean prediction must preserve the diffusion-state shape")
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("ADD student prediction is non-finite")
        if training and not prediction.requires_grad:
            raise RuntimeError("ADD student training prediction is detached from its model graph")
        return prediction, timesteps

    def _decode(self, clean_latents: Tensor, *, expected_shape: torch.Size) -> Tensor:
        images = self.decoder.decode(clean_latents)
        if not isinstance(images, Tensor) or images.shape != expected_shape:
            raise ValueError("ADD decoder output must match real_images exactly")
        if not images.is_floating_point() or not bool(torch.isfinite(images).all()):
            raise FloatingPointError("ADD decoder output must be finite floating point")
        return images

    def _audit_discriminator_output(self, output: object) -> ADDDiscriminatorOutput:
        if not isinstance(output, ADDDiscriminatorOutput):
            raise TypeError("ADD discriminator must return ADDDiscriminatorOutput")
        if output.keys != self.config.feature_keys:
            raise ValueError("ADD discriminator output keys differ from the active config")
        return output

    def _teacher_target(
        self,
        generated_clean: Tensor,
        batch: ADDTrainingBatch,
        *,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        teacher_timesteps = sample_teacher_timesteps(
            generated_clean,
            minimum=self.config.teacher_timestep_min,
            maximum=self.config.teacher_timestep_max,
            probabilities=self.config.teacher_timestep_probabilities,
            generator=generator,
        )
        weights = distillation_weights(
            self.teacher_schedule,
            teacher_timesteps,
            generated_clean,
            weighting=self.config.distillation_weighting,
        )
        with torch.no_grad():
            stopped_clean = generated_clean.detach()
            alpha, sigma = schedule_coefficients(
                self.teacher_schedule,
                teacher_timesteps,
                stopped_clean,
            )
            teacher_noisy = add_forward_noise(
                stopped_clean,
                _randn_like(stopped_clean, generator=generator),
                alpha,
                sigma,
            )
            teacher_clean = self.teacher.predict_clean(
                teacher_noisy,
                teacher_timesteps,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=False,
            )
            if not isinstance(teacher_clean, Tensor) or teacher_clean.shape != stopped_clean.shape:
                raise ValueError("ADD teacher clean target must preserve the diffusion-state shape")
            teacher_images = self._decode(
                teacher_clean,
                expected_shape=torch.Size(batch.real_images.shape),
            ).detach()
        return teacher_images, teacher_timesteps, weights

    @staticmethod
    def _mean(per_sample: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(per_sample, Tensor) or per_sample.ndim != 1 or per_sample.shape[0] == 0:
            raise ValueError("ADD loss must contain one value per sample")
        numerator = per_sample.float().sum()
        denominator = torch.tensor(
            float(per_sample.shape[0]),
            device=per_sample.device,
            dtype=torch.float32,
        )
        return numerator / denominator, numerator, denominator

    def generator_loss(
        self,
        batch: ADDTrainingBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ADDLossResult:
        if not isinstance(batch, ADDTrainingBatch):
            raise TypeError("batch must be ADDTrainingBatch")
        real_images = self._real_images(batch)
        generated_clean, student_timesteps = self._predict_student(
            batch,
            generator=generator,
            training=True,
        )
        generated_images = self._decode(generated_clean, expected_shape=real_images.shape)
        if not generated_images.requires_grad:
            raise RuntimeError("ADD decoder detached the generated image from the student graph")
        teacher_images, teacher_timesteps, weights = self._teacher_target(
            generated_clean,
            batch,
            generator=generator,
        )
        distillation = pixel_distillation_loss_per_sample(
            generated_images,
            teacher_images,
            weights,
        )
        discriminator_output = self._audit_discriminator_output(
            self.discriminator.predict(
                generated_images,
                sample_ids=batch.sample_ids,
                conditioning=batch.discriminator_conditioning,
                track_image_grad=True,
                require_r1_inputs=False,
            )
        )
        adversarial = generator_hinge_loss_per_sample(discriminator_output.heads)
        per_sample = adversarial + self.config.distillation_weight * distillation
        loss, numerator, denominator = self._mean(per_sample)
        return ADDLossResult(
            loss=loss,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator,
                "adversarial": adversarial.detach().mean(),
                "distillation": distillation.detach().mean(),
                "student_timestep_mean": student_timesteps.float().mean(),
                "teacher_timestep_mean": teacher_timesteps.float().mean(),
                "distillation_weight_mean": weights.detach().mean(),
            },
        )

    def discriminator_loss(
        self,
        batch: ADDTrainingBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ADDLossResult:
        if not isinstance(batch, ADDTrainingBatch):
            raise TypeError("batch must be ADDTrainingBatch")
        real_images = self._real_images(batch)
        with torch.no_grad():
            generated_clean, student_timesteps = self._predict_student(
                batch,
                generator=generator,
                training=False,
            )
            generated_images = self._decode(
                generated_clean,
                expected_shape=real_images.shape,
            ).detach()
        real_output = self._audit_discriminator_output(
            self.discriminator.predict(
                real_images.detach(),
                sample_ids=batch.sample_ids,
                conditioning=batch.discriminator_conditioning,
                track_image_grad=False,
                require_r1_inputs=True,
            )
        )
        fake_output = self._audit_discriminator_output(
            self.discriminator.predict(
                generated_images,
                sample_ids=batch.sample_ids,
                conditioning=batch.discriminator_conditioning,
                track_image_grad=False,
                require_r1_inputs=False,
            )
        )
        adversarial, real_hinge, fake_hinge = discriminator_hinge_loss_per_sample(
            real_output.heads,
            fake_output.heads,
        )
        r1 = feature_r1_penalty_per_sample(real_output.heads)
        per_sample = adversarial + self.config.r1_weight * r1
        loss, numerator, denominator = self._mean(per_sample)
        return ADDLossResult(
            loss=loss,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator,
                "real_hinge": real_hinge.detach().mean(),
                "fake_hinge": fake_hinge.detach().mean(),
                "feature_r1": r1.detach().mean(),
                "student_timestep_mean": student_timesteps.float().mean(),
            },
        )


__all__ = ["NativeADDLossAdapter"]
