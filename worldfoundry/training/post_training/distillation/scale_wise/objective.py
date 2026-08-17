"""Native SwD generator, fake-diffusion, GAN, and MMD objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ScaleWiseConfig
from .contracts import (
    ScaleWiseCriticAdapter,
    ScaleWisePredictionAdapter,
    ScaleWiseTrainingBatch,
)
from .math import (
    classifier_free_guidance,
    clean_from_velocity,
    discriminator_logistic_loss,
    dmd_loss_per_sample,
    fake_diffusion_loss_per_sample,
    flow_noise,
    generator_logistic_loss,
    mmd_loss,
    pool_token_features,
    upscale_previous_latents,
)


@dataclass(frozen=True, slots=True)
class ScaleWiseLossResult:
    loss: torch.Tensor
    metrics: dict[str, object]


@dataclass(frozen=True, slots=True)
class ScaleWiseStudentSample:
    generated: torch.Tensor
    noisy_input: torch.Tensor
    start_sigmas: torch.Tensor


class FlowScaleWiseLossAdapter:
    """Compose the released scale-wise losses around native model seams."""

    def __init__(
        self,
        student: ScaleWisePredictionAdapter,
        teacher: ScaleWisePredictionAdapter,
        fake_score: ScaleWiseCriticAdapter,
        config: ScaleWiseConfig,
    ) -> None:
        if not isinstance(student, ScaleWisePredictionAdapter):
            raise TypeError("student must implement ScaleWisePredictionAdapter")
        if not isinstance(teacher, ScaleWisePredictionAdapter):
            raise TypeError("teacher must implement ScaleWisePredictionAdapter")
        if not isinstance(fake_score, ScaleWiseCriticAdapter):
            raise TypeError("fake_score must implement ScaleWiseCriticAdapter")
        if not isinstance(config, ScaleWiseConfig):
            raise TypeError("config must be ScaleWiseConfig")
        self.student = student
        self.teacher = teacher
        self.fake_score = fake_score
        self.config = config
        self.num_intervals = config.schedule.num_intervals
        self.fake_updates_per_iteration = config.fake_updates_per_iteration
        self.batch_mmd = config.batch_mmd

    @staticmethod
    def _tensor(value: object, *, field_name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{field_name} must be a torch.Tensor")
        return value

    def _batch(self, batch: object) -> ScaleWiseTrainingBatch:
        if not isinstance(batch, ScaleWiseTrainingBatch):
            raise TypeError("scale-wise objective requires ScaleWiseTrainingBatch")
        current = self._tensor(batch.current_latents, field_name="current_latents")
        previous = self._tensor(batch.previous_latents, field_name="previous_latents")
        expected_scale = self.config.schedule.scale(batch.interval_index)
        previous_scale = self.config.schedule.previous_scale(batch.interval_index)
        if current.shape[-2:] != (expected_scale, expected_scale):
            raise ValueError("current_latents do not match the selected scale interval")
        if previous.shape[-2:] != (previous_scale, previous_scale):
            raise ValueError("previous_latents do not match the preceding scale interval")
        return batch

    def loss_denominator(
        self,
        batch: ScaleWiseTrainingBatch,
        *,
        role: str,
    ) -> object:
        if role not in {"student", "fake-score"}:
            raise ValueError("scale-wise role must be 'student' or 'fake-score'")
        return self._batch(batch).batch_size

    @staticmethod
    def _randn_like(
        reference: torch.Tensor,
        *,
        generator: object | None,
    ) -> torch.Tensor:
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        return torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )

    def _sample_indices(
        self,
        batch_size: int,
        *,
        start: int,
        end: int,
        device: torch.device,
        generator: object | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        indices = torch.randint(
            start,
            end,
            (batch_size,),
            device=device,
            generator=generator,
        )
        schedule = torch.tensor(
            self.config.schedule.solver_sigmas,
            device=device,
            dtype=torch.float32,
        )
        return indices, schedule[indices]

    def _prediction(
        self,
        adapter: ScaleWisePredictionAdapter,
        noisy: torch.Tensor,
        sigmas: torch.Tensor,
        batch: ScaleWiseTrainingBatch,
        *,
        guidance_scale: float,
        training: bool,
    ) -> torch.Tensor:
        conditional = self._tensor(
            adapter.predict_velocity(
                noisy,
                sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=training,
                branch="positive",
            ),
            field_name="conditional velocity",
        )
        if guidance_scale <= 1.0:
            return conditional
        unconditional = self._tensor(
            adapter.predict_velocity(
                noisy,
                sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.unconditional_conditioning,
                training=training,
                branch="unconditional",
            ),
            field_name="unconditional velocity",
        )
        return classifier_free_guidance(
            unconditional,
            conditional,
            guidance_scale,
        )

    def sample_student(
        self,
        batch: ScaleWiseTrainingBatch,
        *,
        generator: object | None,
        training: bool,
    ) -> ScaleWiseStudentSample:
        resolved = self._batch(batch)
        previous = self._tensor(
            resolved.previous_latents,
            field_name="previous_latents",
        )
        student_input = upscale_previous_latents(
            previous,
            current_scale=self.config.schedule.scale(resolved.interval_index),
        )
        start_sigmas = torch.full(
            (resolved.batch_size,),
            self.config.schedule.start_sigma(resolved.interval_index),
            device=student_input.device,
            dtype=torch.float32,
        )
        noise = self._randn_like(student_input, generator=generator)
        noisy = flow_noise(student_input, noise, start_sigmas)
        velocity = self._prediction(
            self.student,
            noisy,
            start_sigmas,
            resolved,
            guidance_scale=1.0,
            training=training,
        )
        return ScaleWiseStudentSample(
            generated=clean_from_velocity(noisy, velocity, start_sigmas),
            noisy_input=noisy,
            start_sigmas=start_sigmas,
        )

    def _critic_features(
        self,
        noisy: torch.Tensor,
        sigmas: torch.Tensor,
        batch: ScaleWiseTrainingBatch,
        *,
        blocks: tuple[int, ...],
        training: bool,
    ) -> tuple[torch.Tensor, ...]:
        raw = self.fake_score.extract_features(
            noisy,
            sigmas,
            sample_ids=batch.sample_ids,
            conditioning=batch.conditioning,
            block_indices=blocks,
            training=training,
        )
        features = tuple(
            self._tensor(value, field_name="critic feature") for value in raw
        )
        if len(features) != len(blocks):
            raise RuntimeError("critic did not return one feature tensor per requested block")
        return features

    def _classifier_logits(
        self,
        features: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        raw = self.fake_score.classify_features(pool_token_features(features))
        logits = tuple(
            self._tensor(value, field_name="classifier logit") for value in raw
        )
        if len(logits) != len(features):
            raise RuntimeError("critic classifier did not return one logit per feature")
        return logits

    def student_loss(
        self,
        batch: ScaleWiseTrainingBatch,
        *,
        generator: object | None = None,
    ) -> ScaleWiseLossResult:
        resolved = self._batch(batch)
        current = self._tensor(resolved.current_latents, field_name="current_latents")
        sample = self.sample_student(
            resolved,
            generator=generator,
            training=True,
        )
        generated = sample.generated
        total_per_sample = torch.zeros(
            resolved.batch_size,
            device=generated.device,
            dtype=torch.float32,
        )
        metrics: dict[str, object] = {}
        dmd_noisy: torch.Tensor | None = None
        dmd_sigmas: torch.Tensor | None = None

        if self.config.dmd_enabled:
            dmd_indices, dmd_sigmas = self._sample_indices(
                resolved.batch_size,
                start=self.config.dmd_noise_start_index,
                end=self.config.dmd_noise_end_index,
                device=generated.device,
                generator=generator,
            )
            noise = self._randn_like(generated, generator=generator)
            dmd_noisy = flow_noise(generated, noise, dmd_sigmas)
            with torch.no_grad():
                real_velocity = self._prediction(
                    self.teacher,
                    dmd_noisy,
                    dmd_sigmas,
                    resolved,
                    guidance_scale=self.config.teacher_guidance_scale,
                    training=False,
                )
                fake_velocity = self._prediction(
                    self.fake_score,
                    dmd_noisy,
                    dmd_sigmas,
                    resolved,
                    guidance_scale=self.config.fake_guidance_scale,
                    training=False,
                )
                real_clean = clean_from_velocity(
                    dmd_noisy,
                    real_velocity,
                    dmd_sigmas,
                )
                fake_clean = clean_from_velocity(
                    dmd_noisy,
                    fake_velocity,
                    dmd_sigmas,
                )
            dmd_per_sample, gradient, normalizer = dmd_loss_per_sample(
                generated,
                real_clean,
                fake_clean,
            )
            total_per_sample = (
                total_per_sample + self.config.dmd_loss_weight * dmd_per_sample
            )
            metrics.update(
                {
                    "dmd_loss": dmd_per_sample.detach().mean(),
                    "dmd_noise_indices": dmd_indices.detach(),
                    "dmd_gradient_abs_mean": gradient.detach().abs().mean(),
                    "dmd_normalizer_mean": normalizer.detach().mean(),
                }
            )

        if self.config.gan_enabled:
            assert dmd_noisy is not None and dmd_sigmas is not None
            fake_features = self._critic_features(
                dmd_noisy,
                dmd_sigmas,
                resolved,
                blocks=self.config.classifier_blocks,
                training=False,
            )
            generator_gan = generator_logistic_loss(
                self._classifier_logits(fake_features)
            )
            total_per_sample = (
                total_per_sample
                + self.config.generator_gan_weight * generator_gan
            )
            metrics["generator_gan_loss"] = generator_gan.detach().mean()

        if self.config.mmd_enabled:
            mmd_indices, mmd_sigmas = self._sample_indices(
                resolved.batch_size,
                start=self.config.mmd_noise_start_index,
                end=self.config.mmd_noise_end_index,
                device=generated.device,
                generator=generator,
            )
            noise = self._randn_like(generated, generator=generator)
            noisy_real = flow_noise(current, noise, mmd_sigmas)
            noisy_fake = flow_noise(generated, noise, mmd_sigmas)
            fake_features = self._critic_features(
                noisy_fake,
                mmd_sigmas,
                resolved,
                blocks=self.config.mmd_blocks,
                training=False,
            )
            real_features = self._critic_features(
                noisy_real,
                mmd_sigmas,
                resolved,
                blocks=self.config.mmd_blocks,
                training=False,
            )
            # The released trainer indexes the first collected feature tensor.
            feature_mmd = mmd_loss(
                real_features[0],
                fake_features[0],
                kernel=self.config.mmd_kernel,
                rbf_sigma=self.config.mmd_rbf_sigma,
                batch_mmd=self.config.batch_mmd,
                huber_c=self.config.huber_c,
            )
            total_per_sample = (
                total_per_sample + self.config.mmd_loss_weight * feature_mmd
            )
            metrics.update(
                {
                    "mmd_loss": feature_mmd.detach(),
                    "mmd_noise_indices": mmd_indices.detach(),
                }
            )

        loss = total_per_sample.mean()
        denominator = torch.tensor(
            float(resolved.batch_size),
            device=loss.device,
            dtype=torch.float32,
        )
        metrics.update(
            {
                "loss_numerator": loss.detach() * denominator,
                "loss_denominator": denominator,
                "interval_index": resolved.interval_index,
                "scale": self.config.schedule.scale(resolved.interval_index),
            }
        )
        return ScaleWiseLossResult(loss=loss, metrics=metrics)

    def fake_score_loss(
        self,
        batch: ScaleWiseTrainingBatch,
        *,
        generator: object | None = None,
    ) -> ScaleWiseLossResult:
        if not self.config.dmd_enabled:
            raise RuntimeError("fake-score updates are disabled without DMD")
        resolved = self._batch(batch)
        current = self._tensor(resolved.current_latents, field_name="current_latents")
        with torch.no_grad():
            generated = self.sample_student(
                resolved,
                generator=generator,
                training=False,
            ).generated
        indices, sigmas = self._sample_indices(
            resolved.batch_size,
            start=0,
            end=len(self.config.schedule.solver_sigmas) - 1,
            device=generated.device,
            generator=generator,
        )
        noise = self._randn_like(generated, generator=generator)
        noisy_fake = flow_noise(generated, noise, sigmas)
        raw_velocity, raw_fake_features = self.fake_score.predict_velocity_and_features(
            noisy_fake,
            sigmas,
            sample_ids=resolved.sample_ids,
            conditioning=resolved.conditioning,
            block_indices=self.config.classifier_blocks,
            training=True,
        )
        fake_velocity = self._tensor(
            raw_velocity,
            field_name="fake-score velocity",
        )
        fake_features = tuple(
            self._tensor(value, field_name="critic feature")
            for value in raw_fake_features
        )
        if len(fake_features) != len(self.config.classifier_blocks):
            raise RuntimeError(
                "critic did not return one feature tensor per requested block"
            )
        fake_clean = clean_from_velocity(noisy_fake, fake_velocity, sigmas)
        diffusion_per_sample = fake_diffusion_loss_per_sample(
            fake_clean,
            generated,
        )
        total_per_sample = diffusion_per_sample
        metrics: dict[str, object] = {
            "fake_diffusion_loss": diffusion_per_sample.detach().mean(),
            "fake_noise_indices": indices.detach(),
        }
        if self.config.gan_enabled:
            noisy_real = flow_noise(current, noise, sigmas)
            real_features = self._critic_features(
                noisy_real,
                sigmas,
                resolved,
                blocks=self.config.classifier_blocks,
                training=True,
            )
            critic_gan = discriminator_logistic_loss(
                self._classifier_logits(fake_features),
                self._classifier_logits(real_features),
            )
            total_per_sample = (
                total_per_sample + self.config.critic_gan_weight * critic_gan
            )
            metrics["critic_gan_loss"] = critic_gan.detach().mean()
        loss = total_per_sample.mean()
        denominator = torch.tensor(
            float(resolved.batch_size),
            device=loss.device,
            dtype=torch.float32,
        )
        metrics.update(
            {
                "loss_numerator": loss.detach() * denominator,
                "loss_denominator": denominator,
                "interval_index": resolved.interval_index,
                "scale": self.config.schedule.scale(resolved.interval_index),
            }
        )
        return ScaleWiseLossResult(loss=loss, metrics=metrics)


__all__ = [
    "FlowScaleWiseLossAdapter",
    "ScaleWiseLossResult",
    "ScaleWiseStudentSample",
]
