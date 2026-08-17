"""Model-neutral Data-Forcing Distillation objectives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ..dmd2.math import (
    dmd2_generator_adversarial_loss,
    dmd2_guidance_adversarial_loss,
)
from .config import DFDConfig
from .contracts import (
    DFDDiscriminatorAdapter,
    DFDFakeScoreAdapter,
    DFDPredictionAdapter,
    DFDTrainingBatch,
)
from .math import (
    data_forcing_teacher_data,
    dfd_distribution_gradient,
    dfd_proxy_loss_per_sample,
    shifted_uniform_timesteps,
)


@dataclass(frozen=True, slots=True)
class DFDStudentPrediction:
    clean_latents: torch.Tensor
    input_latents: torch.Tensor
    input_noise: torch.Tensor
    timesteps: torch.Tensor
    timestep_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class DFDLossResult:
    loss: torch.Tensor
    metrics: Mapping[str, object]


def _randn_like(
    reference: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _prediction(
    adapter: DFDPredictionAdapter,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    batch: DFDTrainingBatch,
    *,
    conditioning: Mapping[str, object],
    training: bool,
    branch: str,
) -> torch.Tensor:
    value = adapter.predict_clean(
        noisy,
        timesteps,
        sample_ids=batch.sample_ids,
        conditioning=conditioning,
        training=training,
        branch=branch,
    )
    if not isinstance(value, torch.Tensor) or value.shape != noisy.shape:
        raise ValueError("DFD clean prediction must preserve the latent shape")
    if not value.is_floating_point():
        raise TypeError("DFD clean prediction must be floating point")
    return value


def sample_dfd_student_timesteps(
    reference: torch.Tensor,
    config: DFDConfig,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly select a non-clean entry from the released student schedule."""

    if not isinstance(reference, torch.Tensor) or reference.ndim < 2:
        raise TypeError("reference must be a [B,...] tensor")
    indices = torch.randint(
        0,
        len(config.trainable_student_timesteps),
        (int(reference.shape[0]),),
        device=reference.device,
        generator=generator,
        dtype=torch.int64,
    )
    schedule = torch.tensor(
        config.trainable_student_timesteps,
        device=reference.device,
        dtype=torch.float64,
    )
    return schedule[indices], indices


def sample_dfd_score_timesteps(
    reference: torch.Tensor,
    config: DFDConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample shifted RF score times with FastGen's float64 scheduler precision."""

    if not isinstance(reference, torch.Tensor) or reference.ndim < 2:
        raise TypeError("reference must be a [B,...] tensor")
    uniform = torch.rand(
        (int(reference.shape[0]),),
        device=reference.device,
        dtype=torch.float64,
        generator=generator,
    )
    return shifted_uniform_timesteps(
        uniform,
        minimum=config.score_min_timestep,
        maximum=config.score_max_timestep,
        shift=config.score_timestep_shift,
    )


def prepare_dfd_student_prediction(
    student: DFDPredictionAdapter,
    batch: DFDTrainingBatch,
    config: DFDConfig,
    *,
    generator: torch.Generator | None = None,
    training: bool,
) -> DFDStudentPrediction:
    """Noise paired real data at one schedule entry, then make one x0 prediction."""

    if not isinstance(student, DFDPredictionAdapter):
        raise TypeError("student must implement DFDPredictionAdapter")
    if not isinstance(batch, DFDTrainingBatch):
        raise TypeError("batch must be DFDTrainingBatch")
    real = batch.real_latents
    if not isinstance(real, torch.Tensor) or not real.is_floating_point():
        raise TypeError("real_latents must be a floating torch.Tensor")
    timesteps, indices = sample_dfd_student_timesteps(
        real,
        config,
        generator=generator,
    )
    noise = _randn_like(real, generator=generator)
    input_latents = student.add_noise(real, noise, timesteps)
    if not isinstance(input_latents, torch.Tensor) or input_latents.shape != real.shape:
        raise ValueError("DFD add_noise must preserve the latent shape")
    clean = _prediction(
        student,
        input_latents,
        timesteps,
        batch,
        conditioning=batch.conditioning,
        training=training,
        branch="positive",
    )
    return DFDStudentPrediction(
        clean_latents=clean,
        input_latents=input_latents,
        input_noise=noise,
        timesteps=timesteps,
        timestep_indices=indices,
    )


def dfd_teacher_guidance(
    conditional_clean: torch.Tensor,
    unconditional_clean: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if conditional_clean.shape != unconditional_clean.shape:
        raise ValueError("conditional and unconditional teacher predictions must match")
    return conditional_clean + float(scale - 1.0) * (
        conditional_clean - unconditional_clean
    )


def _sample_weights(batch: DFDTrainingBatch, *, device: torch.device) -> torch.Tensor:
    if batch.sample_weights is None:
        return torch.ones((batch.batch_size,), device=device, dtype=torch.float32)
    if not isinstance(batch.sample_weights, torch.Tensor):
        raise TypeError("sample_weights must be a torch.Tensor")
    weights = batch.sample_weights.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
        raise ValueError("sample_weights must be finite and non-negative")
    if not bool(weights.sum() > 0):
        raise ValueError("DFD batch must contain positive sample weight")
    return weights


def _weighted_mean(
    per_sample: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if per_sample.ndim != 1 or per_sample.shape != weights.shape:
        raise ValueError("DFD losses must return exactly one value per sample")
    numerator = (per_sample.float() * weights).sum()
    denominator = weights.sum()
    return numerator / denominator, numerator, denominator


def _discriminator_logits(
    discriminator: DFDDiscriminatorAdapter,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    batch: DFDTrainingBatch,
    *,
    training: bool,
) -> torch.Tensor:
    logits = discriminator.discriminator_logits(
        noisy,
        timesteps,
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
        training=training,
    )
    if not isinstance(logits, torch.Tensor) or logits.shape[0] != batch.batch_size:
        raise ValueError("DFD discriminator logits must preserve the batch dimension")
    return logits


class NativeDFDLossAdapter:
    """DFD's condition-matched teacher discrepancy and inherited DMD2 auxiliaries."""

    def __init__(
        self,
        student: DFDPredictionAdapter,
        teacher: DFDPredictionAdapter,
        fake_score: DFDFakeScoreAdapter,
        config: DFDConfig,
        *,
        discriminator: DFDDiscriminatorAdapter | None = None,
    ) -> None:
        if not isinstance(student, DFDPredictionAdapter):
            raise TypeError("student must implement DFDPredictionAdapter")
        if not isinstance(teacher, DFDPredictionAdapter):
            raise TypeError("teacher must implement DFDPredictionAdapter")
        if not isinstance(fake_score, DFDFakeScoreAdapter):
            raise TypeError("fake_score must implement DFDFakeScoreAdapter")
        if not isinstance(config, DFDConfig):
            raise TypeError("config must be DFDConfig")
        if config.adversarial_enabled and not isinstance(
            discriminator,
            DFDDiscriminatorAdapter,
        ):
            raise ValueError("released adversarial DFD requires a discriminator adapter")
        if not config.adversarial_enabled and discriminator is not None:
            raise ValueError("a DFD discriminator cannot be supplied when adversarial losses are disabled")
        adapters = (student, teacher, fake_score)
        kinds = {
            str(adapter.noise_process_kind).strip().lower().replace("_", "-")
            for adapter in adapters
        }
        if kinds != {"flow-matching"}:
            raise ValueError("DFD requires flow-matching prediction adapters")
        self.student = student
        self.teacher = teacher
        self.fake_score = fake_score
        self.discriminator = discriminator
        self.config = config
        self.data_forcing_probability = config.data_forcing_probability
        self.student_update_frequency = config.student_update_frequency

    def loss_denominator(self, batch: DFDTrainingBatch, *, role: str) -> torch.Tensor:
        if role not in {"student", "guidance"}:
            raise ValueError(f"unsupported DFD loss role: {role!r}")
        if not isinstance(batch, DFDTrainingBatch):
            raise TypeError("batch must be DFDTrainingBatch")
        if not isinstance(batch.real_latents, torch.Tensor):
            raise TypeError("real_latents must be a torch.Tensor")
        return _sample_weights(batch, device=batch.real_latents.device).sum()

    def student_loss(
        self,
        batch: DFDTrainingBatch,
        *,
        data_forcing: bool,
        generator: torch.Generator | None = None,
    ) -> DFDLossResult:
        prediction = prepare_dfd_student_prediction(
            self.student,
            batch,
            self.config,
            generator=generator,
            training=True,
        )
        generated = prediction.clean_latents
        if not generated.requires_grad:
            raise RuntimeError("DFD student prediction must retain gradients")
        score_timesteps = sample_dfd_score_timesteps(
            generated,
            self.config,
            generator=generator,
        )
        score_noise = _randn_like(generated, generator=generator)
        generated_noisy = self.student.add_noise(generated, score_noise, score_timesteps)
        teacher_data = data_forcing_teacher_data(
            generated,
            batch.real_latents,
            enabled=data_forcing,
        )
        teacher_noisy = self.student.add_noise(
            teacher_data,
            score_noise,
            score_timesteps,
        )
        with torch.no_grad():
            fake_clean = _prediction(
                self.fake_score,
                generated_noisy,
                score_timesteps,
                batch,
                conditioning=batch.conditioning,
                training=False,
                branch="positive",
            )
            teacher_conditional = _prediction(
                self.teacher,
                teacher_noisy,
                score_timesteps,
                batch,
                conditioning=batch.conditioning,
                training=False,
                branch="positive",
            )
            teacher_unconditional = _prediction(
                self.teacher,
                teacher_noisy,
                score_timesteps,
                batch,
                conditioning=batch.unconditional_conditioning,
                training=False,
                branch="negative",
            )
            teacher_clean = dfd_teacher_guidance(
                teacher_conditional,
                teacher_unconditional,
                self.config.teacher_guidance_scale,
            )
            gradient, normalizer = dfd_distribution_gradient(
                generated,
                fake_clean,
                teacher_clean,
                epsilon=self.config.normalization_epsilon,
            )
        distribution = dfd_proxy_loss_per_sample(generated, gradient)
        zero = torch.zeros_like(distribution)
        adversarial = zero
        if self.discriminator is not None:
            fake_logits = _discriminator_logits(
                self.discriminator,
                generated_noisy,
                score_timesteps,
                batch,
                training=False,
            )
            adversarial = dmd2_generator_adversarial_loss(fake_logits)
        combined = (
            self.config.distribution_matching_weight * distribution
            + self.config.generator_adversarial_weight * adversarial
        )
        weights = _sample_weights(batch, device=generated.device)
        actual_loss, numerator, denominator = _weighted_mean(combined, weights)
        distribution_loss = (
            _weighted_mean(distribution, weights)[0]
            * self.config.distribution_matching_weight
        )
        if self.discriminator is None:
            adversarial_proxy = torch.zeros((), device=generated.device, dtype=torch.float32)
        else:
            adversarial_loss = (
                _weighted_mean(adversarial, weights)[0]
                * self.config.generator_adversarial_weight
            )
            input_gradient = torch.autograd.grad(
                adversarial_loss,
                generated,
                create_graph=False,
                retain_graph=False,
            )[0].detach()
            if not bool(torch.isfinite(input_gradient).all()):
                raise FloatingPointError("DFD adversarial input gradient is non-finite")
            adversarial_proxy = (generated.float() * input_gradient.float()).sum()
        loss = (
            actual_loss.detach()
            + distribution_loss
            - distribution_loss.detach()
            + adversarial_proxy
            - adversarial_proxy.detach()
        )
        return DFDLossResult(
            loss=loss,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
                "distribution_matching": _weighted_mean(distribution, weights)[0].detach(),
                "generator_adversarial": _weighted_mean(adversarial, weights)[0].detach(),
                "normalizer_mean": normalizer.detach().mean(),
                "score_timestep_mean": score_timesteps.detach().mean(),
                "student_timestep_mean": prediction.timesteps.detach().mean(),
                "data_forcing": bool(data_forcing),
            },
        )

    def guidance_loss(
        self,
        batch: DFDTrainingBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> DFDLossResult:
        with torch.no_grad():
            prediction = prepare_dfd_student_prediction(
                self.student,
                batch,
                self.config,
                generator=generator,
                training=True,
            )
            generated = prediction.clean_latents.detach()
        score_timesteps = sample_dfd_score_timesteps(
            generated,
            self.config,
            generator=generator,
        )
        score_noise = _randn_like(generated, generator=generator)
        generated_noisy = self.fake_score.add_noise(
            generated,
            score_noise,
            score_timesteps,
        )
        denoising = self.fake_score.denoising_loss_per_sample(
            generated,
            generated_noisy,
            score_noise,
            score_timesteps,
            sample_ids=batch.sample_ids,
            conditioning=batch.conditioning,
            training=True,
        )
        if not isinstance(denoising, torch.Tensor) or denoising.shape != (batch.batch_size,):
            raise ValueError("DFD fake-score denoising seam must return shape [B]")
        adversarial = torch.zeros_like(denoising, dtype=torch.float32)
        if self.discriminator is not None:
            real_noisy = self.fake_score.add_noise(
                batch.real_latents,
                score_noise,
                score_timesteps,
            )
            fake_logits = _discriminator_logits(
                self.discriminator,
                generated_noisy,
                score_timesteps,
                batch,
                training=True,
            )
            real_logits = _discriminator_logits(
                self.discriminator,
                real_noisy,
                score_timesteps,
                batch,
                training=True,
            )
            adversarial = dmd2_guidance_adversarial_loss(real_logits, fake_logits)
        combined = (
            self.config.fake_score_denoising_weight * denoising.float()
            + self.config.discriminator_weight * adversarial.float()
        )
        weights = _sample_weights(batch, device=generated.device)
        loss, numerator, denominator = _weighted_mean(combined, weights)
        return DFDLossResult(
            loss=loss,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
                "fake_score_denoising": _weighted_mean(denoising, weights)[0].detach(),
                "discriminator": _weighted_mean(adversarial, weights)[0].detach(),
                "score_timestep_mean": score_timesteps.detach().mean(),
                "student_timestep_mean": prediction.timesteps.detach().mean(),
            },
        )


__all__ = [
    "DFDLossResult",
    "DFDStudentPrediction",
    "NativeDFDLossAdapter",
    "dfd_teacher_guidance",
    "prepare_dfd_student_prediction",
    "sample_dfd_score_timesteps",
    "sample_dfd_student_timesteps",
]
