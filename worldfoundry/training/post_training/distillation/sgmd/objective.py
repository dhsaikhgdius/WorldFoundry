"""Model-neutral SGMD rollouts and two-role objectives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .config import SGMDConfig
from .contracts import SGMDPredictionAdapter, SGMDTrainingBatch
from .math import (
    flow_clean_from_velocity,
    flow_interpolate,
    flow_shift_sigmas,
    sgmd_classifier_free_guidance,
    sgmd_diversity_loss_per_sample,
    sgmd_euler_step,
    sgmd_fake_clean_diagnostic_per_sample,
    sgmd_fake_correction_loss_per_sample,
    sgmd_fake_score_flow_loss_per_sample,
    sgmd_normalized_fisher_loss_per_sample,
)


@dataclass(frozen=True, slots=True)
class SGMDRollout:
    clean_latents: torch.Tensor
    initial_noise: torch.Tensor
    target_index: int
    timestep: float
    sigma: float


@dataclass(frozen=True, slots=True)
class SGMDLossResult:
    loss: torch.Tensor
    metrics: Mapping[str, object]


def _levels(reference: torch.Tensor, value: float) -> torch.Tensor:
    return torch.full(
        (int(reference.shape[0]),),
        float(value),
        device=reference.device,
        dtype=reference.dtype,
    )


def _randn_like(reference: torch.Tensor, *, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _score_noise(reference: torch.Tensor, *, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=torch.float32,
        generator=generator,
    )


def _prediction(
    adapter: SGMDPredictionAdapter,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    batch: SGMDTrainingBatch,
    *,
    conditioning: Mapping[str, object],
    training: bool,
    branch: str,
) -> torch.Tensor:
    value = adapter.predict_velocity(
        latents,
        sigmas,
        sample_ids=batch.sample_ids,
        conditioning=conditioning,
        training=training,
        branch=branch,
    )
    if not isinstance(value, torch.Tensor) or value.shape != latents.shape:
        raise ValueError("SGMD velocity prediction must preserve the latent shape")
    if not value.is_floating_point():
        raise TypeError("SGMD velocity prediction must be floating point")
    return value


def simulate_sgmd_student(
    student: SGMDPredictionAdapter,
    batch: SGMDTrainingBatch,
    config: SGMDConfig,
    *,
    target_index: int,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
    training: bool,
    model_training: bool | None = None,
) -> SGMDRollout:
    """Euler-roll a detached prefix and one differentiable selected step."""

    if not isinstance(student, SGMDPredictionAdapter):
        raise TypeError("student must implement SGMDPredictionAdapter")
    if not isinstance(batch, SGMDTrainingBatch):
        raise TypeError("batch must be SGMDTrainingBatch")
    if not isinstance(config, SGMDConfig):
        raise TypeError("config must be SGMDConfig")
    if isinstance(target_index, bool) or not 0 <= int(target_index) < len(config.student_sigmas):
        raise ValueError("target_index is outside the SGMD student schedule")
    template = batch.latent_template
    if not isinstance(template, torch.Tensor) or not template.is_floating_point():
        raise TypeError("latent_template must be a floating torch.Tensor")
    if initial_noise is None:
        initial = _randn_like(template, generator=generator)
    else:
        if not isinstance(initial_noise, torch.Tensor) or initial_noise.shape != template.shape:
            raise ValueError("initial_noise must match latent_template")
        initial = initial_noise.to(device=template.device, dtype=template.dtype)
    current = initial
    sigmas = config.student_sigmas
    selected = int(target_index)
    resolved_model_training = training if model_training is None else bool(model_training)
    clean = None
    for index in range(selected + 1):
        sigma = _levels(current, sigmas[index])
        context = torch.enable_grad() if training and index == selected else torch.no_grad()
        with context:
            velocity = _prediction(
                student,
                current,
                sigma,
                batch,
                conditioning=batch.conditioning,
                training=resolved_model_training,
                branch="positive",
            )
            clean = flow_clean_from_velocity(current, velocity, sigma).to(dtype=current.dtype)
            next_value = sigmas[index + 1] if index + 1 < len(sigmas) else 0.0
            current = sgmd_euler_step(
                current,
                velocity,
                sigma,
                _levels(current, next_value),
            )
    assert clean is not None
    return SGMDRollout(
        clean_latents=clean,
        initial_noise=initial,
        target_index=selected,
        timestep=config.student_timesteps[selected],
        sigma=sigmas[selected],
    )


def sample_sgmd_score_sigmas(
    reference: torch.Tensor,
    config: SGMDConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample discrete score times, shift them, then apply released bounds."""

    if not isinstance(reference, torch.Tensor) or reference.ndim < 2:
        raise TypeError("reference must be a [B,...] torch.Tensor")
    indices = torch.randint(
        0,
        config.score_discrete_samples,
        (int(reference.shape[0]),),
        device=reference.device,
        generator=generator,
        dtype=torch.int64,
    )
    raw = indices.float() / float(config.score_discrete_samples)
    shifted = flow_shift_sigmas(raw, config.score_flow_shift).clamp(
        min=config.score_min_sigma,
        max=config.score_max_sigma,
    )
    return shifted.to(dtype=reference.dtype)


def _teacher_velocity(
    teacher: SGMDPredictionAdapter,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    batch: SGMDTrainingBatch,
    config: SGMDConfig,
) -> torch.Tensor:
    with torch.no_grad():
        conditional = _prediction(
            teacher,
            latents,
            sigmas,
            batch,
            conditioning=batch.conditioning,
            training=False,
            branch="positive",
        )
        if config.teacher_guidance_scale <= 1.0:
            return conditional
        unconditional = _prediction(
            teacher,
            latents,
            sigmas,
            batch,
            conditioning=batch.unconditional_conditioning,
            training=False,
            branch="negative",
        )
        return sgmd_classifier_free_guidance(
            unconditional,
            conditional,
            config.teacher_guidance_scale,
        )


def _sample_weights(batch: SGMDTrainingBatch, *, device: torch.device) -> torch.Tensor:
    if batch.sample_weights is None:
        return torch.ones((batch.batch_size,), device=device, dtype=torch.float32)
    if not isinstance(batch.sample_weights, torch.Tensor):
        raise TypeError("sample_weights must be a torch.Tensor")
    weights = batch.sample_weights.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
        raise ValueError("sample_weights must be finite and non-negative")
    if not bool(weights.sum() > 0):
        raise ValueError("SGMD batch must contain positive sample weight")
    return weights


def _weighted_mean(
    per_sample: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if per_sample.ndim != 1 or per_sample.shape != weights.shape:
        raise ValueError("SGMD losses must return exactly one value per sample")
    numerator = (per_sample.float() * weights).sum()
    denominator = weights.sum()
    return numerator / denominator, numerator, denominator


def _score_context(
    generated: torch.Tensor,
    batch: SGMDTrainingBatch,
    config: SGMDConfig,
    *,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigmas = sample_sgmd_score_sigmas(generated, config, generator=generator)
    noise = _score_noise(generated, generator=generator)
    noisy = flow_interpolate(generated, noise, sigmas).to(dtype=generated.dtype)
    return noisy, noise, sigmas


def _diversity_per_sample(
    student: SGMDPredictionAdapter,
    teacher: SGMDPredictionAdapter,
    batch: SGMDTrainingBatch,
    config: SGMDConfig,
    initial_noise: torch.Tensor,
) -> torch.Tensor:
    if not config.diversity_enabled:
        return torch.zeros(
            (batch.batch_size,),
            device=initial_noise.device,
            dtype=torch.float32,
        )
    teacher_sigmas = config.teacher_sigmas
    anchor = initial_noise.detach()
    with torch.no_grad():
        for index in range(config.diversity_anchor_step):
            sigma = _levels(anchor, teacher_sigmas[index])
            velocity = _teacher_velocity(teacher, anchor, sigma, batch, config)
            anchor = sgmd_euler_step(
                anchor,
                velocity,
                sigma,
                _levels(anchor, teacher_sigmas[index + 1]),
            )
    first_sigma = _levels(initial_noise, config.student_sigmas[0])
    first_velocity = _prediction(
        student,
        initial_noise.detach(),
        first_sigma,
        batch,
        conditioning=batch.conditioning,
        training=True,
        branch="positive",
    )
    anchor_sigmas = _levels(initial_noise, teacher_sigmas[config.diversity_anchor_step])
    return sgmd_diversity_loss_per_sample(
        initial_noise.detach(),
        anchor.detach(),
        anchor_sigmas,
        first_velocity,
        epsilon=config.numerical_epsilon,
    )


class NativeSGMDLossAdapter:
    """Released SGMD Fisher, correction, diversity, and fake-flow objectives."""

    def __init__(
        self,
        student: SGMDPredictionAdapter,
        teacher: SGMDPredictionAdapter,
        fake_score: SGMDPredictionAdapter,
        config: SGMDConfig,
    ) -> None:
        if not all(
            isinstance(adapter, SGMDPredictionAdapter)
            for adapter in (student, teacher, fake_score)
        ):
            raise TypeError("all SGMD roles must implement SGMDPredictionAdapter")
        if not isinstance(config, SGMDConfig):
            raise TypeError("config must be SGMDConfig")
        self.student = student
        self.teacher = teacher
        self.fake_score = fake_score
        self.config = config
        self.num_student_steps = len(config.student_sigmas)
        self.minimum_student_target_index = config.minimum_student_target_index

    def loss_denominator(
        self,
        batch: SGMDTrainingBatch,
        *,
        role: str,
    ) -> torch.Tensor:
        if role not in {"student", "fake-score"}:
            raise ValueError(f"unsupported SGMD loss role: {role!r}")
        if not isinstance(batch, SGMDTrainingBatch):
            raise TypeError("batch must be SGMDTrainingBatch")
        template = batch.latent_template
        if not isinstance(template, torch.Tensor):
            raise TypeError("latent_template must be a torch.Tensor")
        return _sample_weights(batch, device=template.device).sum()

    def student_loss(
        self,
        batch: SGMDTrainingBatch,
        *,
        target_index: int,
        generator: torch.Generator | None = None,
    ) -> SGMDLossResult:
        rollout = simulate_sgmd_student(
            self.student,
            batch,
            self.config,
            target_index=target_index,
            generator=generator,
            training=True,
        )
        generated = rollout.clean_latents
        if not generated.requires_grad:
            raise RuntimeError("SGMD student rollout did not retain a differentiable selected step")
        noisy, _, sigmas = _score_context(
            generated,
            batch,
            self.config,
            generator=generator,
        )
        fake_velocity = _prediction(
            self.fake_score,
            noisy,
            sigmas,
            batch,
            conditioning=batch.conditioning,
            training=False,
            branch="positive",
        )
        fake_clean = flow_clean_from_velocity(noisy, fake_velocity, sigmas)
        teacher_velocity = _teacher_velocity(
            self.teacher,
            noisy,
            sigmas,
            batch,
            self.config,
        )
        with torch.no_grad():
            teacher_clean = flow_clean_from_velocity(noisy, teacher_velocity, sigmas)
        fisher, normalizer = sgmd_normalized_fisher_loss_per_sample(
            generated,
            fake_clean,
            teacher_clean,
            epsilon=self.config.numerical_epsilon,
        )
        correction = sgmd_fake_correction_loss_per_sample(
            generated,
            fake_clean,
            sigmas,
            epsilon=self.config.numerical_epsilon,
        )
        main_per_sample = fisher - self.config.fake_correction_weight * correction
        weights = _sample_weights(batch, device=generated.device)
        main_mean, main_numerator, denominator = _weighted_mean(main_per_sample, weights)

        # Preserve the exact first-order input Jacobian of the live fake score
        # while preventing its parameters from entering the student optimizer
        # backward. This is equivalent to the released backward-then-zero path.
        input_gradient = torch.autograd.grad(
            main_mean,
            generated,
            create_graph=False,
            retain_graph=False,
        )[0].detach()
        if not bool(torch.isfinite(input_gradient).all()):
            raise FloatingPointError("SGMD fake-score input gradient is non-finite")
        proxy = (generated.float() * input_gradient.float()).sum()
        proxy = proxy + (main_mean.detach() - proxy.detach())

        diversity = _diversity_per_sample(
            self.student,
            self.teacher,
            batch,
            self.config,
            rollout.initial_noise,
        )
        diversity_mean, diversity_numerator, _ = _weighted_mean(diversity, weights)
        loss = proxy + self.config.diversity_weight * diversity_mean
        total_numerator = main_numerator + self.config.diversity_weight * diversity_numerator
        fisher_mean, _, _ = _weighted_mean(fisher.detach(), weights)
        correction_mean, _, _ = _weighted_mean(correction.detach(), weights)
        normalizer_mean = normalizer.detach().float().flatten(1).mean(1)
        normalizer_mean, _, _ = _weighted_mean(normalizer_mean, weights)
        return SGMDLossResult(
            loss=loss,
            metrics={
                "loss_numerator": total_numerator.detach(),
                "loss_denominator": denominator.detach(),
                "fisher": fisher_mean.detach(),
                "fake_correction": correction_mean.detach(),
                "diversity": diversity_mean.detach(),
                "normalizer": normalizer_mean.detach(),
                "target_index": int(target_index),
                "sgmd_step": int(target_index) + 1,
                "score_sigma_mean": sigmas.detach().float().mean(),
            },
        )

    def fake_score_loss(
        self,
        batch: SGMDTrainingBatch,
        *,
        target_index: int,
        generator: torch.Generator | None = None,
    ) -> SGMDLossResult:
        with torch.no_grad():
            rollout = simulate_sgmd_student(
                self.student,
                batch,
                self.config,
                target_index=target_index,
                generator=generator,
                training=False,
                model_training=True,
            )
            generated = rollout.clean_latents.detach()
            noisy, noise, sigmas = _score_context(
                generated,
                batch,
                self.config,
                generator=generator,
            )
            teacher_velocity = _teacher_velocity(
                self.teacher,
                noisy,
                sigmas,
                batch,
                self.config,
            )
            teacher_clean = flow_clean_from_velocity(noisy, teacher_velocity, sigmas)
            axes = tuple(range(1, generated.ndim))
            normalizer = (
                generated.float() - teacher_clean.float()
            ).abs().mean(dim=axes, keepdim=True)
        fake_velocity = _prediction(
            self.fake_score,
            noisy,
            sigmas,
            batch,
            conditioning=batch.conditioning,
            training=False,
            branch="positive",
        )
        fake_clean = flow_clean_from_velocity(noisy, fake_velocity, sigmas)
        flow_per_sample = sgmd_fake_score_flow_loss_per_sample(
            fake_velocity,
            generated,
            noise,
        )
        diagnostic = sgmd_fake_clean_diagnostic_per_sample(
            generated,
            fake_clean,
            normalizer,
            epsilon=self.config.numerical_epsilon,
        )
        weights = _sample_weights(batch, device=generated.device)
        loss, numerator, denominator = _weighted_mean(flow_per_sample, weights)
        diagnostic_mean, _, _ = _weighted_mean(diagnostic.detach(), weights)
        return SGMDLossResult(
            loss=loss,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
                "velocity": loss.detach(),
                "fake_clean_diagnostic": diagnostic_mean.detach(),
                "target_index": int(target_index),
                "score_sigma_mean": sigmas.detach().float().mean(),
            },
        )


__all__ = [
    "NativeSGMDLossAdapter",
    "SGMDLossResult",
    "SGMDRollout",
    "sample_sgmd_score_sigmas",
    "simulate_sgmd_student",
]
