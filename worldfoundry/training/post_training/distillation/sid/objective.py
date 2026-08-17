"""Native model-neutral Score Identity Distillation objective."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from ....recipes.post_training.algorithms.sid import SID_WEIGHTING_SCHEMES, SIDAlgorithmSpec
from ..dmd.objective import FewStepSchedule
from .contracts import (
    SIDDiscriminatorAdapter,
    SIDPredictionAdapter,
    SIDTrainingBatch,
)
from .math import (
    sid_classifier_free_guidance,
    sid_fake_score_adversarial_loss_per_sample,
    sid_fake_score_flow_loss_per_sample,
    sid_generator_adversarial_loss_per_sample,
    sid_generator_loss_per_sample,
    sid_score_weight,
)


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("native SiD requires the 'train-core' extra") from error
    return torch


@dataclass(frozen=True, slots=True)
class SIDConfig:
    schedule: FewStepSchedule
    alpha: float
    noise_policy: str = "fresh"
    score_weighting: str = "1-minus-sigma"
    num_train_timesteps: int = 1000
    score_logit_mean: float = 0.6931471805599453
    score_logit_std: float = 1.6
    weighting_epsilon: float = 1.0e-5
    teacher_guidance_scale: float = 4.5
    fake_score_guidance_scale: float = 4.5
    score_identity_weight: float = 100.0
    fake_score_flow_weight: float = 1.0
    generator_adversarial_weight: float = 0.0
    fake_score_adversarial_weight: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, FewStepSchedule):
            raise TypeError("schedule must be FewStepSchedule")
        policy = str(self.noise_policy).strip().lower().replace("_", "-")
        if policy not in {"fresh", "fixed", "ddim"}:
            raise ValueError("noise_policy must be fresh, fixed, or ddim")
        weighting = str(self.score_weighting).strip().lower().replace("_", "-")
        if weighting not in SID_WEIGHTING_SCHEMES:
            raise ValueError(f"unsupported SiD score weighting: {weighting!r}")
        if isinstance(self.num_train_timesteps, bool) or int(self.num_train_timesteps) < 2:
            raise ValueError("num_train_timesteps must be an integer >= 2")
        values = {
            name: float(getattr(self, name))
            for name in (
                "alpha",
                "score_logit_mean",
                "score_logit_std",
                "weighting_epsilon",
                "teacher_guidance_scale",
                "fake_score_guidance_scale",
                "score_identity_weight",
                "fake_score_flow_weight",
                "generator_adversarial_weight",
                "fake_score_adversarial_weight",
            )
        }
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("SiD scalar configuration must be finite")
        if values["score_logit_std"] <= 0 or values["weighting_epsilon"] <= 0:
            raise ValueError("score_logit_std and weighting_epsilon must be positive")
        for name in (
            "teacher_guidance_scale",
            "fake_score_guidance_scale",
            "score_identity_weight",
            "fake_score_flow_weight",
            "generator_adversarial_weight",
            "fake_score_adversarial_weight",
        ):
            if values[name] < 0:
                raise ValueError(f"{name} must be non-negative")
        if values["score_identity_weight"] <= 0 or values["fake_score_flow_weight"] <= 0:
            raise ValueError("SiD score-identity and fake-score flow weights must be positive")
        if (values["generator_adversarial_weight"] == 0) != (
            values["fake_score_adversarial_weight"] == 0
        ):
            raise ValueError("SiD DiffusionGAN weights must be enabled together")
        object.__setattr__(self, "noise_policy", policy)
        object.__setattr__(self, "score_weighting", weighting)
        object.__setattr__(self, "num_train_timesteps", int(self.num_train_timesteps))
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_recipe(cls, spec: SIDAlgorithmSpec) -> SIDConfig:
        if not isinstance(spec, SIDAlgorithmSpec):
            raise TypeError("spec must be SIDAlgorithmSpec")
        return cls(
            schedule=FewStepSchedule(spec.student_timesteps, spec.student_sigmas),
            alpha=spec.alpha,
            noise_policy=spec.noise_policy,
            score_weighting=spec.score_weighting,
            num_train_timesteps=spec.num_train_timesteps,
            score_logit_mean=spec.score_logit_mean,
            score_logit_std=spec.score_logit_std,
            weighting_epsilon=spec.weighting_epsilon,
            teacher_guidance_scale=spec.teacher_guidance_scale,
            fake_score_guidance_scale=spec.fake_score_guidance_scale,
            score_identity_weight=spec.score_identity_weight,
            fake_score_flow_weight=spec.fake_score_flow_weight,
            generator_adversarial_weight=spec.generator_adversarial_weight,
            fake_score_adversarial_weight=spec.fake_score_adversarial_weight,
        )

    @property
    def diffusion_gan_enabled(self) -> bool:
        return self.generator_adversarial_weight > 0


@dataclass(frozen=True, slots=True)
class SIDFewStepPrediction:
    clean_latents: object
    target_index: int
    timestep: float
    sigma: float


@dataclass(frozen=True, slots=True)
class SIDLossResult:
    loss: object
    metrics: Mapping[str, object]


def _randn_like(reference: object, *, generator: object | None) -> object:
    torch = _require_torch()
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _levels(reference: object, sigma: float) -> object:
    torch = _require_torch()
    return torch.full(
        (int(reference.shape[0]),),
        float(sigma),
        device=reference.device,
        dtype=torch.float32,
    )


def simulate_sid_student(
    student: SIDPredictionAdapter,
    batch: SIDTrainingBatch,
    schedule: FewStepSchedule,
    *,
    target_index: int,
    noise_policy: str,
    generator: object | None,
    training: bool,
) -> SIDFewStepPrediction:
    """Run a detached prefix and exactly one differentiable selected step."""

    torch = _require_torch()
    if not isinstance(student, SIDPredictionAdapter):
        raise TypeError("student must implement SIDPredictionAdapter")
    if not isinstance(batch, SIDTrainingBatch):
        raise TypeError("batch must be SIDTrainingBatch")
    if isinstance(target_index, bool) or not 0 <= int(target_index) < len(schedule.sigmas):
        raise ValueError("target_index is outside the SiD student schedule")
    policy = str(noise_policy).strip().lower().replace("_", "-")
    if policy not in {"fresh", "fixed", "ddim"}:
        raise ValueError("noise_policy must be fresh, fixed, or ddim")
    template = batch.latent_template
    if not torch.is_tensor(template) or not template.is_floating_point():
        raise TypeError("latent_template must be a floating torch.Tensor")
    initial_noise = _randn_like(template, generator=generator)
    previous_clean = torch.zeros_like(template)
    previous_input = None
    previous_sigma = None

    for index in range(int(target_index) + 1):
        sigma = float(schedule.sigmas[index])
        if index == 0 or policy == "fixed":
            noise = initial_noise
        elif policy == "fresh":
            noise = _randn_like(template, generator=generator)
        else:
            assert previous_input is not None and previous_sigma is not None
            noise = (
                (previous_input - (1.0 - previous_sigma) * previous_clean)
                / previous_sigma
            ).detach()
        sigmas = _levels(template, sigma)
        model_input = student.add_noise(previous_clean.detach(), noise, sigmas)
        if index < int(target_index):
            with torch.no_grad():
                previous_clean = student.predict_clean(
                    model_input,
                    sigmas,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    training=training,
                ).detach()
            previous_input = model_input.detach()
            previous_sigma = sigma
            continue
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            clean = student.predict_clean(
                model_input,
                sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=training,
            )
        return SIDFewStepPrediction(
            clean_latents=clean,
            target_index=int(target_index),
            timestep=float(schedule.timesteps[index]),
            sigma=sigma,
        )
    raise AssertionError("unreachable SiD schedule state")


def sample_sid_score_sigmas(
    reference: object,
    config: SIDConfig,
    *,
    generator: object | None,
) -> object:
    torch = _require_torch()
    if not torch.is_tensor(reference) or reference.ndim < 2:
        raise TypeError("reference must be a [B,...] tensor")
    normal = torch.randn(
        (int(reference.shape[0]),),
        device=reference.device,
        dtype=torch.float32,
        generator=generator,
    )
    return torch.sigmoid(normal * config.score_logit_std + config.score_logit_mean)


def _sample_weights(batch: SIDTrainingBatch, *, device: object) -> object:
    torch = _require_torch()
    if batch.sample_weights is None:
        return torch.ones((batch.batch_size,), device=device, dtype=torch.float32)
    if not torch.is_tensor(batch.sample_weights):
        raise TypeError("sample_weights must be a tensor")
    weights = batch.sample_weights.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
        raise ValueError("sample_weights must be finite and non-negative")
    if not bool(weights.sum() > 0):
        raise ValueError("SiD batch must contain positive sample weight")
    return weights


def _weighted_mean(per_sample: object, weights: object) -> tuple[object, object, object]:
    if per_sample.ndim != 1 or per_sample.shape != weights.shape:
        raise ValueError("SiD losses must return exactly one value per sample")
    numerator = (per_sample.float() * weights).sum()
    denominator = weights.sum()
    return numerator / denominator, numerator, denominator


def _cfg_clean(
    adapter: SIDPredictionAdapter,
    noisy: object,
    sigmas: object,
    batch: SIDTrainingBatch,
    *,
    scale: float,
    training: bool,
) -> object:
    unconditional = adapter.predict_clean(
        noisy,
        sigmas,
        sample_ids=batch.sample_ids,
        conditioning=batch.unconditional_conditioning,
        training=training,
        branch="negative",
    )
    conditional = adapter.predict_clean(
        noisy,
        sigmas,
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
        training=training,
        branch="positive",
    )
    return sid_classifier_free_guidance(unconditional, conditional, scale)


def _cfg_velocity(
    adapter: SIDPredictionAdapter,
    noisy: object,
    sigmas: object,
    batch: SIDTrainingBatch,
    *,
    scale: float,
    training: bool,
) -> object:
    unconditional = adapter.predict_velocity(
        noisy,
        sigmas,
        sample_ids=batch.sample_ids,
        conditioning=batch.unconditional_conditioning,
        training=training,
        branch="negative",
    )
    conditional = adapter.predict_velocity(
        noisy,
        sigmas,
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
        training=training,
        branch="positive",
    )
    return sid_classifier_free_guidance(unconditional, conditional, scale)


class NativeSIDLossAdapter:
    """Own the fake-score flow update and score-identity generator update."""

    def __init__(
        self,
        student: SIDPredictionAdapter,
        teacher: SIDPredictionAdapter,
        fake_score: SIDPredictionAdapter,
        config: SIDConfig,
    ) -> None:
        torch = _require_torch()
        if not all(isinstance(value, SIDPredictionAdapter) for value in (student, teacher, fake_score)):
            raise TypeError("SiD roles must implement SIDPredictionAdapter")
        modules = (student.module, teacher.module, fake_score.module)
        if not all(isinstance(module, torch.nn.Module) for module in modules):
            raise TypeError("SiD adapters must expose nn.Module instances")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("SiD student, teacher, and fake-score modules must be distinct")
        kinds = tuple(
            str(adapter.noise_process_kind).strip().lower()
            for adapter in (student, teacher, fake_score)
        )
        if set(kinds) != {"flow-matching"}:
            raise ValueError("native SiD adapters must use the flow-matching noise process")
        if not isinstance(config, SIDConfig):
            raise TypeError("config must be SIDConfig")
        if config.diffusion_gan_enabled and not isinstance(fake_score, SIDDiscriminatorAdapter):
            raise TypeError("SiD DiffusionGAN requires a discriminator-capable fake-score adapter")
        self.student = student
        self.teacher = teacher
        self.fake_score = fake_score
        self.config = config

    @property
    def num_student_steps(self) -> int:
        return len(self.config.schedule.sigmas)

    def loss_denominator(self, batch: SIDTrainingBatch, *, role: str) -> object:
        if role not in {"fake-score", "generator"}:
            raise ValueError(f"unsupported SiD loss role: {role!r}")
        return _sample_weights(batch, device=batch.latent_template.device).sum()

    def _result(
        self,
        batch: SIDTrainingBatch,
        components: Mapping[str, object],
        component_weights: Mapping[str, float],
        *,
        target_index: int,
        score_sigmas: object,
    ) -> SIDLossResult:
        torch = _require_torch()
        if set(components) != set(component_weights) or not components:
            raise ValueError("SiD loss components and weights must have equal non-empty keys")
        per_sample = torch.zeros_like(next(iter(components.values())), dtype=torch.float32)
        for name, value in components.items():
            if value.ndim != 1 or value.shape != (batch.batch_size,):
                raise ValueError(f"SiD {name} must return shape [B]")
            per_sample = per_sample + value.float() * float(component_weights[name])
        weights = _sample_weights(batch, device=per_sample.device)
        loss, numerator, denominator = _weighted_mean(per_sample, weights)
        metrics: dict[str, object] = {
            "loss_numerator": numerator.detach(),
            "loss_denominator": denominator.detach(),
            "target_index": int(target_index),
            "score_sigma_mean": score_sigmas.detach().float().mean(),
        }
        for name, value in components.items():
            component_loss, _, _ = _weighted_mean(value, weights)
            metrics[name] = component_loss.detach()
        return SIDLossResult(loss=loss, metrics=metrics)

    def fake_score_loss(
        self,
        batch: SIDTrainingBatch,
        *,
        target_index: int,
        generator: object | None = None,
    ) -> SIDLossResult:
        torch = _require_torch()
        with torch.no_grad():
            generated = simulate_sid_student(
                self.student,
                batch,
                self.config.schedule,
                target_index=target_index,
                noise_policy=self.config.noise_policy,
                generator=generator,
                training=False,
            ).clean_latents.detach()
        score_sigmas = sample_sid_score_sigmas(generated, self.config, generator=generator)
        score_noise = _randn_like(generated, generator=generator)
        noisy_generated = self.fake_score.add_noise(generated, score_noise, score_sigmas)
        target_velocity = score_noise - generated
        prediction = _cfg_velocity(
            self.fake_score,
            noisy_generated,
            score_sigmas,
            batch,
            scale=self.config.fake_score_guidance_scale,
            training=True,
        )
        components: dict[str, object] = {
            "fake_score_flow": sid_fake_score_flow_loss_per_sample(
                prediction,
                target_velocity,
            )
        }
        component_weights = {"fake_score_flow": self.config.fake_score_flow_weight}
        if self.config.diffusion_gan_enabled:
            if not batch.has_real_samples:
                raise ValueError("SiD DiffusionGAN requires a real latent batch")
            assert isinstance(self.fake_score, SIDDiscriminatorAdapter)
            assert batch.real_latents is not None and batch.real_conditioning is not None
            real_noise = _randn_like(batch.real_latents, generator=generator)
            noisy_real = self.fake_score.add_noise(
                batch.real_latents,
                real_noise,
                score_sigmas,
            )
            fake_logits = self.fake_score.discriminator_logits(
                noisy_generated,
                score_sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=True,
            )
            real_logits = self.fake_score.discriminator_logits(
                noisy_real,
                score_sigmas,
                sample_ids=batch.real_sample_ids,
                conditioning=batch.real_conditioning,
                training=True,
            )
            components["fake_score_adversarial"] = sid_fake_score_adversarial_loss_per_sample(
                real_logits,
                fake_logits,
                latent_elements=int(generated[0].numel()),
            )
            component_weights["fake_score_adversarial"] = self.config.fake_score_adversarial_weight
        return self._result(
            batch,
            components,
            component_weights,
            target_index=target_index,
            score_sigmas=score_sigmas,
        )

    def generator_loss(
        self,
        batch: SIDTrainingBatch,
        *,
        target_index: int,
        generator: object | None = None,
    ) -> SIDLossResult:
        torch = _require_torch()
        generated = simulate_sid_student(
            self.student,
            batch,
            self.config.schedule,
            target_index=target_index,
            noise_policy=self.config.noise_policy,
            generator=generator,
            training=True,
        ).clean_latents
        score_sigmas = sample_sid_score_sigmas(generated, self.config, generator=generator)
        score_noise = _randn_like(generated, generator=generator)
        noisy_generated = self.teacher.add_noise(generated, score_noise, score_sigmas)
        with torch.no_grad():
            teacher_clean = _cfg_clean(
                self.teacher,
                noisy_generated.detach(),
                score_sigmas,
                batch,
                scale=self.config.teacher_guidance_scale,
                training=False,
            )
            fake_clean = _cfg_clean(
                self.fake_score,
                noisy_generated.detach(),
                score_sigmas,
                batch,
                scale=self.config.fake_score_guidance_scale,
                training=False,
            )
            score_weight = sid_score_weight(
                score_sigmas,
                scheme=self.config.score_weighting,
                epsilon=self.config.weighting_epsilon,
                generated=generated.detach(),
                teacher_clean=teacher_clean,
            )
        components: dict[str, object] = {
            "score_identity": sid_generator_loss_per_sample(
                generated,
                teacher_clean,
                fake_clean,
                score_weight,
                alpha=self.config.alpha,
            )
        }
        component_weights = {"score_identity": self.config.score_identity_weight}
        if self.config.diffusion_gan_enabled:
            if not batch.has_real_samples:
                raise ValueError("SiD DiffusionGAN requires a real latent batch")
            assert isinstance(self.fake_score, SIDDiscriminatorAdapter)
            fake_logits = self.fake_score.discriminator_logits(
                noisy_generated,
                score_sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=False,
            )
            adversarial_value = sid_generator_adversarial_loss_per_sample(
                fake_logits,
                score_weight,
                latent_elements=int(generated[0].numel()),
            )
            input_gradient = torch.autograd.grad(
                adversarial_value.sum(),
                generated,
                retain_graph=False,
                create_graph=False,
            )[0].detach()
            adversarial_proxy = (generated.float() * input_gradient.float()).reshape(
                generated.shape[0], -1
            ).sum(dim=1)
            components["generator_adversarial"] = adversarial_proxy + (
                adversarial_value.detach() - adversarial_proxy.detach()
            )
            component_weights["generator_adversarial"] = self.config.generator_adversarial_weight
        return self._result(
            batch,
            components,
            component_weights,
            target_index=target_index,
            score_sigmas=score_sigmas,
        )


__all__ = [
    "NativeSIDLossAdapter",
    "SIDConfig",
    "SIDFewStepPrediction",
    "SIDLossResult",
    "sample_sid_score_sigmas",
    "simulate_sid_student",
]
