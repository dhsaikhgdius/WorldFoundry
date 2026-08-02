"""Model-neutral DMD2 objective with a shared guidance/discriminator role."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.objectives.flow_matching import flow_shift_sigmas

from ....recipes.post_training.algorithms.dmd2 import DMD2AlgorithmSpec
from ..dmd.objective import FewStepSchedule
from .contracts import (
    DMD2GuidanceAdapter,
    DMD2PredictionAdapter,
    DMD2TrainingBatch,
)
from .math import (
    dmd2_distribution_gradient,
    dmd2_generator_adversarial_loss,
    dmd2_guidance_adversarial_loss,
    dmd2_proxy_loss_per_sample,
    dmd2_weighted_total,
)


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("native DMD2 requires the 'train-core' extra") from error
    return torch


@dataclass(frozen=True, slots=True)
class DMD2Config:
    schedule: FewStepSchedule
    normalization_axes: tuple[int, ...]
    num_train_timesteps: int = 1000
    score_min_sigma: float = 0.02
    score_max_sigma: float = 0.98
    score_flow_shift: float = 1.0
    teacher_guidance_scale: float = 6.0
    normalization_epsilon: float = 0.0
    score_timestep_mode: str = "per-sample"
    distribution_matching_weight: float = 1.0
    generator_adversarial_weight: float = 1.0
    guidance_denoising_weight: float = 1.0
    guidance_adversarial_weight: float = 1.0
    diffusion_gan_max_sigma: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, FewStepSchedule):
            raise TypeError("schedule must be FewStepSchedule")
        axes = tuple(int(axis) for axis in self.normalization_axes)
        if not axes or any(axis <= 0 for axis in axes):
            raise ValueError("normalization_axes must list positive non-batch axes")
        if len(set(axes)) != len(axes) or tuple(sorted(axes)) != axes:
            raise ValueError("normalization_axes must be unique and strictly increasing")
        if isinstance(self.num_train_timesteps, bool) or int(self.num_train_timesteps) < 2:
            raise ValueError("num_train_timesteps must be an integer >= 2")
        minimum = float(self.score_min_sigma)
        maximum = float(self.score_max_sigma)
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("DMD2 score sigma bounds must satisfy 0 <= min < max <= 1")
        if self.score_timestep_mode not in {"per-sample", "batch-shared"}:
            raise ValueError("score_timestep_mode must be 'per-sample' or 'batch-shared'")
        for name in (
            "score_flow_shift",
            "teacher_guidance_scale",
            "normalization_epsilon",
            "distribution_matching_weight",
            "generator_adversarial_weight",
            "guidance_denoising_weight",
            "guidance_adversarial_weight",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name != "teacher_guidance_scale" and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.score_flow_shift) <= 0:
            raise ValueError("score_flow_shift must be positive")
        if self.distribution_matching_weight + self.generator_adversarial_weight <= 0:
            raise ValueError("DMD2 generator has no enabled loss")
        if self.guidance_denoising_weight + self.guidance_adversarial_weight <= 0:
            raise ValueError("DMD2 guidance has no enabled loss")
        if self.diffusion_gan_max_sigma is not None:
            maximum_gan_sigma = float(self.diffusion_gan_max_sigma)
            if not isfinite(maximum_gan_sigma) or not 0.0 < maximum_gan_sigma <= 1.0:
                raise ValueError("diffusion_gan_max_sigma must be in (0,1]")
            object.__setattr__(self, "diffusion_gan_max_sigma", maximum_gan_sigma)
        object.__setattr__(self, "normalization_axes", axes)

    @classmethod
    def from_recipe(cls, spec: DMD2AlgorithmSpec) -> DMD2Config:
        if not isinstance(spec, DMD2AlgorithmSpec):
            raise TypeError("spec must be DMD2AlgorithmSpec")
        return cls(
            schedule=FewStepSchedule(spec.student_timesteps, spec.student_sigmas),
            normalization_axes=spec.normalization_axes,
            num_train_timesteps=spec.num_train_timesteps,
            score_min_sigma=spec.score_min_sigma,
            score_max_sigma=spec.score_max_sigma,
            score_flow_shift=spec.score_flow_shift,
            teacher_guidance_scale=spec.teacher_guidance_scale,
            normalization_epsilon=spec.normalization_epsilon,
            score_timestep_mode=spec.score_timestep_mode,
            distribution_matching_weight=spec.distribution_matching_weight,
            generator_adversarial_weight=spec.generator_adversarial_weight,
            guidance_denoising_weight=spec.guidance_denoising_weight,
            guidance_adversarial_weight=spec.guidance_adversarial_weight,
            diffusion_gan_max_sigma=spec.diffusion_gan_max_sigma,
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-dmd2-config",
                "schedule_digest": self.schedule.digest,
                "normalization_axes": self.normalization_axes,
                "num_train_timesteps": self.num_train_timesteps,
                "score_min_sigma": self.score_min_sigma,
                "score_max_sigma": self.score_max_sigma,
                "score_flow_shift": self.score_flow_shift,
                "teacher_guidance_scale": self.teacher_guidance_scale,
                "normalization_epsilon": self.normalization_epsilon,
                "score_timestep_mode": self.score_timestep_mode,
                "distribution_matching_weight": self.distribution_matching_weight,
                "generator_adversarial_weight": self.generator_adversarial_weight,
                "guidance_denoising_weight": self.guidance_denoising_weight,
                "guidance_adversarial_weight": self.guidance_adversarial_weight,
                "diffusion_gan_max_sigma": self.diffusion_gan_max_sigma,
            }
        )


@dataclass(frozen=True, slots=True)
class DMD2FewStepPrediction:
    clean_latents: object
    target_index: int
    timestep: float
    sigma: float


@dataclass(frozen=True, slots=True)
class DMD2LossResult:
    loss: object
    metrics: Mapping[str, object]


def dmd2_teacher_guidance(
    conditional_clean: object,
    unconditional_clean: object,
    guidance_scale: float,
) -> object:
    if getattr(conditional_clean, "shape", None) != getattr(unconditional_clean, "shape", None):
        raise ValueError("conditional and unconditional teacher predictions must match")
    scale = float(guidance_scale)
    if not isfinite(scale):
        raise ValueError("guidance_scale must be finite")
    return unconditional_clean + scale * (conditional_clean - unconditional_clean)


def _randn_like(reference: object, *, generator: object | None) -> object:
    torch = _require_torch()
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _levels(reference: object, level: float) -> object:
    torch = _require_torch()
    return torch.full(
        (int(reference.shape[0]),),
        float(level),
        device=reference.device,
        dtype=torch.float32,
    )


def simulate_dmd2_student(
    student: DMD2PredictionAdapter,
    batch: DMD2TrainingBatch,
    schedule: FewStepSchedule,
    *,
    generator: object | None = None,
    target_index: int | None = None,
    training: bool = True,
) -> DMD2FewStepPrediction:
    """Run a no-grad prefix and independently re-noise before the selected step."""

    torch = _require_torch()
    if not isinstance(student, DMD2PredictionAdapter):
        raise TypeError("student must implement DMD2PredictionAdapter")
    reference = batch.real_latents
    if not torch.is_tensor(reference):
        raise TypeError("real_latents must be a tensor")
    if target_index is None:
        target_index = int(
            torch.randint(
                0,
                len(schedule.sigmas),
                (),
                device=reference.device,
                generator=generator,
            ).item()
        )
    if isinstance(target_index, bool) or not 0 <= int(target_index) < len(schedule.sigmas):
        raise ValueError("target_index falls outside the few-step schedule")
    selected = int(target_index)
    current = _randn_like(reference, generator=generator)
    for index in range(selected):
        with torch.no_grad():
            predicted = student.predict_clean(
                current,
                _levels(current, schedule.sigmas[index]),
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=False,
            )
            if not torch.is_tensor(predicted) or predicted.shape != reference.shape:
                raise ValueError("student predict_clean must preserve the latent shape")
            independent_noise = _randn_like(predicted, generator=generator)
            current = student.add_noise(
                predicted,
                independent_noise,
                _levels(predicted, schedule.sigmas[index + 1]),
            )
    predicted = student.predict_clean(
        current,
        _levels(current, schedule.sigmas[selected]),
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
        training=training,
    )
    if not torch.is_tensor(predicted) or predicted.shape != reference.shape:
        raise ValueError("student predict_clean must preserve the latent shape")
    return DMD2FewStepPrediction(
        clean_latents=predicted,
        target_index=selected,
        timestep=schedule.timesteps[selected],
        sigma=schedule.sigmas[selected],
    )


def sample_dmd2_score_levels(
    reference: object,
    config: DMD2Config,
    *,
    generator: object | None = None,
) -> object:
    torch = _require_torch()
    if not torch.is_tensor(reference) or reference.ndim < 2:
        raise TypeError("reference must be a [B,...] tensor")
    count = 1 if config.score_timestep_mode == "batch-shared" else int(reference.shape[0])
    indices = torch.randint(
        0,
        config.num_train_timesteps,
        (count,),
        device=reference.device,
        generator=generator,
    )
    base = indices.float() / float(config.num_train_timesteps)
    shifted = flow_shift_sigmas(base, config.score_flow_shift).clamp(
        min=config.score_min_sigma,
        max=config.score_max_sigma,
    )
    return shifted.expand(int(reference.shape[0])) if count == 1 else shifted


def _sample_weights(batch: DMD2TrainingBatch, *, device: object) -> object:
    torch = _require_torch()
    if batch.sample_weights is None:
        return torch.ones((batch.batch_size,), device=device, dtype=torch.float32)
    if not torch.is_tensor(batch.sample_weights):
        raise TypeError("sample_weights must be a tensor")
    weights = batch.sample_weights.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
        raise ValueError("sample_weights must be finite and non-negative")
    if not bool(weights.sum() > 0):
        raise ValueError("DMD2 batch must contain positive sample weight")
    return weights


def _weighted_mean(per_sample: object, weights: object) -> tuple[object, object, object]:
    torch = _require_torch()
    if not torch.is_tensor(per_sample) or per_sample.ndim != 1 or per_sample.shape != weights.shape:
        raise ValueError("DMD2 losses must return one value per weighted sample")
    numerator = (per_sample.float() * weights).sum()
    denominator = weights.sum()
    return numerator / denominator, numerator, denominator


def _gan_input(
    adapter: DMD2GuidanceAdapter,
    clean_latents: object,
    max_sigma: float | None,
    *,
    generator: object | None,
) -> tuple[object, object]:
    torch = _require_torch()
    if max_sigma is None:
        return clean_latents, torch.zeros(
            (int(clean_latents.shape[0]),),
            device=clean_latents.device,
            dtype=torch.float32,
        )
    levels = torch.rand(
        (int(clean_latents.shape[0]),),
        device=clean_latents.device,
        dtype=torch.float32,
        generator=generator,
    ) * float(max_sigma)
    noise = _randn_like(clean_latents, generator=generator)
    return adapter.add_noise(clean_latents, noise, levels), levels


class NativeDMD2LossAdapter:
    """Own the DMD proxy, native fake denoising, and softplus GAN losses."""

    def __init__(
        self,
        student: DMD2PredictionAdapter,
        real_score: DMD2PredictionAdapter,
        guidance: DMD2GuidanceAdapter,
        config: DMD2Config,
        *,
        config_digest: str | None = None,
    ) -> None:
        if not isinstance(student, DMD2PredictionAdapter):
            raise TypeError("student must implement DMD2PredictionAdapter")
        if not isinstance(real_score, DMD2PredictionAdapter):
            raise TypeError("real_score must implement DMD2PredictionAdapter")
        if not isinstance(guidance, DMD2GuidanceAdapter):
            raise TypeError("guidance must implement DMD2GuidanceAdapter")
        modules = (student.module, real_score.module, guidance.module)
        torch = _require_torch()
        if not all(isinstance(module, torch.nn.Module) for module in modules):
            raise TypeError("DMD2 adapters must expose nn.Module instances")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("DMD2 student, teacher, and guidance modules must be distinct")
        kinds = tuple(str(adapter.noise_process_kind).strip().lower() for adapter in (student, real_score, guidance))
        if set(kinds) != {"flow-matching"}:
            raise ValueError("native DMD2 adapters must use the flow-matching noise process")
        digests = tuple(str(adapter.noise_process_digest).strip() for adapter in (student, real_score, guidance))
        if not all(digests) or len(set(digests)) != 1:
            raise ValueError("DMD2 roles must declare the same non-empty noise_process_digest")
        if not isinstance(config, DMD2Config):
            raise TypeError("config must be DMD2Config")
        self.student = student
        self.real_score = real_score
        self.guidance = guidance
        self.config = config
        self.config_digest = config.digest if config_digest is None else str(config_digest)
        if not self.config_digest.strip():
            raise ValueError("config_digest must be non-empty")

    def loss_denominator(self, batch: DMD2TrainingBatch, *, role: str) -> object:
        if role not in {"generator", "guidance"}:
            raise ValueError(f"unsupported DMD2 loss role: {role!r}")
        return _sample_weights(batch, device=batch.real_latents.device).sum()

    def generator_loss(
        self,
        batch: DMD2TrainingBatch,
        *,
        generator: object | None = None,
    ) -> DMD2LossResult:
        torch = _require_torch()
        prediction = simulate_dmd2_student(
            self.student,
            batch,
            self.config.schedule,
            generator=generator,
            training=True,
        )
        generated = prediction.clean_latents
        zero = torch.zeros((batch.batch_size,), device=generated.device, dtype=torch.float32)
        dm_per_sample = zero
        normalizer = zero
        score_levels = zero
        if self.config.distribution_matching_weight > 0:
            with torch.no_grad():
                score_levels = sample_dmd2_score_levels(generated, self.config, generator=generator)
                noise = _randn_like(generated, generator=generator)
                noisy = self.student.add_noise(generated.detach(), noise, score_levels)
                fake_clean = self.guidance.predict_clean(
                    noisy,
                    score_levels,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    training=False,
                )
                real_conditional = self.real_score.predict_clean(
                    noisy,
                    score_levels,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    training=False,
                )
                real_unconditional = self.real_score.predict_clean(
                    noisy,
                    score_levels,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.unconditional_conditioning,
                    training=False,
                    branch="negative",
                )
                guided_real = dmd2_teacher_guidance(
                    real_conditional,
                    real_unconditional,
                    self.config.teacher_guidance_scale,
                )
                gradient, normalizer = dmd2_distribution_gradient(
                    noisy,
                    fake_clean,
                    guided_real,
                    normalization_axes=self.config.normalization_axes,
                    normalization_epsilon=self.config.normalization_epsilon,
                )
            dm_per_sample = dmd2_proxy_loss_per_sample(generated, gradient)
        generator_adversarial = zero
        if self.config.generator_adversarial_weight > 0:
            fake_input, fake_levels = _gan_input(
                self.guidance,
                generated,
                self.config.diffusion_gan_max_sigma,
                generator=generator,
            )
            fake_logits = self.guidance.discriminator_logits(
                fake_input,
                fake_levels,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=False,
            )
            generator_adversarial = dmd2_generator_adversarial_loss(fake_logits)
        combined = dmd2_weighted_total(
            {
                "distribution_matching": dm_per_sample,
                "generator_adversarial": generator_adversarial,
            },
            {
                "distribution_matching": self.config.distribution_matching_weight,
                "generator_adversarial": self.config.generator_adversarial_weight,
            },
        )
        weights = _sample_weights(batch, device=generated.device)
        actual_loss, numerator, denominator = _weighted_mean(combined, weights)
        dm_loss = (
            _weighted_mean(dm_per_sample, weights)[0]
            * self.config.distribution_matching_weight
        )
        if self.config.generator_adversarial_weight > 0:
            adversarial_loss = (
                _weighted_mean(generator_adversarial, weights)[0]
                * self.config.generator_adversarial_weight
            )
            adversarial_gradient = torch.autograd.grad(
                adversarial_loss,
                generated,
                create_graph=False,
                retain_graph=False,
            )[0]
            adversarial_proxy = (generated.float() * adversarial_gradient.detach()).sum()
        else:
            adversarial_proxy = torch.zeros((), device=generated.device, dtype=torch.float32)
        # Preserve the reported value while backpropagating only into the
        # student.  autograd.grad above does not materialize guidance grads,
        # which is safe for sharded guidance modules as well as plain/DDP.
        loss = (
            actual_loss.detach()
            + dm_loss
            - dm_loss.detach()
            + adversarial_proxy
            - adversarial_proxy.detach()
        )
        return DMD2LossResult(
            loss=loss,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
                "distribution_matching": _weighted_mean(dm_per_sample, weights)[0].detach(),
                "generator_adversarial": _weighted_mean(generator_adversarial, weights)[0].detach(),
                "normalizer_mean": normalizer.detach().mean(),
                "score_level_mean": score_levels.detach().mean(),
                "student_target_index": torch.tensor(prediction.target_index, device=loss.device),
                "student_timestep": torch.tensor(prediction.timestep, device=loss.device),
            },
        )

    def guidance_loss(
        self,
        batch: DMD2TrainingBatch,
        *,
        generator: object | None = None,
    ) -> DMD2LossResult:
        torch = _require_torch()
        with torch.no_grad():
            prediction = simulate_dmd2_student(
                self.student,
                batch,
                self.config.schedule,
                generator=generator,
                training=False,
            )
            generated = prediction.clean_latents.detach()
        zero = torch.zeros((batch.batch_size,), device=generated.device, dtype=torch.float32)
        denoising = zero
        score_levels = zero
        if self.config.guidance_denoising_weight > 0:
            score_levels = sample_dmd2_score_levels(generated, self.config, generator=generator)
            noise = _randn_like(generated, generator=generator)
            noisy = self.guidance.add_noise(generated, noise, score_levels)
            denoising = self.guidance.denoising_loss_per_sample(
                generated,
                noisy,
                noise,
                score_levels,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=True,
            )
            if not torch.is_tensor(denoising) or denoising.shape != zero.shape:
                raise ValueError("guidance denoising seam must return shape [B]")
        adversarial = zero
        if self.config.guidance_adversarial_weight > 0:
            fake_input, fake_levels = _gan_input(
                self.guidance,
                generated,
                self.config.diffusion_gan_max_sigma,
                generator=generator,
            )
            real_input, real_levels = _gan_input(
                self.guidance,
                batch.real_latents,
                self.config.diffusion_gan_max_sigma,
                generator=generator,
            )
            fake_logits = self.guidance.discriminator_logits(
                fake_input,
                fake_levels,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=True,
            )
            real_logits = self.guidance.discriminator_logits(
                real_input,
                real_levels,
                sample_ids=batch.real_sample_ids,
                conditioning=batch.real_conditioning,
                training=True,
            )
            adversarial = dmd2_guidance_adversarial_loss(real_logits, fake_logits)
        combined = dmd2_weighted_total(
            {"denoising": denoising, "adversarial": adversarial},
            {
                "denoising": self.config.guidance_denoising_weight,
                "adversarial": self.config.guidance_adversarial_weight,
            },
        )
        weights = _sample_weights(batch, device=generated.device)
        loss, numerator, denominator = _weighted_mean(combined, weights)
        return DMD2LossResult(
            loss=loss,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
                "denoising": _weighted_mean(denoising, weights)[0].detach(),
                "adversarial": _weighted_mean(adversarial, weights)[0].detach(),
                "score_level_mean": score_levels.detach().mean(),
                "student_target_index": torch.tensor(prediction.target_index, device=loss.device),
                "student_timestep": torch.tensor(prediction.timestep, device=loss.device),
            },
        )


__all__ = [
    "DMD2Config",
    "DMD2FewStepPrediction",
    "DMD2LossResult",
    "NativeDMD2LossAdapter",
    "dmd2_teacher_guidance",
    "sample_dmd2_score_levels",
    "simulate_dmd2_student",
]
