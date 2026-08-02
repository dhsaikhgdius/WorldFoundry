"""Native model-neutral multi-step-to-few-step distribution matching."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.objectives.flow_matching import (
    flow_interpolate,
    flow_matching_denominator,
    flow_matching_mse,
    flow_shift_sigmas,
    flow_velocity_target,
)

from ...shared.contracts import FlowPredictionAdapter
from .contracts import DMDTrainingBatch


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("native DMD requires the 'train-core' extra") from error
    return torch


@dataclass(frozen=True, slots=True)
class FewStepSchedule:
    """Explicit model timesteps and effective flow sigmas for the student."""

    timesteps: tuple[float, ...]
    sigmas: tuple[float, ...]

    def __post_init__(self) -> None:
        timesteps = tuple(float(value) for value in self.timesteps)
        sigmas = tuple(float(value) for value in self.sigmas)
        if not timesteps or len(timesteps) != len(sigmas):
            raise ValueError("few-step timesteps and sigmas must be non-empty and equal length")
        if any(not isfinite(value) or value < 0 for value in timesteps):
            raise ValueError("few-step timesteps must be finite and non-negative")
        if any(not isfinite(value) or not 0 < value <= 1 for value in sigmas):
            raise ValueError("few-step sigmas must be finite and in (0,1]")
        if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
            raise ValueError("few-step timesteps must be strictly descending")
        if any(left <= right for left, right in zip(sigmas, sigmas[1:])):
            raise ValueError("few-step sigmas must be strictly descending")
        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "sigmas", sigmas)

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-few-step-schedule",
                "timesteps": self.timesteps,
                "sigmas": self.sigmas,
            }
        )

    @classmethod
    def from_effective_timesteps(
        cls,
        timesteps: tuple[float, ...],
        *,
        num_train_timesteps: int,
    ) -> FewStepSchedule:
        """Build only when the supplied timesteps already encode flow shifting."""

        if isinstance(num_train_timesteps, bool) or int(num_train_timesteps) < 2:
            raise ValueError("num_train_timesteps must be an integer >= 2")
        resolved = tuple(float(value) for value in timesteps)
        return cls(
            timesteps=resolved,
            sigmas=tuple(value / int(num_train_timesteps) for value in resolved),
        )


@dataclass(frozen=True, slots=True)
class DMDConfig:
    schedule: FewStepSchedule
    num_train_timesteps: int = 1000
    score_min_sigma: float = 0.02
    score_max_sigma: float = 0.98
    score_flow_shift: float = 1.0
    teacher_guidance_scale: float = 3.5
    normalization_epsilon: float = 0.0
    shared_score_timestep: bool = True
    per_sample_normalization: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, FewStepSchedule):
            raise TypeError("schedule must be a FewStepSchedule")
        if isinstance(self.num_train_timesteps, bool) or int(self.num_train_timesteps) < 2:
            raise ValueError("num_train_timesteps must be an integer >= 2")
        minimum = float(self.score_min_sigma)
        maximum = float(self.score_max_sigma)
        if not 0 <= minimum < maximum <= 1:
            raise ValueError("score sigma bounds must satisfy 0 <= min < max <= 1")
        for name, value in (
            ("score_flow_shift", self.score_flow_shift),
            ("teacher_guidance_scale", self.teacher_guidance_scale),
            ("normalization_epsilon", self.normalization_epsilon),
        ):
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(self.score_flow_shift) <= 0 or float(self.normalization_epsilon) < 0:
            raise ValueError("score_flow_shift must be positive and normalization_epsilon non-negative")
        if not isinstance(self.shared_score_timestep, bool):
            raise TypeError("shared_score_timestep must be a bool")
        if not isinstance(self.per_sample_normalization, bool):
            raise TypeError("per_sample_normalization must be a bool")

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-dmd-config",
                "schedule_digest": self.schedule.digest,
                "num_train_timesteps": int(self.num_train_timesteps),
                "score_min_sigma": float(self.score_min_sigma),
                "score_max_sigma": float(self.score_max_sigma),
                "score_flow_shift": float(self.score_flow_shift),
                "teacher_guidance_scale": float(self.teacher_guidance_scale),
                "normalization_epsilon": float(self.normalization_epsilon),
                "shared_score_timestep": bool(self.shared_score_timestep),
                "per_sample_normalization": self.per_sample_normalization,
            }
        )


@dataclass(frozen=True, slots=True)
class FewStepPrediction:
    clean_latents: object
    target_index: int
    timestep: float
    sigma: float


@runtime_checkable
class DMDStudentSampler(Protocol):
    """Optional execution seam for architecture-specific student rollout."""

    execution_digest: str

    def sample(
        self,
        batch: DMDTrainingBatch,
        schedule: FewStepSchedule,
        *,
        generator: object | None,
        training: bool,
    ) -> FewStepPrediction: ...


@dataclass(frozen=True, slots=True)
class DMDLossResult:
    loss: object
    metrics: dict[str, object]


def dmd_teacher_guidance(conditional_clean: object, unconditional_clean: object, guidance_scale: float) -> object:
    """DMD2 guidance: ``x_cond + w * (x_cond - x_uncond)``."""

    if getattr(conditional_clean, "shape", None) != getattr(unconditional_clean, "shape", None):
        raise ValueError("conditional and unconditional teacher predictions must match")
    resolved = float(guidance_scale)
    if not isfinite(resolved):
        raise ValueError("guidance_scale must be finite")
    return conditional_clean + resolved * (conditional_clean - unconditional_clean)


def dmd_distribution_gradient(
    generated_clean: object,
    fake_score_clean: object,
    real_score_clean: object,
    *,
    normalization_epsilon: float = 0.0,
    per_sample_normalization: bool = False,
) -> tuple[object, object]:
    """Return the normalized distribution-matching gradient and denominator."""

    torch = _require_torch()
    if not all(torch.is_tensor(value) for value in (generated_clean, fake_score_clean, real_score_clean)):
        raise TypeError("DMD gradient inputs must be torch.Tensor values")
    if generated_clean.shape != fake_score_clean.shape or generated_clean.shape != real_score_clean.shape:
        raise ValueError("DMD gradient inputs must share a shape")
    epsilon = float(normalization_epsilon)
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError("normalization_epsilon must be finite and non-negative")
    if not isinstance(per_sample_normalization, bool):
        raise TypeError("per_sample_normalization must be a bool")
    reduction_dims = tuple(range(1, generated_clean.ndim)) if per_sample_normalization else None
    denominator = (
        (generated_clean.float() - real_score_clean.float())
        .abs()
        .mean(
            dim=reduction_dims,
            keepdim=per_sample_normalization,
        )
    )
    normalizer = denominator.clamp_min(epsilon) if epsilon > 0 else denominator
    gradient = (fake_score_clean.float() - real_score_clean.float()) / normalizer
    gradient = torch.nan_to_num(gradient)
    return gradient, denominator


def dmd_proxy_loss(generated_clean: object, distribution_gradient: object) -> object:
    """Construct a scalar whose gradient w.r.t. generated clean is the DMD gradient."""

    torch = _require_torch()
    if not torch.is_tensor(generated_clean) or not torch.is_tensor(distribution_gradient):
        raise TypeError("DMD proxy-loss inputs must be torch.Tensor values")
    if generated_clean.shape != distribution_gradient.shape:
        raise ValueError("generated_clean and distribution_gradient must match")
    generated = generated_clean.float()
    target = (generated - distribution_gradient.float()).detach()
    return 0.5 * torch.nn.functional.mse_loss(generated, target)


def _random_normal_like(reference: object, *, generator: object | None) -> object:
    torch = _require_torch()
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _batch_sigmas(reference: object, sigma: float) -> object:
    torch = _require_torch()
    return torch.full((int(reference.shape[0]),), float(sigma), device=reference.device, dtype=torch.float32)


def simulate_few_step_student(
    predictor: FlowPredictionAdapter,
    batch: DMDTrainingBatch,
    schedule: FewStepSchedule,
    *,
    generator: object | None = None,
    target_index: int | None = None,
    training: bool = True,
) -> FewStepPrediction:
    """Simulate only the inference prefix preceding the differentiable step.

    Every intermediate clean estimate is re-noised independently at the next
    scheduled sigma, matching the model-author DMD training procedure.  Only
    the selected final student call retains autograd state.
    """

    torch = _require_torch()
    clean_reference = batch.clean_latents
    if not torch.is_tensor(clean_reference):
        raise TypeError("DMD clean_latents must be a torch.Tensor")
    if target_index is None:
        target_index = int(
            torch.randint(
                0,
                len(schedule.sigmas),
                (),
                device=clean_reference.device,
                generator=generator,
            ).item()
        )
    if isinstance(target_index, bool) or not 0 <= int(target_index) < len(schedule.sigmas):
        raise ValueError("target_index falls outside the few-step schedule")
    selected = int(target_index)
    current = _random_normal_like(clean_reference, generator=generator)
    for index in range(selected):
        sigma = _batch_sigmas(current, schedule.sigmas[index])
        with torch.no_grad():
            predicted_clean = predictor.predict_clean(
                current,
                sigma,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=False,
            )
            next_noise = _random_normal_like(predicted_clean, generator=generator)
            current = flow_interpolate(
                predicted_clean,
                next_noise,
                _batch_sigmas(predicted_clean, schedule.sigmas[index + 1]),
            )
    selected_sigma = _batch_sigmas(current, schedule.sigmas[selected])
    predicted_clean = predictor.predict_clean(
        current,
        selected_sigma,
        sample_ids=batch.sample_ids,
        conditioning=batch.conditioning,
        training=training,
    )
    return FewStepPrediction(
        clean_latents=predicted_clean,
        target_index=selected,
        timestep=schedule.timesteps[selected],
        sigma=schedule.sigmas[selected],
    )


def sample_dmd_score_sigmas(reference: object, config: DMDConfig, *, generator: object | None = None) -> object:
    """Sample shifted score-model sigmas with an explicit shared/per-sample mode."""

    torch = _require_torch()
    count = 1 if config.shared_score_timestep else int(reference.shape[0])
    indices = torch.randint(
        0,
        int(config.num_train_timesteps),
        (count,),
        device=reference.device,
        generator=generator,
    )
    base = indices.float() / float(config.num_train_timesteps)
    shifted = flow_shift_sigmas(base, float(config.score_flow_shift)).clamp(
        min=float(config.score_min_sigma),
        max=float(config.score_max_sigma),
    )
    if count == 1:
        shifted = shifted.expand(int(reference.shape[0]))
    return shifted


class FlowDMDLossAdapter:
    """Own DMD generator/teacher/fake-score equations over native predictors."""

    def __init__(
        self,
        student: FlowPredictionAdapter | None,
        real_score: FlowPredictionAdapter,
        fake_score: FlowPredictionAdapter,
        config: DMDConfig,
        *,
        student_sampler: DMDStudentSampler | None = None,
    ) -> None:
        if not all(isinstance(adapter, FlowPredictionAdapter) for adapter in (real_score, fake_score)):
            raise TypeError("real_score and fake_score must implement FlowPredictionAdapter")
        if student_sampler is None and not isinstance(student, FlowPredictionAdapter):
            raise TypeError("student must implement FlowPredictionAdapter without a student_sampler")
        if student_sampler is not None and not isinstance(student_sampler, DMDStudentSampler):
            raise TypeError("student_sampler must implement DMDStudentSampler")
        if student is not None and not isinstance(student, FlowPredictionAdapter):
            raise TypeError("student must be None or implement FlowPredictionAdapter")
        if not isinstance(config, DMDConfig):
            raise TypeError("config must be DMDConfig")
        self.student = student
        self.student_sampler = student_sampler
        self.real_score = real_score
        self.fake_score = fake_score
        self.config = config
        self.schedule_digest = (
            config.schedule.digest
            if student_sampler is None
            else canonical_sha256(
                {
                    "schema": "worldfoundry-dmd-execution",
                    "dmd_config_digest": config.digest,
                    "student_execution_digest": student_sampler.execution_digest,
                }
            )
        )

    def sample_student(
        self,
        batch: DMDTrainingBatch,
        *,
        generator: object | None,
        training: bool,
    ) -> FewStepPrediction:
        if self.student_sampler is not None:
            return self.student_sampler.sample(
                batch,
                self.config.schedule,
                generator=generator,
                training=training,
            )
        assert self.student is not None
        return simulate_few_step_student(
            self.student,
            batch,
            self.config.schedule,
            generator=generator,
            training=training,
        )

    def generator_loss_from_prediction(
        self,
        batch: DMDTrainingBatch,
        generated: FewStepPrediction,
        *,
        generator: object | None = None,
    ) -> DMDLossResult:
        """Evaluate DMD for an already sampled student prediction.

        Composite video objectives use this seam so the differentiable
        student rollout is shared by DMD and their additional regularizers.
        Running the student a second time would consume different randomness
        and would no longer describe one optimizer update.
        """

        torch = _require_torch()
        if not isinstance(generated, FewStepPrediction):
            raise TypeError("generated must be FewStepPrediction")
        generated_clean = generated.clean_latents
        if not torch.is_tensor(generated_clean):
            raise TypeError("generated clean latents must be a torch.Tensor")
        if tuple(generated_clean.shape) != tuple(batch.clean_latents.shape):
            raise ValueError("generated clean latents must preserve the batch latent shape")
        with torch.no_grad():
            score_sigmas = sample_dmd_score_sigmas(
                generated_clean,
                self.config,
                generator=generator,
            )
            noise = _random_normal_like(generated_clean, generator=generator)
            noisy = flow_interpolate(generated_clean.detach(), noise, score_sigmas)
            fake_clean = self.fake_score.predict_clean(
                noisy,
                score_sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=False,
            )
            real_conditional = self.real_score.predict_clean(
                noisy,
                score_sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=False,
            )
            real_unconditional = self.real_score.predict_clean(
                noisy,
                score_sigmas,
                sample_ids=batch.sample_ids,
                conditioning=batch.unconditional_conditioning,
                training=False,
                branch="negative",
            )
            guided_real = dmd_teacher_guidance(
                real_conditional,
                real_unconditional,
                self.config.teacher_guidance_scale,
            )
            gradient, denominator = dmd_distribution_gradient(
                generated_clean,
                fake_clean,
                guided_real,
                normalization_epsilon=self.config.normalization_epsilon,
                per_sample_normalization=self.config.per_sample_normalization,
            )
        loss = dmd_proxy_loss(generated_clean, gradient)
        loss_denominator = torch.tensor(
            generated_clean.numel(),
            device=loss.device,
            dtype=torch.float32,
        )
        return DMDLossResult(
            loss=loss,
            metrics={
                "loss_numerator": loss.detach() * loss_denominator,
                "loss_denominator": loss_denominator,
                "dmd_normalizer": denominator.detach(),
                "dmd_gradient_abs_mean": gradient.detach().abs().mean(),
                "score_sigma_mean": score_sigmas.detach().float().mean(),
                "student_target_index": torch.tensor(
                    generated.target_index,
                    device=loss.device,
                ),
                "student_timestep": torch.tensor(
                    generated.timestep,
                    device=loss.device,
                ),
            },
        )

    def loss_denominator(self, batch: DMDTrainingBatch, *, role: str) -> object:
        """Return the reduction weight before a model forward for DP scaling."""

        torch = _require_torch()
        if not torch.is_tensor(batch.clean_latents):
            raise TypeError("DMD clean_latents must be a torch.Tensor")
        if role == "generator":
            return torch.tensor(
                batch.clean_latents.numel(),
                device=batch.clean_latents.device,
                dtype=torch.float32,
            )
        if role == "fake-score":
            return flow_matching_denominator(
                batch.clean_latents,
                loss_mask=batch.loss_mask,
                sample_weights=batch.sample_weights,
            )
        raise ValueError(f"unsupported DMD loss role: {role!r}")

    def generator_loss(
        self,
        batch: DMDTrainingBatch,
        *,
        generator: object | None = None,
    ) -> DMDLossResult:
        generated = self.sample_student(
            batch,
            generator=generator,
            training=True,
        )
        return self.generator_loss_from_prediction(
            batch,
            generated,
            generator=generator,
        )

    def fake_score_loss(
        self,
        batch: DMDTrainingBatch,
        *,
        generator: object | None = None,
    ) -> DMDLossResult:
        torch = _require_torch()
        with torch.no_grad():
            generated = self.sample_student(
                batch,
                generator=generator,
                training=False,
            )
            generated_clean = generated.clean_latents.detach()
            score_sigmas = sample_dmd_score_sigmas(generated_clean, self.config, generator=generator)
            noise = _random_normal_like(generated_clean, generator=generator)
            noisy = flow_interpolate(generated_clean, noise, score_sigmas)
            target = flow_velocity_target(generated_clean, noise)
        prediction = self.fake_score.predict_velocity(
            noisy,
            score_sigmas,
            sample_ids=batch.sample_ids,
            conditioning=batch.conditioning,
            training=True,
        )
        reduced = flow_matching_mse(
            prediction,
            target,
            loss_mask=batch.loss_mask,
            sample_weights=batch.sample_weights,
        )
        return DMDLossResult(
            loss=reduced.loss,
            metrics={
                "loss_numerator": reduced.numerator.detach(),
                "loss_denominator": reduced.denominator.detach(),
                "score_sigma_mean": score_sigmas.detach().float().mean(),
                "student_target_index": torch.tensor(generated.target_index, device=reduced.loss.device),
                "student_timestep": torch.tensor(generated.timestep, device=reduced.loss.device),
            },
        )


__all__ = [
    "DMDConfig",
    "DMDLossResult",
    "DMDStudentSampler",
    "FewStepPrediction",
    "FewStepSchedule",
    "FlowDMDLossAdapter",
    "dmd_distribution_gradient",
    "dmd_proxy_loss",
    "dmd_teacher_guidance",
    "sample_dmd_score_sigmas",
    "simulate_few_step_student",
]
