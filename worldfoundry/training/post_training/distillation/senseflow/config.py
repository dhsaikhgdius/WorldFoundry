"""Validated controls for native flow-based SenseFlow training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.objectives.flow_matching import flow_shift_sigmas
from worldfoundry.training.recipes.post_training.algorithms.senseflow import (
    SenseFlowAlgorithmSpec,
    SenseFlowScheduleSpec,
)
from worldfoundry.training.recipes.spec import OptimizerSpec

ISGLoss = Literal["charbonnier", "mse"]
ScoreSampling = Literal["uniform-schedule-index", "logit-normal-scheduler-index"]
SchedulerCadence = Literal["iteration", "generator-update"]


def _finite_range(value: tuple[float, float], *, field_name: str) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{field_name} must be a two-value tuple")
    lower, upper = (float(item) for item in value)
    if not isfinite(lower) or not isfinite(upper) or lower < 0 or lower > upper:
        raise ValueError(f"{field_name} must be finite and satisfy 0 <= lower <= upper")
    return lower, upper


@dataclass(frozen=True, slots=True)
class SenseFlowSchedule:
    """Coarse anchors plus the exact FlowMatch scheduler index mapping."""

    timesteps: tuple[int, ...]
    sigmas: tuple[float, ...]
    isg_margin: int
    num_train_timesteps: int = 1000
    flow_shift: float = 1.0
    timestep_index_offset: int = 0
    terminal_timestep: int = 0
    terminal_sigma: float = 0.0
    adversarial_scales: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.timesteps or len(self.timesteps) != len(self.sigmas):
            raise ValueError("SenseFlow timesteps and sigmas must be non-empty and aligned")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.timesteps):
            raise TypeError("SenseFlow timesteps must be integers")
        if any(value <= 0 for value in self.timesteps):
            raise ValueError("SenseFlow anchor timesteps must be positive")
        if any(left <= right for left, right in zip(self.timesteps, self.timesteps[1:])):
            raise ValueError("SenseFlow timesteps must be strictly descending")
        sigmas = tuple(float(value) for value in self.sigmas)
        if any(not isfinite(value) or not 0 < value <= 1 for value in sigmas):
            raise ValueError("SenseFlow anchor sigmas must lie in (0,1]")
        if any(left <= right for left, right in zip(sigmas, sigmas[1:])):
            raise ValueError("SenseFlow sigmas must be strictly descending")
        if isinstance(self.isg_margin, bool) or not isinstance(self.isg_margin, int):
            raise TypeError("isg_margin must be an integer")
        if self.isg_margin < 0:
            raise ValueError("isg_margin must be non-negative")
        if (
            isinstance(self.num_train_timesteps, bool)
            or not isinstance(self.num_train_timesteps, int)
            or self.num_train_timesteps < 2
        ):
            raise ValueError("num_train_timesteps must be an integer >= 2")
        flow_shift = float(self.flow_shift)
        if not isfinite(flow_shift) or flow_shift <= 0:
            raise ValueError("flow_shift must be finite and positive")
        if (
            isinstance(self.timestep_index_offset, bool)
            or not isinstance(self.timestep_index_offset, int)
            or self.timestep_index_offset < 0
        ):
            raise ValueError("timestep_index_offset must be a non-negative integer")
        if isinstance(self.terminal_timestep, bool) or not isinstance(self.terminal_timestep, int):
            raise TypeError("terminal_timestep must be an integer")
        terminal_sigma = float(self.terminal_sigma)
        if self.terminal_timestep < 0 or self.terminal_timestep >= self.timesteps[-1]:
            raise ValueError("terminal_timestep must be below the final anchor")
        if not isfinite(terminal_sigma) or not 0 <= terminal_sigma < sigmas[-1]:
            raise ValueError("terminal_sigma must be finite and below the final anchor sigma")
        for index, timestep in enumerate(self.timesteps):
            next_timestep = self.next_timestep(index)
            if next_timestep + self.isg_margin > timestep - self.isg_margin:
                raise ValueError("every SenseFlow segment must contain an ISG midpoint after margins")
        expected_sigmas = tuple(self._sigma_at_index(value) for value in self.timesteps)
        if any(abs(actual - expected) > 1.0e-9 for actual, expected in zip(sigmas, expected_sigmas)):
            raise ValueError("SenseFlow anchor sigmas differ from the declared scheduler mapping")
        expected_terminal = self._sigma_at_index(self.terminal_timestep)
        if abs(terminal_sigma - expected_terminal) > 1.0e-9:
            raise ValueError("terminal_sigma differs from the declared scheduler mapping")
        if self.adversarial_scales is None:
            adversarial_scales = tuple(1.0 - value for value in sigmas)
        else:
            adversarial_scales = tuple(float(value) for value in self.adversarial_scales)
        if len(adversarial_scales) != len(sigmas) or any(
            not isfinite(value) or not 0 <= value <= 1 for value in adversarial_scales
        ):
            raise ValueError("adversarial_scales must align with anchors and lie in [0,1]")
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(self, "flow_shift", flow_shift)
        object.__setattr__(self, "terminal_sigma", terminal_sigma)
        object.__setattr__(self, "adversarial_scales", adversarial_scales)

    def _sigma_at_index(self, timestep: int) -> float:
        base = (float(timestep) + float(self.timestep_index_offset)) / float(
            self.num_train_timesteps
        )
        if not 0 <= base <= 1:
            raise ValueError("SenseFlow timestep index maps outside the scheduler sigma grid")
        return float(flow_shift_sigmas(base, self.flow_shift))

    def next_timestep(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.timesteps):
            raise ValueError("SenseFlow anchor index is out of range")
        return self.timesteps[index + 1] if index + 1 < len(self.timesteps) else self.terminal_timestep

    def next_sigma(self, index: int) -> float:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.sigmas):
            raise ValueError("SenseFlow anchor index is out of range")
        return self.sigmas[index + 1] if index + 1 < len(self.sigmas) else self.terminal_sigma

    def adversarial_scale(self, index: int) -> float:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.sigmas):
            raise ValueError("SenseFlow anchor index is out of range")
        assert self.adversarial_scales is not None
        return self.adversarial_scales[index]

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-senseflow-schedule",
                "timesteps": self.timesteps,
                "sigmas": self.sigmas,
                "isg_margin": self.isg_margin,
                "num_train_timesteps": self.num_train_timesteps,
                "flow_shift": self.flow_shift,
                "timestep_index_offset": self.timestep_index_offset,
                "terminal_timestep": self.terminal_timestep,
                "terminal_sigma": self.terminal_sigma,
                "adversarial_scales": self.adversarial_scales,
            }
        )

    @classmethod
    def sd35_released(cls) -> SenseFlowSchedule:
        """Four anchors and the 50-index exclusion used by the released SD3.5 loop."""

        return cls(
            timesteps=(999, 749, 499, 249),
            sigmas=(1.0, 0.9, 0.75, 0.5),
            isg_margin=50,
            flow_shift=3.0,
            timestep_index_offset=1,
            terminal_sigma=3.0 * 0.001 / (1.0 + 2.0 * 0.001),
            adversarial_scales=tuple(
                float(flow_shift_sigmas((1000.0 - timestep) / 1000.0, 3.0))
                for timestep in (999, 749, 499, 249)
            ),
        )

    @classmethod
    def flux_released(cls) -> SenseFlowSchedule:
        """Shifted FLUX anchors and the 20-index exclusion used by the release."""

        return cls(
            timesteps=(1000, 904, 759, 512),
            sigmas=(1.0, 0.904, 0.759, 0.512),
            isg_margin=20,
        )

    @classmethod
    def from_recipe(cls, spec: SenseFlowScheduleSpec) -> SenseFlowSchedule:
        if not isinstance(spec, SenseFlowScheduleSpec):
            raise TypeError("spec must be SenseFlowScheduleSpec")
        return cls(
            timesteps=spec.timesteps,
            sigmas=spec.sigmas,
            isg_margin=spec.isg_margin,
            num_train_timesteps=spec.num_train_timesteps,
            flow_shift=spec.flow_shift,
            timestep_index_offset=spec.timestep_index_offset,
            terminal_timestep=spec.terminal_timestep,
            terminal_sigma=spec.terminal_sigma,
            adversarial_scales=spec.adversarial_scales,
        )


@dataclass(frozen=True, slots=True)
class SenseFlowConfig:
    """All values consumed by IDA, ISG, DMD, and VFM-GAN execution."""

    schedule: SenseFlowSchedule
    generator_update_interval: int = 5
    backward_simulation_probability: float = 0.5
    ida_decay: float = 0.97
    isg_weight: float = 1.0
    isg_loss: ISGLoss = "charbonnier"
    isg_epsilon: float = 1.0e-3
    isg_teacher_guidance: tuple[float, float] = (5.0, 5.0)
    dmd_teacher_guidance: tuple[float, float] = (3.0, 10.0)
    score_sampling: ScoreSampling = "uniform-schedule-index"
    fake_score_sampling: ScoreSampling = "logit-normal-scheduler-index"
    score_min_timestep_fraction: float = 0.02
    score_max_timestep_fraction: float = 0.98
    fake_score_min_timestep_fraction: float = 0.0
    fake_score_max_timestep_fraction: float = 1.0
    score_flow_shift: float = 3.0
    normalization_epsilon: float = 0.0
    distribution_matching_weight: float = 1.0
    generator_adversarial_weight: float = 0.1
    fake_score_weight: float = 1.0
    discriminator_weight: float = 1.0
    seed: int = 71801
    student_scheduler_cadence: SchedulerCadence = "iteration"

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, SenseFlowSchedule):
            raise TypeError("schedule must be SenseFlowSchedule")
        if (
            isinstance(self.generator_update_interval, bool)
            or not isinstance(self.generator_update_interval, int)
            or self.generator_update_interval <= 0
        ):
            raise ValueError("generator_update_interval must be a positive integer")
        probability = float(self.backward_simulation_probability)
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("backward_simulation_probability must lie in [0,1]")
        decay = float(self.ida_decay)
        if not isfinite(decay) or not 0 <= decay <= 1:
            raise ValueError("ida_decay must lie in [0,1]")
        if self.isg_loss not in {"charbonnier", "mse"}:
            raise ValueError("isg_loss must be 'charbonnier' or 'mse'")
        epsilon = float(self.isg_epsilon)
        if not isfinite(epsilon) or epsilon <= 0:
            raise ValueError("isg_epsilon must be finite and positive")
        score_min = float(self.score_min_timestep_fraction)
        score_max = float(self.score_max_timestep_fraction)
        if not 0 <= score_min < score_max <= 1:
            raise ValueError("score timestep fractions must satisfy 0 <= min < max <= 1")
        fake_score_min = float(self.fake_score_min_timestep_fraction)
        fake_score_max = float(self.fake_score_max_timestep_fraction)
        if not 0 <= fake_score_min < fake_score_max <= 1:
            raise ValueError(
                "fake-score timestep fractions must satisfy 0 <= min < max <= 1"
            )
        sampling_modes = {"uniform-schedule-index", "logit-normal-scheduler-index"}
        if self.score_sampling not in sampling_modes:
            raise ValueError("unsupported score_sampling mode")
        if self.fake_score_sampling not in sampling_modes:
            raise ValueError("unsupported fake_score_sampling mode")
        score_flow_shift = float(self.score_flow_shift)
        if not isfinite(score_flow_shift) or score_flow_shift <= 0:
            raise ValueError("score_flow_shift must be finite and positive")
        normalization_epsilon = float(self.normalization_epsilon)
        if not isfinite(normalization_epsilon) or normalization_epsilon < 0:
            raise ValueError("normalization_epsilon must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.seed >= 2**63:
            raise ValueError("seed must fit the torch RNG range")
        cadence = str(self.student_scheduler_cadence).strip().lower().replace("_", "-")
        if cadence not in {"iteration", "generator-update"}:
            raise ValueError(
                "student_scheduler_cadence must be 'iteration' or 'generator-update'"
            )
        for field_name in (
            "isg_weight",
            "distribution_matching_weight",
            "generator_adversarial_weight",
            "fake_score_weight",
            "discriminator_weight",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, value)
        if self.distribution_matching_weight + self.generator_adversarial_weight + self.isg_weight <= 0:
            raise ValueError("SenseFlow generator has no enabled objective")
        if self.fake_score_weight <= 0 or self.discriminator_weight <= 0:
            raise ValueError("SenseFlow fake-score and discriminator objectives must be enabled")
        object.__setattr__(self, "backward_simulation_probability", probability)
        object.__setattr__(self, "ida_decay", decay)
        object.__setattr__(self, "isg_epsilon", epsilon)
        object.__setattr__(self, "isg_teacher_guidance", _finite_range(
            self.isg_teacher_guidance,
            field_name="isg_teacher_guidance",
        ))
        object.__setattr__(self, "dmd_teacher_guidance", _finite_range(
            self.dmd_teacher_guidance,
            field_name="dmd_teacher_guidance",
        ))
        object.__setattr__(self, "score_min_timestep_fraction", score_min)
        object.__setattr__(self, "score_max_timestep_fraction", score_max)
        object.__setattr__(self, "fake_score_min_timestep_fraction", fake_score_min)
        object.__setattr__(self, "fake_score_max_timestep_fraction", fake_score_max)
        object.__setattr__(self, "score_flow_shift", score_flow_shift)
        object.__setattr__(self, "normalization_epsilon", normalization_epsilon)
        object.__setattr__(self, "student_scheduler_cadence", cadence)

    @property
    def ida_enabled(self) -> bool:
        return self.ida_decay < 1.0

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-senseflow-config",
                "schedule_digest": self.schedule.digest,
                "generator_update_interval": self.generator_update_interval,
                "backward_simulation_probability": self.backward_simulation_probability,
                "ida_decay": self.ida_decay,
                "isg_weight": self.isg_weight,
                "isg_loss": self.isg_loss,
                "isg_epsilon": self.isg_epsilon,
                "isg_teacher_guidance": self.isg_teacher_guidance,
                "dmd_teacher_guidance": self.dmd_teacher_guidance,
                "score_sampling": self.score_sampling,
                "fake_score_sampling": self.fake_score_sampling,
                "score_min_timestep_fraction": self.score_min_timestep_fraction,
                "score_max_timestep_fraction": self.score_max_timestep_fraction,
                "fake_score_min_timestep_fraction": self.fake_score_min_timestep_fraction,
                "fake_score_max_timestep_fraction": self.fake_score_max_timestep_fraction,
                "score_flow_shift": self.score_flow_shift,
                "normalization_epsilon": self.normalization_epsilon,
                "distribution_matching_weight": self.distribution_matching_weight,
                "generator_adversarial_weight": self.generator_adversarial_weight,
                "fake_score_weight": self.fake_score_weight,
                "discriminator_weight": self.discriminator_weight,
                "seed": self.seed,
                "student_scheduler_cadence": self.student_scheduler_cadence,
            }
        )

    @classmethod
    def sd35_large_released(cls) -> SenseFlowConfig:
        """Controls from the released SD3.5 Large trainer."""

        return cls(schedule=SenseFlowSchedule.sd35_released())

    @classmethod
    def sd35_medium_released(cls) -> SenseFlowConfig:
        """Controls from the released SD3.5 Medium trainer."""

        return cls(
            schedule=SenseFlowSchedule.sd35_released(),
            generator_update_interval=10,
            ida_decay=0.98,
            isg_weight=0.5,
            isg_teacher_guidance=(2.0, 4.0),
            dmd_teacher_guidance=(2.0, 8.0),
        )

    @classmethod
    def flux_released(cls) -> SenseFlowConfig:
        return cls(
            schedule=SenseFlowSchedule.flux_released(),
            isg_teacher_guidance=(1.0, 8.0),
            dmd_teacher_guidance=(1.0, 8.0),
            score_sampling="logit-normal-scheduler-index",
            score_min_timestep_fraction=0.0,
            score_max_timestep_fraction=1.0,
            score_flow_shift=1.0,
            generator_adversarial_weight=2.0,
        )

    @classmethod
    def from_recipe(cls, spec: SenseFlowAlgorithmSpec) -> SenseFlowConfig:
        if not isinstance(spec, SenseFlowAlgorithmSpec):
            raise TypeError("spec must be SenseFlowAlgorithmSpec")
        return cls(
            schedule=SenseFlowSchedule.from_recipe(spec.schedule),
            generator_update_interval=spec.generator_update_interval,
            backward_simulation_probability=spec.backward_simulation_probability,
            ida_decay=spec.ida_decay,
            isg_weight=spec.isg_weight,
            isg_loss=spec.isg_loss,
            isg_epsilon=spec.isg_epsilon,
            isg_teacher_guidance=spec.isg_teacher_guidance,
            dmd_teacher_guidance=spec.dmd_teacher_guidance,
            score_sampling=spec.score_sampling,
            fake_score_sampling=spec.fake_score_sampling,
            score_min_timestep_fraction=spec.score_min_timestep_fraction,
            score_max_timestep_fraction=spec.score_max_timestep_fraction,
            fake_score_min_timestep_fraction=spec.fake_score_min_timestep_fraction,
            fake_score_max_timestep_fraction=spec.fake_score_max_timestep_fraction,
            score_flow_shift=spec.score_flow_shift,
            normalization_epsilon=spec.normalization_epsilon,
            distribution_matching_weight=spec.distribution_matching_weight,
            generator_adversarial_weight=spec.generator_adversarial_weight,
            fake_score_weight=spec.fake_score_weight,
            discriminator_weight=spec.discriminator_weight,
            seed=spec.seed,
            student_scheduler_cadence=spec.student_scheduler_cadence,
        )


@dataclass(frozen=True, slots=True)
class SenseFlowOptimizerConfig:
    """Three AdamW roles and the released linear-warmup/constant schedule."""

    student_learning_rate: float = 1.0e-6
    fake_score_learning_rate: float = 1.0e-6
    discriminator_learning_rate: float = 1.0e-6
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1.0e-8
    weight_decay: float = 0.0
    student_max_grad_norm: float = 1.0
    fake_score_max_grad_norm: float = 1.0
    discriminator_max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 500
    warmup_start_ratio: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "student_learning_rate",
            "fake_score_learning_rate",
            "discriminator_learning_rate",
            "epsilon",
            "student_max_grad_norm",
            "fake_score_max_grad_norm",
            "discriminator_max_grad_norm",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
            object.__setattr__(self, field_name, value)
        decay = float(self.weight_decay)
        if not isfinite(decay) or decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not isinstance(self.betas, tuple) or len(self.betas) != 2:
            raise TypeError("betas must be a two-value tuple")
        betas = tuple(float(value) for value in self.betas)
        if any(not isfinite(value) or not 0 <= value < 1 for value in betas):
            raise ValueError("betas must be finite and lie in [0,1)")
        if (
            isinstance(self.gradient_accumulation_steps, bool)
            or not isinstance(self.gradient_accumulation_steps, int)
            or self.gradient_accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or self.warmup_steps < 0
        ):
            raise ValueError("warmup_steps must be a non-negative integer")
        start_ratio = float(self.warmup_start_ratio)
        if not isfinite(start_ratio) or not 0 < start_ratio <= 1:
            raise ValueError("warmup_start_ratio must lie in (0,1]")
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "weight_decay", decay)
        object.__setattr__(self, "warmup_start_ratio", start_ratio)

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-senseflow-optimizer-config",
                "student_learning_rate": self.student_learning_rate,
                "fake_score_learning_rate": self.fake_score_learning_rate,
                "discriminator_learning_rate": self.discriminator_learning_rate,
                "betas": self.betas,
                "epsilon": self.epsilon,
                "weight_decay": self.weight_decay,
                "student_max_grad_norm": self.student_max_grad_norm,
                "fake_score_max_grad_norm": self.fake_score_max_grad_norm,
                "discriminator_max_grad_norm": self.discriminator_max_grad_norm,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "warmup_steps": self.warmup_steps,
                "warmup_start_ratio": self.warmup_start_ratio,
            }
        )

    @classmethod
    def sd35_released(cls) -> SenseFlowOptimizerConfig:
        return cls()

    @classmethod
    def flux_released(cls) -> SenseFlowOptimizerConfig:
        return cls(
            student_learning_rate=1.0e-5,
            fake_score_learning_rate=1.0e-5,
            discriminator_learning_rate=1.0e-5,
            warmup_start_ratio=0.5,
        )

    @classmethod
    def from_recipe(
        cls,
        algorithm: SenseFlowAlgorithmSpec,
        student: OptimizerSpec,
        fake_score: OptimizerSpec,
        discriminator: OptimizerSpec,
    ) -> SenseFlowOptimizerConfig:
        if not isinstance(algorithm, SenseFlowAlgorithmSpec):
            raise TypeError("algorithm must be SenseFlowAlgorithmSpec")
        optimizers = (student, fake_score, discriminator)
        if not all(isinstance(value, OptimizerSpec) for value in optimizers):
            raise TypeError("SenseFlow optimizer sections must be OptimizerSpec values")
        if any(value.type != "adamw" for value in optimizers):
            raise ValueError("SenseFlow released training requires AdamW for every role")
        accumulation = student.gradient_accumulation_steps
        if any(value.gradient_accumulation_steps != accumulation for value in optimizers[1:]):
            raise ValueError("SenseFlow optimizer gradient accumulation steps must match")
        for field_name in ("betas", "epsilon", "weight_decay"):
            expected = getattr(student, field_name)
            if any(getattr(value, field_name) != expected for value in optimizers[1:]):
                raise ValueError(f"SenseFlow optimizer {field_name} values must match")
        if not isinstance(student.epsilon, float) or len(student.betas) != 2:
            raise RuntimeError("validated SenseFlow AdamW fields have an invalid shape")
        if algorithm.preset != "custom":
            expected_learning_rate = (
                1.0e-5 if algorithm.preset == "flux-released" else 1.0e-6
            )
            mismatches: list[str] = []
            for role, optimizer in zip(
                ("student", "fake-score", "discriminator"),
                optimizers,
                strict=True,
            ):
                if optimizer.learning_rate != expected_learning_rate:
                    mismatches.append(f"{role}.learning_rate")
                if optimizer.max_grad_norm != 1.0:
                    mismatches.append(f"{role}.max_grad_norm")
            if student.betas != (0.9, 0.999):
                mismatches.append("betas")
            if student.epsilon != 1.0e-8:
                mismatches.append("epsilon")
            if student.weight_decay != 0.0:
                mismatches.append("weight_decay")
            if mismatches:
                raise ValueError(
                    f"SenseFlow {algorithm.preset} optimizer fields differ from the "
                    f"released preset: {mismatches}"
                )
        return cls(
            student_learning_rate=student.learning_rate,
            fake_score_learning_rate=fake_score.learning_rate,
            discriminator_learning_rate=discriminator.learning_rate,
            betas=(student.betas[0], student.betas[1]),
            epsilon=student.epsilon,
            weight_decay=student.weight_decay,
            student_max_grad_norm=student.max_grad_norm,
            fake_score_max_grad_norm=fake_score.max_grad_norm,
            discriminator_max_grad_norm=discriminator.max_grad_norm,
            gradient_accumulation_steps=accumulation,
            warmup_steps=algorithm.lr_warmup_steps,
            warmup_start_ratio=algorithm.lr_warmup_start_ratio,
        )


__all__ = [
    "ISGLoss",
    "SchedulerCadence",
    "ScoreSampling",
    "SenseFlowConfig",
    "SenseFlowOptimizerConfig",
    "SenseFlowSchedule",
]
