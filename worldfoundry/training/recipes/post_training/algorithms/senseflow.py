"""Strict recipe contract for native SenseFlow distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import strict_mapping
from .auxiliary_optimizers import (
    AuxiliaryOptimizerRule,
    forbids_auxiliary,
    requires_auxiliary,
)


def _finite(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _range(value: object, *, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{field_name} must contain two values")
    lower, upper = (_finite(item, field_name=field_name) for item in value)
    if lower < 0 or lower > upper:
        raise ValueError(f"{field_name} must satisfy 0 <= lower <= upper")
    return lower, upper


def _flow_shift(value: float, shift: float) -> float:
    return shift * value / (1.0 + (shift - 1.0) * value)


@dataclass(frozen=True, slots=True)
class SenseFlowScheduleSpec:
    """Coarse anchors and their exact flow-scheduler mapping."""

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
        timesteps = tuple(self.timesteps)
        sigmas = tuple(_finite(value, field_name="schedule.sigmas") for value in self.sigmas)
        if not timesteps or len(timesteps) != len(sigmas):
            raise ValueError("SenseFlow schedule timesteps and sigmas must be aligned")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in timesteps):
            raise TypeError("SenseFlow schedule timesteps must be integers")
        if any(value <= 0 for value in timesteps) or any(
            left <= right for left, right in zip(timesteps, timesteps[1:])
        ):
            raise ValueError("SenseFlow schedule timesteps must be positive and descending")
        if any(not 0 < value <= 1 for value in sigmas) or any(
            left <= right for left, right in zip(sigmas, sigmas[1:])
        ):
            raise ValueError("SenseFlow schedule sigmas must lie in (0,1] and descend")
        margin = self.isg_margin
        if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
            raise ValueError("schedule.isg_margin must be a non-negative integer")
        train_steps = _positive_int(
            self.num_train_timesteps,
            field_name="schedule.num_train_timesteps",
        )
        if train_steps < 2:
            raise ValueError("schedule.num_train_timesteps must be at least two")
        shift = _finite(self.flow_shift, field_name="schedule.flow_shift")
        if shift <= 0:
            raise ValueError("schedule.flow_shift must be positive")
        offset = self.timestep_index_offset
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("schedule.timestep_index_offset must be a non-negative integer")
        terminal_timestep = self.terminal_timestep
        if (
            isinstance(terminal_timestep, bool)
            or not isinstance(terminal_timestep, int)
            or terminal_timestep < 0
            or terminal_timestep >= timesteps[-1]
        ):
            raise ValueError("schedule.terminal_timestep must be below the last anchor")
        terminal_sigma = _finite(
            self.terminal_sigma,
            field_name="schedule.terminal_sigma",
        )
        if not 0 <= terminal_sigma < sigmas[-1]:
            raise ValueError("schedule.terminal_sigma must be below the last anchor sigma")

        def sigma_at(timestep: int) -> float:
            base = (float(timestep) + float(offset)) / float(train_steps)
            if not 0 <= base <= 1:
                raise ValueError("SenseFlow schedule maps outside the scheduler grid")
            return _flow_shift(base, shift)

        expected_sigmas = tuple(sigma_at(timestep) for timestep in timesteps)
        if any(
            abs(actual - expected) > 1.0e-9
            for actual, expected in zip(sigmas, expected_sigmas, strict=True)
        ):
            raise ValueError("SenseFlow schedule sigmas differ from its scheduler mapping")
        if abs(terminal_sigma - sigma_at(terminal_timestep)) > 1.0e-9:
            raise ValueError("SenseFlow terminal sigma differs from its scheduler mapping")
        for index, timestep in enumerate(timesteps):
            next_timestep = (
                timesteps[index + 1]
                if index + 1 < len(timesteps)
                else terminal_timestep
            )
            if next_timestep + margin > timestep - margin:
                raise ValueError("every SenseFlow segment must contain an ISG midpoint")
        adversarial_scales = (
            tuple(1.0 - value for value in sigmas)
            if self.adversarial_scales is None
            else tuple(
                _finite(value, field_name="schedule.adversarial_scales")
                for value in self.adversarial_scales
            )
        )
        if len(adversarial_scales) != len(sigmas) or any(
            not 0 <= value <= 1 for value in adversarial_scales
        ):
            raise ValueError("schedule.adversarial_scales must align and lie in [0,1]")
        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(self, "num_train_timesteps", train_steps)
        object.__setattr__(self, "flow_shift", shift)
        object.__setattr__(self, "terminal_sigma", terminal_sigma)
        object.__setattr__(self, "adversarial_scales", adversarial_scales)

    @classmethod
    def sd35_released(cls) -> SenseFlowScheduleSpec:
        return cls(
            timesteps=(999, 749, 499, 249),
            sigmas=(1.0, 0.9, 0.75, 0.5),
            isg_margin=50,
            flow_shift=3.0,
            timestep_index_offset=1,
            terminal_sigma=3.0 * 0.001 / (1.0 + 2.0 * 0.001),
            adversarial_scales=tuple(
                _flow_shift((1000.0 - timestep) / 1000.0, 3.0)
                for timestep in (999, 749, 499, 249)
            ),
        )

    @classmethod
    def flux_released(cls) -> SenseFlowScheduleSpec:
        return cls(
            timesteps=(1000, 904, 759, 512),
            sigmas=(1.0, 0.904, 0.759, 0.512),
            isg_margin=20,
        )


@dataclass(frozen=True, slots=True)
class SenseFlowAlgorithmSpec:
    """Every role, equation, cadence, and scheduler choice used by SenseFlow."""

    schedule: SenseFlowScheduleSpec
    teacher_checkpoint: str
    fake_score_checkpoint: str
    discriminator_checkpoint: str
    generator_update_interval: int = 5
    backward_simulation_probability: float = 0.5
    ida_decay: float = 0.97
    isg_weight: float = 1.0
    isg_loss: str = "charbonnier"
    isg_epsilon: float = 1.0e-3
    isg_teacher_guidance: tuple[float, float] = (5.0, 5.0)
    dmd_teacher_guidance: tuple[float, float] = (3.0, 10.0)
    score_sampling: str = "uniform-schedule-index"
    fake_score_sampling: str = "logit-normal-scheduler-index"
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
    student_scheduler_cadence: str = "iteration"
    lr_warmup_steps: int = 500
    lr_warmup_start_ratio: float = 1.0
    preset: str = "custom"
    type: str = "senseflow"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "senseflow":
            raise ValueError("SenseFlow algorithm type must be 'senseflow'")
        if not isinstance(self.schedule, SenseFlowScheduleSpec):
            raise TypeError("schedule must be SenseFlowScheduleSpec")
        for name in (
            "teacher_checkpoint",
            "fake_score_checkpoint",
            "discriminator_checkpoint",
        ):
            raw = getattr(self, name)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"{name} must be a non-empty checkpoint reference")
            value = raw.strip()
            object.__setattr__(self, name, value)
        interval = _positive_int(
            self.generator_update_interval,
            field_name="algorithm.generator_update_interval",
        )
        probability = _finite(
            self.backward_simulation_probability,
            field_name="backward_simulation_probability",
        )
        decay = _finite(self.ida_decay, field_name="ida_decay")
        if not 0 <= probability <= 1 or not 0 <= decay <= 1:
            raise ValueError("SenseFlow probability and IDA decay must lie in [0,1]")
        isg_loss = str(self.isg_loss).strip().lower().replace("_", "-")
        if isg_loss not in {"charbonnier", "mse"}:
            raise ValueError("isg_loss must be 'charbonnier' or 'mse'")
        sampling_modes = {"uniform-schedule-index", "logit-normal-scheduler-index"}
        score_sampling = str(self.score_sampling).strip().lower().replace("_", "-")
        fake_sampling = str(self.fake_score_sampling).strip().lower().replace("_", "-")
        if score_sampling not in sampling_modes or fake_sampling not in sampling_modes:
            raise ValueError("SenseFlow score sampling mode is unsupported")
        cadence = str(self.student_scheduler_cadence).strip().lower().replace("_", "-")
        if cadence not in {"iteration", "generator-update"}:
            raise ValueError(
                "student_scheduler_cadence must be 'iteration' or 'generator-update'"
            )
        preset = str(self.preset).strip().lower().replace("_", "-")
        if preset not in {
            "custom",
            "sd35-large-released",
            "sd35-medium-released",
            "flux-released",
        }:
            raise ValueError("unsupported SenseFlow preset")
        score_min = _finite(
            self.score_min_timestep_fraction,
            field_name="score_min_timestep_fraction",
        )
        score_max = _finite(
            self.score_max_timestep_fraction,
            field_name="score_max_timestep_fraction",
        )
        fake_min = _finite(
            self.fake_score_min_timestep_fraction,
            field_name="fake_score_min_timestep_fraction",
        )
        fake_max = _finite(
            self.fake_score_max_timestep_fraction,
            field_name="fake_score_max_timestep_fraction",
        )
        if not 0 <= score_min < score_max <= 1:
            raise ValueError("score timestep fractions must satisfy 0 <= min < max <= 1")
        if not 0 <= fake_min < fake_max <= 1:
            raise ValueError(
                "fake-score timestep fractions must satisfy 0 <= min < max <= 1"
            )
        positive: dict[str, float] = {}
        for name in ("isg_epsilon", "score_flow_shift"):
            value = _finite(getattr(self, name), field_name=name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            positive[name] = value
        non_negative: dict[str, float] = {}
        for name in (
            "isg_weight",
            "normalization_epsilon",
            "distribution_matching_weight",
            "generator_adversarial_weight",
            "fake_score_weight",
            "discriminator_weight",
        ):
            value = _finite(getattr(self, name), field_name=name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            non_negative[name] = value
        if (
            non_negative["isg_weight"]
            + non_negative["distribution_matching_weight"]
            + non_negative["generator_adversarial_weight"]
            <= 0
        ):
            raise ValueError("SenseFlow generator has no enabled objective")
        if non_negative["fake_score_weight"] <= 0 or non_negative["discriminator_weight"] <= 0:
            raise ValueError("SenseFlow fake-score and discriminator losses must be enabled")
        seed = self.seed
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be a non-negative torch RNG integer")
        warmup_steps = self.lr_warmup_steps
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be a non-negative integer")
        warmup_ratio = _finite(
            self.lr_warmup_start_ratio,
            field_name="lr_warmup_start_ratio",
        )
        if not 0 < warmup_ratio <= 1:
            raise ValueError("lr_warmup_start_ratio must lie in (0,1]")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "generator_update_interval", interval)
        object.__setattr__(self, "backward_simulation_probability", probability)
        object.__setattr__(self, "ida_decay", decay)
        object.__setattr__(self, "isg_loss", isg_loss)
        object.__setattr__(
            self,
            "isg_teacher_guidance",
            _range(self.isg_teacher_guidance, field_name="isg_teacher_guidance"),
        )
        object.__setattr__(
            self,
            "dmd_teacher_guidance",
            _range(self.dmd_teacher_guidance, field_name="dmd_teacher_guidance"),
        )
        object.__setattr__(self, "score_sampling", score_sampling)
        object.__setattr__(self, "fake_score_sampling", fake_sampling)
        object.__setattr__(self, "student_scheduler_cadence", cadence)
        object.__setattr__(self, "score_min_timestep_fraction", score_min)
        object.__setattr__(self, "score_max_timestep_fraction", score_max)
        object.__setattr__(self, "fake_score_min_timestep_fraction", fake_min)
        object.__setattr__(self, "fake_score_max_timestep_fraction", fake_max)
        object.__setattr__(self, "lr_warmup_start_ratio", warmup_ratio)
        object.__setattr__(self, "preset", preset)
        for name, value in {**positive, **non_negative}.items():
            object.__setattr__(self, name, value)
        if preset != "custom":
            reference = _released_reference(
                preset,
                teacher_checkpoint=self.teacher_checkpoint,
                fake_score_checkpoint=self.fake_score_checkpoint,
                discriminator_checkpoint=self.discriminator_checkpoint,
            )
            excluded = {
                "teacher_checkpoint",
                "fake_score_checkpoint",
                "discriminator_checkpoint",
                "preset",
                "type",
            }
            mismatches = [
                name
                for name in self.__dataclass_fields__
                if name not in excluded and getattr(self, name) != getattr(reference, name)
            ]
            if mismatches:
                raise ValueError(
                    f"SenseFlow {preset} fields differ from the released preset: {mismatches}"
                )

    def auxiliary_optimizer_rules(self) -> tuple[AuxiliaryOptimizerRule, ...]:
        return (
            requires_auxiliary("fake_score_optimizer", "SenseFlow requires fake_score_optimizer"),
            requires_auxiliary("discriminator_optimizer", "SenseFlow requires discriminator_optimizer"),
            forbids_auxiliary(
                "guidance_optimizer",
                message="SenseFlow does not accept guidance_optimizer",
            ),
        )

    @classmethod
    def sd35_large_released(
        cls,
        *,
        teacher_checkpoint: str,
        fake_score_checkpoint: str,
        discriminator_checkpoint: str,
    ) -> SenseFlowAlgorithmSpec:
        return cls(
            schedule=SenseFlowScheduleSpec.sd35_released(),
            teacher_checkpoint=teacher_checkpoint,
            fake_score_checkpoint=fake_score_checkpoint,
            discriminator_checkpoint=discriminator_checkpoint,
            preset="sd35-large-released",
        )

    @classmethod
    def sd35_medium_released(
        cls,
        *,
        teacher_checkpoint: str,
        fake_score_checkpoint: str,
        discriminator_checkpoint: str,
    ) -> SenseFlowAlgorithmSpec:
        return cls(
            schedule=SenseFlowScheduleSpec.sd35_released(),
            teacher_checkpoint=teacher_checkpoint,
            fake_score_checkpoint=fake_score_checkpoint,
            discriminator_checkpoint=discriminator_checkpoint,
            generator_update_interval=10,
            ida_decay=0.98,
            isg_weight=0.5,
            isg_teacher_guidance=(2.0, 4.0),
            dmd_teacher_guidance=(2.0, 8.0),
            preset="sd35-medium-released",
        )

    @classmethod
    def flux_released(
        cls,
        *,
        teacher_checkpoint: str,
        fake_score_checkpoint: str,
        discriminator_checkpoint: str,
    ) -> SenseFlowAlgorithmSpec:
        return cls(
            schedule=SenseFlowScheduleSpec.flux_released(),
            teacher_checkpoint=teacher_checkpoint,
            fake_score_checkpoint=fake_score_checkpoint,
            discriminator_checkpoint=discriminator_checkpoint,
            isg_teacher_guidance=(1.0, 8.0),
            dmd_teacher_guidance=(1.0, 8.0),
            score_sampling="logit-normal-scheduler-index",
            score_min_timestep_fraction=0.0,
            score_max_timestep_fraction=1.0,
            score_flow_shift=1.0,
            generator_adversarial_weight=2.0,
            lr_warmup_start_ratio=0.5,
            preset="flux-released",
        )


def _released_reference(
    preset: str,
    *,
    teacher_checkpoint: str,
    fake_score_checkpoint: str,
    discriminator_checkpoint: str,
) -> SenseFlowAlgorithmSpec:
    common = {
        "teacher_checkpoint": teacher_checkpoint,
        "fake_score_checkpoint": fake_score_checkpoint,
        "discriminator_checkpoint": discriminator_checkpoint,
        "preset": "custom",
    }
    if preset == "sd35-large-released":
        return SenseFlowAlgorithmSpec(
            schedule=SenseFlowScheduleSpec.sd35_released(),
            **common,
        )
    if preset == "sd35-medium-released":
        return SenseFlowAlgorithmSpec(
            schedule=SenseFlowScheduleSpec.sd35_released(),
            generator_update_interval=10,
            ida_decay=0.98,
            isg_weight=0.5,
            isg_teacher_guidance=(2.0, 4.0),
            dmd_teacher_guidance=(2.0, 8.0),
            **common,
        )
    if preset == "flux-released":
        return SenseFlowAlgorithmSpec(
            schedule=SenseFlowScheduleSpec.flux_released(),
            isg_teacher_guidance=(1.0, 8.0),
            dmd_teacher_guidance=(1.0, 8.0),
            score_sampling="logit-normal-scheduler-index",
            score_min_timestep_fraction=0.0,
            score_max_timestep_fraction=1.0,
            score_flow_shift=1.0,
            generator_adversarial_weight=2.0,
            lr_warmup_start_ratio=0.5,
            **common,
        )
    raise RuntimeError(f"unsupported released SenseFlow preset: {preset!r}")


SENSEFLOW_SCHEDULE_FIELDS = frozenset(SenseFlowScheduleSpec.__dataclass_fields__)
SENSEFLOW_ALGORITHM_FIELDS = frozenset(SenseFlowAlgorithmSpec.__dataclass_fields__)


def parse_senseflow_algorithm(value: object) -> SenseFlowAlgorithmSpec:
    payload = strict_mapping(
        value,
        field_name="algorithm",
        allowed=set(SENSEFLOW_ALGORITHM_FIELDS),
    )
    schedule = payload.get("schedule")
    if not isinstance(schedule, SenseFlowScheduleSpec):
        schedule = SenseFlowScheduleSpec(
            **strict_mapping(
                schedule,
                field_name="algorithm.schedule",
                allowed=set(SENSEFLOW_SCHEDULE_FIELDS),
            )
        )
    payload["schedule"] = schedule
    return SenseFlowAlgorithmSpec(**payload)


__all__ = [
    "SenseFlowAlgorithmSpec",
    "SenseFlowScheduleSpec",
    "parse_senseflow_algorithm",
]
