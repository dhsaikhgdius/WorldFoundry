"""Runtime configuration for native diagonal distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from worldfoundry.core.io.integrity import canonical_sha256

from ..dmd.objective import DMDConfig, FewStepSchedule
from ..self_forcing.config import shifted_few_step_schedule

ExitStepMode = Literal["sequence", "block"]
RegressionLossType = Literal["mse", "charbonnier", "cauchy"]


def build_block_denoising_steps(
    base_schedule: FewStepSchedule,
    *,
    block_index: int,
    use_diagonal_denoising: bool,
    warmup_mid_schedule: FewStepSchedule | None = None,
) -> FewStepSchedule:
    """Build the released 4/3/2 denoising schedule for one causal block."""

    if not isinstance(base_schedule, FewStepSchedule):
        raise TypeError("base_schedule must be a FewStepSchedule")
    if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 0:
        raise ValueError("block_index must be a non-negative integer")
    if not isinstance(use_diagonal_denoising, bool):
        raise TypeError("use_diagonal_denoising must be bool")
    if warmup_mid_schedule is not None and not isinstance(warmup_mid_schedule, FewStepSchedule):
        raise TypeError("warmup_mid_schedule must be a FewStepSchedule or None")
    if not use_diagonal_denoising or len(base_schedule.timesteps) <= 1:
        return base_schedule

    first_timestep = base_schedule.timesteps[:1]
    first_sigma = base_schedule.sigmas[:1]
    last_timestep = base_schedule.timesteps[-1:]
    last_sigma = base_schedule.sigmas[-1:]
    if warmup_mid_schedule is None:
        mid_timesteps = base_schedule.timesteps[1:-1]
        mid_sigmas = base_schedule.sigmas[1:-1]
    else:
        mid_timesteps = warmup_mid_schedule.timesteps
        mid_sigmas = warmup_mid_schedule.sigmas

    mid_count = 2 if block_index == 0 else 1 if block_index == 1 else 0
    return FewStepSchedule(
        timesteps=first_timestep + mid_timesteps[:mid_count] + last_timestep,
        sigmas=first_sigma + mid_sigmas[:mid_count] + last_sigma,
    )


@dataclass(frozen=True, slots=True)
class DiagonalScheduleConfig:
    """Fields consumed by diagonal causal rollout and cache refresh."""

    base_schedule: FewStepSchedule
    frames_per_block: int
    frame_dim: int = 2
    use_diagonal_denoising: bool = True
    warmup_mid_schedule: FewStepSchedule | None = None
    context_timestep: float = 0.0
    context_sigma: float = 0.0
    exit_step_mode: ExitStepMode = "block"
    last_step_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base_schedule, FewStepSchedule):
            raise TypeError("base_schedule must be a FewStepSchedule")
        if isinstance(self.frames_per_block, bool) or int(self.frames_per_block) <= 0:
            raise ValueError("frames_per_block must be a positive integer")
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")
        if not isinstance(self.use_diagonal_denoising, bool):
            raise TypeError("use_diagonal_denoising must be bool")
        if self.warmup_mid_schedule is not None and not isinstance(
            self.warmup_mid_schedule,
            FewStepSchedule,
        ):
            raise TypeError("warmup_mid_schedule must be a FewStepSchedule or None")
        context_timestep = float(self.context_timestep)
        context_sigma = float(self.context_sigma)
        if not isfinite(context_timestep) or context_timestep < 0:
            raise ValueError("context_timestep must be finite and non-negative")
        if not isfinite(context_sigma) or not 0 <= context_sigma <= 1:
            raise ValueError("context_sigma must be finite and in [0,1]")
        mode = str(self.exit_step_mode).strip().lower().replace("_", "-")
        if mode not in {"sequence", "block"}:
            raise ValueError("exit_step_mode must be 'sequence' or 'block'")
        if not isinstance(self.last_step_only, bool):
            raise TypeError("last_step_only must be bool")
        object.__setattr__(self, "frames_per_block", int(self.frames_per_block))
        object.__setattr__(self, "context_timestep", context_timestep)
        object.__setattr__(self, "context_sigma", context_sigma)
        object.__setattr__(self, "exit_step_mode", mode)

    def block_schedule(self, block_index: int) -> FewStepSchedule:
        return build_block_denoising_steps(
            self.base_schedule,
            block_index=block_index,
            use_diagonal_denoising=self.use_diagonal_denoising,
            warmup_mid_schedule=self.warmup_mid_schedule,
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-diagonal-schedule",
                "base_schedule_digest": self.base_schedule.digest,
                "frames_per_block": self.frames_per_block,
                "frame_dim": self.frame_dim,
                "use_diagonal_denoising": self.use_diagonal_denoising,
                "warmup_mid_schedule_digest": (
                    None if self.warmup_mid_schedule is None else self.warmup_mid_schedule.digest
                ),
                "context_timestep": self.context_timestep,
                "context_sigma": self.context_sigma,
                "exit_step_mode": self.exit_step_mode,
                "last_step_only": self.last_step_only,
            }
        )

    @classmethod
    def from_raw_timesteps(
        cls,
        base_timesteps: tuple[float, ...],
        *,
        frames_per_block: int,
        warmup_mid_timesteps: tuple[float, ...] = (),
        num_train_timesteps: int = 1000,
        flow_shift: float = 5.0,
        frame_dim: int = 2,
        use_diagonal_denoising: bool,
        context_timestep: float = 100.0,
        exit_step_mode: ExitStepMode = "block",
        last_step_only: bool = False,
    ) -> DiagonalScheduleConfig:
        base_schedule = shifted_few_step_schedule(
            base_timesteps,
            num_train_timesteps=num_train_timesteps,
            flow_shift=flow_shift,
        )
        mids = (
            shifted_few_step_schedule(
                warmup_mid_timesteps,
                num_train_timesteps=num_train_timesteps,
                flow_shift=flow_shift,
            )
            if warmup_mid_timesteps
            else None
        )
        effective_context = float(context_timestep)
        if not isfinite(effective_context) or not 0 <= effective_context <= num_train_timesteps:
            raise ValueError("context_timestep must lie on the effective scheduler timeline")
        context_sigma = effective_context / float(num_train_timesteps)
        return cls(
            base_schedule=base_schedule,
            frames_per_block=frames_per_block,
            frame_dim=frame_dim,
            use_diagonal_denoising=use_diagonal_denoising,
            warmup_mid_schedule=mids,
            context_timestep=effective_context,
            context_sigma=context_sigma,
            exit_step_mode=exit_step_mode,
            last_step_only=last_step_only,
        )

    @classmethod
    def stage_one(
        cls,
        *,
        frames_per_block: int = 3,
        frame_dim: int = 2,
        exit_step_mode: ExitStepMode = "sequence",
        last_step_only: bool = False,
    ) -> DiagonalScheduleConfig:
        """Released all-four-step base-DMD schedule."""

        return cls.from_raw_timesteps(
            (1000.0, 750.0, 500.0, 100.0),
            frames_per_block=frames_per_block,
            frame_dim=frame_dim,
            use_diagonal_denoising=False,
            exit_step_mode=exit_step_mode,
            last_step_only=last_step_only,
        )

    @classmethod
    def stage_two(
        cls,
        *,
        frames_per_block: int = 3,
        frame_dim: int = 2,
        exit_step_mode: ExitStepMode = "sequence",
        last_step_only: bool = False,
    ) -> DiagonalScheduleConfig:
        """Released two-step base schedule with 750/500 warm-up steps."""

        return cls.from_raw_timesteps(
            (1000.0, 100.0),
            frames_per_block=frames_per_block,
            frame_dim=frame_dim,
            warmup_mid_timesteps=(750.0, 500.0),
            use_diagonal_denoising=True,
            exit_step_mode=exit_step_mode,
            last_step_only=last_step_only,
        )

    @classmethod
    def fixed_teacher(
        cls,
        *,
        frames_per_block: int = 3,
        frame_dim: int = 2,
    ) -> DiagonalScheduleConfig:
        """Released frozen four-step regression-teacher trajectory."""

        return cls.from_raw_timesteps(
            (1000.0, 750.0, 500.0, 250.0),
            frames_per_block=frames_per_block,
            frame_dim=frame_dim,
            use_diagonal_denoising=False,
            exit_step_mode="block",
            last_step_only=True,
        )


@dataclass(frozen=True, slots=True)
class DiagonalObjectiveConfig:
    """Official compound spatial, temporal, and regression loss controls."""

    dmd: DMDConfig
    frame_dim: int = 2
    use_motion_loss: bool = True
    use_flow_reg_loss: bool = True
    flow_reg_ema_decay: float = 0.95
    lambda_spatial_dmd: float = 4.0
    lambda_flow_dmd: float = 4.0
    gamma_temporal: float = 1.0
    lambda_reg: float = 0.0
    regression_loss_type: RegressionLossType = "mse"
    regression_epsilon: float = 1.0e-3
    regression_cauchy_scale: float = 1.0e-2
    use_teacher_regression: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.dmd, DMDConfig):
            raise TypeError("dmd must be DMDConfig")
        if not self.dmd.per_sample_normalization:
            raise ValueError("diagonal DMD requires per-sample gradient normalization")
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")
        for name in ("use_motion_loss", "use_flow_reg_loss", "use_teacher_regression"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        values: dict[str, float] = {}
        for name in (
            "flow_reg_ema_decay",
            "lambda_spatial_dmd",
            "lambda_flow_dmd",
            "gamma_temporal",
            "lambda_reg",
            "regression_epsilon",
            "regression_cauchy_scale",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            values[name] = value
        if not 0 <= values["flow_reg_ema_decay"] < 1:
            raise ValueError("flow_reg_ema_decay must be in [0,1)")
        if any(values[name] < 0 for name in ("lambda_spatial_dmd", "lambda_flow_dmd", "gamma_temporal", "lambda_reg")):
            raise ValueError("loss weights must be non-negative")
        if values["regression_epsilon"] <= 0 or values["regression_cauchy_scale"] <= 0:
            raise ValueError("regression scales must be positive")
        loss_type = str(self.regression_loss_type).strip().lower().replace("_", "-")
        if loss_type not in {"mse", "charbonnier", "cauchy"}:
            raise ValueError("regression_loss_type must be mse, charbonnier, or cauchy")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "regression_loss_type", loss_type)

    @classmethod
    def released(cls, schedule: DiagonalScheduleConfig) -> DiagonalObjectiveConfig:
        """Build the compound objective used by the released two-stage runs."""

        if not isinstance(schedule, DiagonalScheduleConfig):
            raise TypeError("schedule must be DiagonalScheduleConfig")
        return cls(
            dmd=DMDConfig(
                schedule=schedule.base_schedule,
                num_train_timesteps=1000,
                score_min_sigma=0.02,
                score_max_sigma=0.98,
                score_flow_shift=5.0,
                teacher_guidance_scale=3.0,
                normalization_epsilon=0.0,
                # The released ``uniform_timestep=True`` samples one value per
                # video and repeats it over frames; videos in a batch remain
                # independent.
                shared_score_timestep=False,
                per_sample_normalization=True,
            ),
            frame_dim=schedule.frame_dim,
            use_motion_loss=True,
            use_flow_reg_loss=True,
            flow_reg_ema_decay=0.95,
            lambda_spatial_dmd=4.0,
            lambda_flow_dmd=4.0,
            gamma_temporal=1.0,
            lambda_reg=0.0,
            regression_loss_type="mse",
            regression_epsilon=1.0e-3,
            regression_cauchy_scale=1.0e-2,
            use_teacher_regression=True,
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-diagonal-objective",
                "dmd_digest": self.dmd.digest,
                "frame_dim": self.frame_dim,
                "use_motion_loss": self.use_motion_loss,
                "use_flow_reg_loss": self.use_flow_reg_loss,
                "flow_reg_ema_decay": self.flow_reg_ema_decay,
                "lambda_spatial_dmd": self.lambda_spatial_dmd,
                "lambda_flow_dmd": self.lambda_flow_dmd,
                "gamma_temporal": self.gamma_temporal,
                "lambda_reg": self.lambda_reg,
                "regression_loss_type": self.regression_loss_type,
                "regression_epsilon": self.regression_epsilon,
                "regression_cauchy_scale": self.regression_cauchy_scale,
                "use_teacher_regression": self.use_teacher_regression,
            }
        )


__all__ = [
    "DiagonalObjectiveConfig",
    "DiagonalScheduleConfig",
    "ExitStepMode",
    "RegressionLossType",
    "build_block_denoising_steps",
]
