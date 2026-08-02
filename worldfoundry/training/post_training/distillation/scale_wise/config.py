"""Execution configuration for scale-wise flow distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.objectives.flow_matching import flow_match_solver_sigmas
from worldfoundry.training.recipes.post_training.algorithms.scale_wise import (
    ScaleWiseAlgorithmSpec,
)


@dataclass(frozen=True, slots=True)
class ScaleWiseSchedule:
    """One spatial scale and one solver interval per student step."""

    scales: tuple[int, ...]
    boundary_indices: tuple[int, ...]
    solver_sigmas: tuple[float, ...]

    def __post_init__(self) -> None:
        scales = tuple(int(value) for value in self.scales)
        boundaries = tuple(int(value) for value in self.boundary_indices)
        sigmas = tuple(float(value) for value in self.solver_sigmas)
        if not scales or any(value <= 0 for value in scales):
            raise ValueError("scales must contain positive latent resolutions")
        if any(left >= right for left, right in zip(scales, scales[1:])):
            raise ValueError("scales must be strictly increasing")
        if len(boundaries) != len(scales) + 1:
            raise ValueError("boundary_indices must contain one more value than scales")
        if boundaries[0] != 0 or any(
            left >= right for left, right in zip(boundaries, boundaries[1:])
        ):
            raise ValueError("boundary_indices must start at zero and increase strictly")
        if len(sigmas) < 2 or boundaries[-1] != len(sigmas) - 1:
            raise ValueError("the final boundary must select the terminal solver sigma")
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in sigmas):
            raise ValueError("solver_sigmas must be finite values in [0,1]")
        if any(left < right for left, right in zip(sigmas, sigmas[1:])):
            raise ValueError("solver_sigmas must be non-increasing")
        if sigmas[0] != 1.0 or sigmas[-1] != 0.0:
            raise ValueError("solver_sigmas must start at one and terminate at zero")
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "boundary_indices", boundaries)
        object.__setattr__(self, "solver_sigmas", sigmas)

    @classmethod
    def released_sd35_four_step(cls) -> ScaleWiseSchedule:
        return cls(
            scales=(64, 80, 96, 128),
            boundary_indices=(0, 7, 14, 18, 28),
            solver_sigmas=flow_match_solver_sigmas(),
        )

    @property
    def num_intervals(self) -> int:
        return len(self.scales)

    def scale(self, interval_index: int) -> int:
        return self.scales[self._interval(interval_index)]

    def previous_scale(self, interval_index: int) -> int:
        index = self._interval(interval_index)
        return self.scales[max(0, index - 1)]

    def start_sigma(self, interval_index: int) -> float:
        index = self._interval(interval_index)
        return self.solver_sigmas[self.boundary_indices[index]]

    def _interval(self, value: int) -> int:
        if isinstance(value, bool):
            raise TypeError("interval_index must be an integer")
        index = int(value)
        if not 0 <= index < self.num_intervals:
            raise ValueError("interval_index is outside the scale-wise schedule")
        return index

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-scale-wise-schedule",
                "scales": list(self.scales),
                "boundary_indices": list(self.boundary_indices),
                "solver_sigmas": list(self.solver_sigmas),
            }
        )


def _positive(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


@dataclass(frozen=True, slots=True)
class ScaleWiseConfig:
    """All behavior-bearing choices in the released SwD objective."""

    schedule: ScaleWiseSchedule
    dmd_enabled: bool = True
    gan_enabled: bool = True
    mmd_enabled: bool = True
    fake_updates_per_iteration: int = 5
    dmd_noise_start_index: int = 12
    dmd_noise_end_index: int = 28
    mmd_noise_start_index: int = 18
    mmd_noise_end_index: int = 28
    teacher_guidance_scale: float = 7.0
    fake_guidance_scale: float = 1.0
    dmd_loss_weight: float = 1.0
    generator_gan_weight: float = 5.0e-3
    critic_gan_weight: float = 1.0e-2
    mmd_loss_weight: float = 1.0
    classifier_blocks: tuple[int, ...] = (11,)
    mmd_blocks: tuple[int, ...] = (11,)
    discriminator_layers: int = 4
    mmd_kernel: str = "linear"
    mmd_rbf_sigma: float = 100.0
    batch_mmd: bool = False
    huber_c: float = 1.0e-3

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, ScaleWiseSchedule):
            raise TypeError("schedule must be ScaleWiseSchedule")
        enabled = tuple(
            bool(value)
            for value in (self.dmd_enabled, self.gan_enabled, self.mmd_enabled)
        )
        if not any(enabled):
            raise ValueError("at least one scale-wise generator objective must be enabled")
        if self.gan_enabled and not self.dmd_enabled:
            raise ValueError("released scale-wise GAN training requires DMD fake updates")
        if isinstance(self.fake_updates_per_iteration, bool):
            raise TypeError("fake_updates_per_iteration must be an integer")
        fake_updates = int(self.fake_updates_per_iteration)
        if fake_updates < 0 or (self.dmd_enabled and fake_updates == 0):
            raise ValueError("DMD requires a positive fake update count")
        if not self.dmd_enabled and fake_updates != 0:
            raise ValueError("fake updates must be zero when DMD is disabled")
        maximum_index = len(self.schedule.solver_sigmas) - 1

        def index_range(start: object, end: object, *, name: str) -> tuple[int, int]:
            if isinstance(start, bool) or isinstance(end, bool):
                raise TypeError(f"{name} indices must be integers")
            resolved_start, resolved_end = int(start), int(end)
            if not 0 <= resolved_start < resolved_end <= maximum_index:
                raise ValueError(
                    f"{name} indices must satisfy 0 <= start < end <= {maximum_index}"
                )
            return resolved_start, resolved_end

        dmd_start, dmd_end = index_range(
            self.dmd_noise_start_index,
            self.dmd_noise_end_index,
            name="DMD noise",
        )
        mmd_start, mmd_end = index_range(
            self.mmd_noise_start_index,
            self.mmd_noise_end_index,
            name="MMD noise",
        )
        classifier_blocks = tuple(int(value) for value in self.classifier_blocks)
        mmd_blocks = tuple(int(value) for value in self.mmd_blocks)
        if any(value < 0 for value in (*classifier_blocks, *mmd_blocks)):
            raise ValueError("feature block indices must be non-negative")
        if self.gan_enabled and not classifier_blocks:
            raise ValueError("GAN training requires classifier_blocks")
        if self.mmd_enabled and not mmd_blocks:
            raise ValueError("MMD training requires mmd_blocks")
        if isinstance(self.discriminator_layers, bool) or int(self.discriminator_layers) <= 0:
            raise ValueError("discriminator_layers must be positive")
        kernel = str(self.mmd_kernel).strip().lower()
        if kernel not in {"linear", "rbf"}:
            raise ValueError("mmd_kernel must be 'linear' or 'rbf'")
        teacher_guidance = float(self.teacher_guidance_scale)
        fake_guidance = float(self.fake_guidance_scale)
        if any(
            not isfinite(value) or value < 0.0
            for value in (teacher_guidance, fake_guidance)
        ):
            raise ValueError("guidance scales must be finite and non-negative")
        weights = {
            "dmd_loss_weight": _positive(
                self.dmd_loss_weight,
                field_name="dmd_loss_weight",
            ),
            "generator_gan_weight": _positive(
                self.generator_gan_weight,
                field_name="generator_gan_weight",
            ),
            "critic_gan_weight": _positive(
                self.critic_gan_weight,
                field_name="critic_gan_weight",
            ),
            "mmd_loss_weight": _positive(
                self.mmd_loss_weight,
                field_name="mmd_loss_weight",
            ),
            "mmd_rbf_sigma": _positive(
                self.mmd_rbf_sigma,
                field_name="mmd_rbf_sigma",
            ),
        }
        huber_c = float(self.huber_c)
        if not isfinite(huber_c) or huber_c < 0.0:
            raise ValueError("huber_c must be finite and non-negative")
        object.__setattr__(self, "dmd_enabled", enabled[0])
        object.__setattr__(self, "gan_enabled", enabled[1])
        object.__setattr__(self, "mmd_enabled", enabled[2])
        object.__setattr__(self, "fake_updates_per_iteration", fake_updates)
        object.__setattr__(self, "dmd_noise_start_index", dmd_start)
        object.__setattr__(self, "dmd_noise_end_index", dmd_end)
        object.__setattr__(self, "mmd_noise_start_index", mmd_start)
        object.__setattr__(self, "mmd_noise_end_index", mmd_end)
        object.__setattr__(self, "teacher_guidance_scale", teacher_guidance)
        object.__setattr__(self, "fake_guidance_scale", fake_guidance)
        object.__setattr__(self, "classifier_blocks", classifier_blocks)
        object.__setattr__(self, "mmd_blocks", mmd_blocks)
        object.__setattr__(self, "discriminator_layers", int(self.discriminator_layers))
        object.__setattr__(self, "mmd_kernel", kernel)
        object.__setattr__(self, "batch_mmd", bool(self.batch_mmd))
        object.__setattr__(self, "huber_c", huber_c)
        for name, value in weights.items():
            object.__setattr__(self, name, value)

    @classmethod
    def released_sd35_medium(cls) -> ScaleWiseConfig:
        return cls(schedule=ScaleWiseSchedule.released_sd35_four_step())

    @classmethod
    def released_sd35_large(cls) -> ScaleWiseConfig:
        return cls(
            schedule=ScaleWiseSchedule.released_sd35_four_step(),
            teacher_guidance_scale=4.5,
            classifier_blocks=(20,),
            mmd_blocks=(20,),
        )

    @classmethod
    def from_recipe(cls, spec: ScaleWiseAlgorithmSpec) -> ScaleWiseConfig:
        if not isinstance(spec, ScaleWiseAlgorithmSpec):
            raise TypeError("spec must be ScaleWiseAlgorithmSpec")
        return cls(
            schedule=ScaleWiseSchedule(
                scales=spec.scales,
                boundary_indices=spec.boundary_indices,
                solver_sigmas=spec.solver_sigmas,
            ),
            dmd_enabled=spec.dmd_enabled,
            gan_enabled=spec.gan_enabled,
            mmd_enabled=spec.mmd_enabled,
            fake_updates_per_iteration=spec.fake_updates_per_iteration,
            dmd_noise_start_index=spec.dmd_noise_start_index,
            dmd_noise_end_index=spec.dmd_noise_end_index,
            mmd_noise_start_index=spec.mmd_noise_start_index,
            mmd_noise_end_index=spec.mmd_noise_end_index,
            teacher_guidance_scale=spec.teacher_guidance_scale,
            fake_guidance_scale=spec.fake_guidance_scale,
            dmd_loss_weight=spec.dmd_loss_weight,
            generator_gan_weight=spec.generator_gan_weight,
            critic_gan_weight=spec.critic_gan_weight,
            mmd_loss_weight=spec.mmd_loss_weight,
            classifier_blocks=spec.classifier_blocks,
            mmd_blocks=spec.mmd_blocks,
            discriminator_layers=spec.discriminator_layers,
            mmd_kernel=spec.mmd_kernel,
            mmd_rbf_sigma=spec.mmd_rbf_sigma,
            batch_mmd=spec.batch_mmd,
            huber_c=spec.huber_c,
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-scale-wise-config",
                "schedule_digest": self.schedule.digest,
                "dmd_enabled": self.dmd_enabled,
                "gan_enabled": self.gan_enabled,
                "mmd_enabled": self.mmd_enabled,
                "fake_updates_per_iteration": self.fake_updates_per_iteration,
                "dmd_noise_range": [
                    self.dmd_noise_start_index,
                    self.dmd_noise_end_index,
                ],
                "mmd_noise_range": [
                    self.mmd_noise_start_index,
                    self.mmd_noise_end_index,
                ],
                "teacher_guidance_scale": self.teacher_guidance_scale,
                "fake_guidance_scale": self.fake_guidance_scale,
                "dmd_loss_weight": self.dmd_loss_weight,
                "generator_gan_weight": self.generator_gan_weight,
                "critic_gan_weight": self.critic_gan_weight,
                "mmd_loss_weight": self.mmd_loss_weight,
                "classifier_blocks": list(self.classifier_blocks),
                "mmd_blocks": list(self.mmd_blocks),
                "discriminator_layers": self.discriminator_layers,
                "mmd_kernel": self.mmd_kernel,
                "mmd_rbf_sigma": self.mmd_rbf_sigma,
                "batch_mmd": self.batch_mmd,
                "huber_c": self.huber_c,
            }
        )


__all__ = [
    "ScaleWiseConfig",
    "ScaleWiseSchedule",
    "flow_match_solver_sigmas",
]
