"""Strict recipe contract for scale-wise flow distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from worldfoundry.training.objectives.flow_matching import flow_match_solver_sigmas

from ..common import positive_int, strict_mapping


def _finite(value: object, *, field_name: str, minimum: float = 0.0) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < minimum:
        raise ValueError(f"{field_name} must be finite and at least {minimum}")
    return resolved


@dataclass(frozen=True, slots=True)
class ScaleWiseAlgorithmSpec:
    """Progressive scale schedule and every released SwD loss choice."""

    teacher_checkpoint: str
    fake_score_checkpoint: str
    scales: tuple[int, ...] = (64, 80, 96, 128)
    boundary_indices: tuple[int, ...] = (0, 7, 14, 18, 28)
    solver_sigmas: tuple[float, ...] = flow_match_solver_sigmas()
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
    type: str = "scale-wise-distillation"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "scale-wise-distillation":
            raise ValueError(
                "scale-wise algorithm type must be 'scale-wise-distillation'"
            )
        for name in ("teacher_checkpoint", "fake_score_checkpoint"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be a non-empty checkpoint reference")
            object.__setattr__(self, name, value)
        scales = tuple(positive_int(value, field_name="algorithm.scales") for value in self.scales)
        if not scales or any(left >= right for left, right in zip(scales, scales[1:])):
            raise ValueError("algorithm.scales must be strictly increasing")
        boundaries = tuple(int(value) for value in self.boundary_indices)
        sigmas = tuple(float(value) for value in self.solver_sigmas)
        if (
            len(boundaries) != len(scales) + 1
            or boundaries[0] != 0
            or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError(
                "boundary_indices must start at zero and contain one increasing interval per scale"
            )
        if len(sigmas) < 2 or boundaries[-1] != len(sigmas) - 1:
            raise ValueError("final boundary must select the terminal solver sigma")
        if (
            sigmas[0] != 1.0
            or sigmas[-1] != 0.0
            or any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in sigmas)
            or any(left < right for left, right in zip(sigmas, sigmas[1:]))
        ):
            raise ValueError("solver_sigmas must decrease from one to zero")
        for name in ("dmd_enabled", "gan_enabled", "mmd_enabled", "batch_mmd"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if not any((self.dmd_enabled, self.gan_enabled, self.mmd_enabled)):
            raise ValueError("at least one scale-wise generator objective must be enabled")
        if self.gan_enabled and not self.dmd_enabled:
            raise ValueError("released scale-wise GAN training requires DMD")
        if isinstance(self.fake_updates_per_iteration, bool):
            raise TypeError("fake_updates_per_iteration must be an integer")
        fake_updates = int(self.fake_updates_per_iteration)
        if self.dmd_enabled:
            if fake_updates <= 0:
                raise ValueError("DMD requires positive fake_updates_per_iteration")
        elif fake_updates != 0:
            raise ValueError("fake updates are unused when DMD is disabled")
        maximum_index = len(sigmas) - 1

        def index_range(start: object, end: object, *, name: str) -> tuple[int, int]:
            if isinstance(start, bool) or isinstance(end, bool):
                raise TypeError(f"{name} indices must be integers")
            resolved = (int(start), int(end))
            if not 0 <= resolved[0] < resolved[1] <= maximum_index:
                raise ValueError(
                    f"{name} indices must satisfy 0 <= start < end <= {maximum_index}"
                )
            return resolved

        dmd_range = index_range(
            self.dmd_noise_start_index,
            self.dmd_noise_end_index,
            name="DMD noise",
        )
        mmd_range = index_range(
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
        guidance = tuple(
            _finite(getattr(self, name), field_name=name)
            for name in ("teacher_guidance_scale", "fake_guidance_scale")
        )
        positive_names = (
            "dmd_loss_weight",
            "generator_gan_weight",
            "critic_gan_weight",
            "mmd_loss_weight",
            "mmd_rbf_sigma",
        )
        positives = {
            name: _finite(getattr(self, name), field_name=name, minimum=0.0)
            for name in positive_names
        }
        if any(value <= 0.0 for value in positives.values()):
            raise ValueError("scale-wise loss weights and RBF sigma must be positive")
        kernel = str(self.mmd_kernel).strip().lower()
        if kernel not in {"linear", "rbf"}:
            raise ValueError("mmd_kernel must be 'linear' or 'rbf'")
        huber_c = _finite(self.huber_c, field_name="huber_c")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "boundary_indices", boundaries)
        object.__setattr__(self, "solver_sigmas", sigmas)
        object.__setattr__(self, "fake_updates_per_iteration", fake_updates)
        object.__setattr__(self, "dmd_noise_start_index", dmd_range[0])
        object.__setattr__(self, "dmd_noise_end_index", dmd_range[1])
        object.__setattr__(self, "mmd_noise_start_index", mmd_range[0])
        object.__setattr__(self, "mmd_noise_end_index", mmd_range[1])
        object.__setattr__(self, "teacher_guidance_scale", guidance[0])
        object.__setattr__(self, "fake_guidance_scale", guidance[1])
        object.__setattr__(self, "classifier_blocks", classifier_blocks)
        object.__setattr__(self, "mmd_blocks", mmd_blocks)
        object.__setattr__(
            self,
            "discriminator_layers",
            positive_int(
                self.discriminator_layers,
                field_name="algorithm.discriminator_layers",
            ),
        )
        object.__setattr__(self, "mmd_kernel", kernel)
        object.__setattr__(self, "huber_c", huber_c)
        for name, value in positives.items():
            object.__setattr__(self, name, value)


SCALE_WISE_ALGORITHM_FIELDS = frozenset(ScaleWiseAlgorithmSpec.__dataclass_fields__)


def parse_scale_wise_algorithm(value: object) -> ScaleWiseAlgorithmSpec:
    return ScaleWiseAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=set(SCALE_WISE_ALGORITHM_FIELDS),
        )
    )


__all__ = ["ScaleWiseAlgorithmSpec", "parse_scale_wise_algorithm"]
