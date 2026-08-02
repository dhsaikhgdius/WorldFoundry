"""Strict recipe contract for native Score Identity Distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log

from ..common import positive_int, strict_mapping
from .dmd import _normalize_few_step_schedule

SID_ALGORITHM_FIELDS = {
    "type",
    "student_timesteps",
    "student_sigmas",
    "teacher_checkpoint",
    "fake_score_checkpoint",
    "alpha",
    "noise_policy",
    "score_weighting",
    "num_train_timesteps",
    "score_logit_mean",
    "score_logit_std",
    "weighting_epsilon",
    "teacher_guidance_scale",
    "fake_score_guidance_scale",
    "score_identity_weight",
    "fake_score_flow_weight",
    "generator_adversarial_weight",
    "fake_score_adversarial_weight",
}

SID_WEIGHTING_SCHEMES = frozenset(
    {
        "sid-legacy",
        "snr-sqrt",
        "snr",
        "1-over-sigma2",
        "1-over-sigma",
        "1-minus-sigma-squared",
        "1-minus-sigma",
    }
)


def _finite(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class SIDAlgorithmSpec:
    """All mathematical switches consumed by the native SiD-DiT runtime."""

    student_timesteps: tuple[float, ...]
    student_sigmas: tuple[float, ...]
    teacher_checkpoint: str
    fake_score_checkpoint: str
    alpha: float
    noise_policy: str = "fresh"
    score_weighting: str = "1-minus-sigma"
    num_train_timesteps: int = 1000
    score_logit_mean: float = log(2.0)
    score_logit_std: float = 1.6
    weighting_epsilon: float = 1.0e-5
    teacher_guidance_scale: float = 4.5
    fake_score_guidance_scale: float = 4.5
    score_identity_weight: float = 100.0
    fake_score_flow_weight: float = 1.0
    generator_adversarial_weight: float = 0.0
    fake_score_adversarial_weight: float = 0.0
    type: str = "sid"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "sid":
            raise ValueError("SiD algorithm type must be 'sid'")
        timesteps, sigmas = _normalize_few_step_schedule(
            self.student_timesteps,
            self.student_sigmas,
        )
        for name in ("teacher_checkpoint", "fake_score_checkpoint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty checkpoint identity")
            object.__setattr__(self, name, value.strip())
        noise_policy = str(self.noise_policy).strip().lower().replace("_", "-")
        if noise_policy not in {"fresh", "fixed", "ddim"}:
            raise ValueError("noise_policy must be fresh, fixed, or ddim")
        weighting = str(self.score_weighting).strip().lower().replace("_", "-")
        if weighting not in SID_WEIGHTING_SCHEMES:
            raise ValueError(f"score_weighting must be one of {sorted(SID_WEIGHTING_SCHEMES)}")
        num_train_timesteps = positive_int(
            self.num_train_timesteps,
            field_name="algorithm.num_train_timesteps",
        )
        if num_train_timesteps < 2:
            raise ValueError("algorithm.num_train_timesteps must be at least two")
        values = {
            name: _finite(getattr(self, name), field_name=name)
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
        generator_gan = values["generator_adversarial_weight"]
        fake_score_gan = values["fake_score_adversarial_weight"]
        if (generator_gan == 0) != (fake_score_gan == 0):
            raise ValueError("SiD DiffusionGAN generator and fake-score weights must be enabled together")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "student_timesteps", timesteps)
        object.__setattr__(self, "student_sigmas", sigmas)
        object.__setattr__(self, "noise_policy", noise_policy)
        object.__setattr__(self, "score_weighting", weighting)
        object.__setattr__(self, "num_train_timesteps", num_train_timesteps)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def diffusion_gan_enabled(self) -> bool:
        return self.generator_adversarial_weight > 0


def parse_sid_algorithm(value: object) -> SIDAlgorithmSpec:
    return SIDAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=SID_ALGORITHM_FIELDS,
        )
    )


__all__ = ["SIDAlgorithmSpec", "SID_WEIGHTING_SCHEMES"]
