"""SANA-Sprint sCM-LADD recipe contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import strict_mapping
from .auxiliary_optimizers import (
    AuxiliaryOptimizerRule,
    forbids_auxiliary,
    requires_auxiliary,
)

SCM_LADD_ALGORITHM_FIELDS = {
    "type",
    "teacher_checkpoint",
    "sigma_data",
    "generator_logit_mean",
    "generator_logit_std",
    "discriminator_logit_mean",
    "discriminator_logit_std",
    "teacher_guidance_scales",
    "guidance_embedding_scale",
    "discriminator_head_block_ids",
    "lr_scheduler",
    "lr_warmup_steps",
    "student_fp32_attention",
    "teacher_fp32_attention",
    "tangent_warmup_steps",
    "tangent_normalization_constant",
    "consistency_weight",
    "adversarial_weight",
    "max_time_probability",
    "largest_time_enabled",
    "largest_time",
    "misaligned_pairs",
    "independent_real_fake_discriminator_times",
    "adversarial_loss",
}


def _finite_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class SCMLADDAlgorithmSpec:
    """Every mathematical choice executed by the native sCM-LADD objective."""

    teacher_checkpoint: str = "default"
    sigma_data: float = 0.5
    generator_logit_mean: float = 0.2
    generator_logit_std: float = 1.6
    discriminator_logit_mean: float = -0.6
    discriminator_logit_std: float = 1.0
    teacher_guidance_scales: tuple[float, ...] = (4.0, 4.5, 5.0)
    guidance_embedding_scale: float = 0.1
    discriminator_head_block_ids: tuple[int, ...] = (2, 8, 14, 19)
    lr_scheduler: str = "constant-with-warmup"
    lr_warmup_steps: int = 5000
    student_fp32_attention: bool = True
    teacher_fp32_attention: bool = False
    tangent_warmup_steps: int = 4000
    tangent_normalization_constant: float = 0.1
    consistency_weight: float = 1.0
    adversarial_weight: float = 0.5
    max_time_probability: float = 0.5
    largest_time_enabled: bool = True
    largest_time: float = 1.57080
    misaligned_pairs: bool = True
    independent_real_fake_discriminator_times: bool = True
    adversarial_loss: str = "hinge"
    type: str = "scm-ladd"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "scm-ladd":
            raise ValueError("SCM-LADD algorithm type must be 'scm-ladd'")
        object.__setattr__(self, "type", algorithm_type)
        teacher_checkpoint = str(self.teacher_checkpoint).strip()
        if not teacher_checkpoint:
            raise ValueError("teacher_checkpoint cannot be empty")
        object.__setattr__(self, "teacher_checkpoint", teacher_checkpoint)
        for name in (
            "sigma_data",
            "generator_logit_std",
            "discriminator_logit_std",
            "tangent_normalization_constant",
            "consistency_weight",
            "adversarial_weight",
            "guidance_embedding_scale",
        ):
            value = _finite_float(getattr(self, name), field_name=name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("generator_logit_mean", "discriminator_logit_mean"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), field_name=name))
        scheduler = str(self.lr_scheduler).strip().lower().replace("_", "-")
        if scheduler != "constant-with-warmup":
            raise ValueError("SCM-LADD lr_scheduler must be 'constant-with-warmup'")
        object.__setattr__(self, "lr_scheduler", scheduler)
        if isinstance(self.lr_warmup_steps, bool) or int(self.lr_warmup_steps) < 0:
            raise ValueError("lr_warmup_steps must be a non-negative integer")
        object.__setattr__(self, "lr_warmup_steps", int(self.lr_warmup_steps))
        for name in ("student_fp32_attention", "teacher_fp32_attention"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if isinstance(self.tangent_warmup_steps, bool) or int(self.tangent_warmup_steps) <= 0:
            raise ValueError("tangent_warmup_steps must be a positive integer")
        object.__setattr__(self, "tangent_warmup_steps", int(self.tangent_warmup_steps))
        probability = _finite_float(self.max_time_probability, field_name="max_time_probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("max_time_probability must be in [0,1]")
        object.__setattr__(self, "max_time_probability", probability)
        if not isinstance(self.largest_time_enabled, bool):
            raise TypeError("largest_time_enabled must be bool")
        largest_time = _finite_float(self.largest_time, field_name="largest_time")
        if not 0.0 < largest_time <= 1.57080:
            raise ValueError("largest_time must be in (0,1.57080]")
        object.__setattr__(self, "largest_time", largest_time)
        for name in ("misaligned_pairs", "independent_real_fake_discriminator_times"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        adversarial_loss = str(self.adversarial_loss).strip().lower().replace("_", "-")
        if adversarial_loss != "hinge":
            raise ValueError("SCM-LADD adversarial_loss must be 'hinge'")
        object.__setattr__(self, "adversarial_loss", adversarial_loss)
        guidance = tuple(
            _finite_float(value, field_name="teacher_guidance_scales")
            for value in self.teacher_guidance_scales
        )
        if not guidance or any(value <= 0 for value in guidance):
            raise ValueError("teacher_guidance_scales must contain positive values")
        object.__setattr__(self, "teacher_guidance_scales", guidance)
        head_ids = tuple(int(value) for value in self.discriminator_head_block_ids)
        if not head_ids or any(value < 0 for value in head_ids) or len(set(head_ids)) != len(head_ids):
            raise ValueError("discriminator_head_block_ids must be unique non-negative integers")
        if any(left >= right for left, right in zip(head_ids, head_ids[1:])):
            raise ValueError("discriminator_head_block_ids must be strictly increasing")
        object.__setattr__(self, "discriminator_head_block_ids", head_ids)

    def auxiliary_optimizer_rules(self) -> tuple[AuxiliaryOptimizerRule, ...]:
        return (
            requires_auxiliary("discriminator_optimizer", "SCM-LADD requires discriminator_optimizer"),
            forbids_auxiliary(
                "fake_score_optimizer",
                "guidance_optimizer",
                message="SCM-LADD only accepts discriminator_optimizer",
            ),
        )


def parse_scm_ladd_algorithm(value: object) -> SCMLADDAlgorithmSpec:
    """Parse a strict sCM-LADD algorithm section."""

    return SCMLADDAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=SCM_LADD_ALGORITHM_FIELDS,
        )
    )


__all__ = ["SCMLADDAlgorithmSpec"]
