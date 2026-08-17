"""Strict recipe contract for teacher-anchored on-policy diffusion distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import mapping, strict_mapping

DIFFUSION_OPD_TEACHER_FIELDS = {"name", "checkpoint", "guidance_scale"}
DIFFUSION_OPD_ALGORITHM_FIELDS = {
    "type",
    "teachers",
    "sigmas",
    "sde_step_indices",
    "eta",
    "guidance_scale",
    "add_kl_coefficient",
    "trajectory_dtype",
}


@dataclass(frozen=True, slots=True)
class DiffusionOPDTeacherSpec:
    """One domain teacher and the guidance used for its frozen replay."""

    name: str
    checkpoint: str
    guidance_scale: float | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        checkpoint = str(self.checkpoint).strip()
        if not name or not checkpoint:
            raise ValueError("DiffusionOPD teacher name/checkpoint must be non-empty")
        guidance = None if self.guidance_scale is None else float(self.guidance_scale)
        if guidance is not None and (not isfinite(guidance) or guidance < 0):
            raise ValueError("DiffusionOPD teacher guidance_scale must be finite and non-negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(self, "guidance_scale", guidance)


@dataclass(frozen=True, slots=True)
class DiffusionOPDAlgorithmSpec:
    """Every behavior-bearing field in native DiffusionOPD execution."""

    teachers: tuple[DiffusionOPDTeacherSpec, ...]
    sigmas: tuple[float, ...]
    sde_step_indices: tuple[int, ...]
    eta: float = 1.0e-6
    guidance_scale: float = 4.5
    add_kl_coefficient: bool = False
    trajectory_dtype: str = "float16"
    type: str = "diffusion-opd"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "diffusion-opd":
            raise ValueError("DiffusionOPD algorithm type must be 'diffusion-opd'")
        teachers = tuple(self.teachers)
        if not teachers or not all(isinstance(item, DiffusionOPDTeacherSpec) for item in teachers):
            raise ValueError("DiffusionOPD requires typed teacher specs")
        names = tuple(item.name for item in teachers)
        if len(set(names)) != len(names):
            raise ValueError("DiffusionOPD teacher names must be unique")
        sigmas = tuple(float(value) for value in self.sigmas)
        if (
            len(sigmas) < 2
            or any(not isfinite(value) or not 0 <= value <= 1 for value in sigmas)
            or any(left <= right for left, right in zip(sigmas, sigmas[1:]))
        ):
            raise ValueError("DiffusionOPD sigmas must be finite and strictly descending in [0,1]")
        if sigmas[0] != 1.0 or sigmas[-1] != 0.0:
            raise ValueError("DiffusionOPD sigmas must start at one and end at zero")
        indices = tuple(int(index) for index in self.sde_step_indices)
        if not indices or indices != tuple(sorted(set(indices))) or indices[0] < 0 or indices[-1] >= len(sigmas) - 1:
            raise ValueError("DiffusionOPD sde_step_indices must be non-empty, sorted, unique, and in range")
        eta = float(self.eta)
        guidance = float(self.guidance_scale)
        if not isfinite(eta) or eta <= 0:
            raise ValueError("DiffusionOPD eta must be finite and positive")
        if not isfinite(guidance) or guidance < 0:
            raise ValueError("DiffusionOPD guidance_scale must be finite and non-negative")
        if not isinstance(self.add_kl_coefficient, bool):
            raise TypeError("DiffusionOPD add_kl_coefficient must be a bool")
        dtype = str(self.trajectory_dtype).strip().lower().removeprefix("torch.")
        dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(dtype, dtype)
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("DiffusionOPD trajectory_dtype must be bfloat16, float16, or float32")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "teachers", teachers)
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(self, "sde_step_indices", indices)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "guidance_scale", guidance)
        object.__setattr__(self, "trajectory_dtype", dtype)


def parse_diffusion_opd_algorithm(value: object) -> DiffusionOPDAlgorithmSpec:
    """Parse the algorithm and its domain-teacher list without runtime imports."""

    payload = mapping(value, field_name="algorithm")
    missing = sorted({"teachers", "sigmas", "sde_step_indices"} - set(payload))
    if missing:
        raise ValueError(f"DiffusionOPD algorithm is missing required fields: {missing}")
    algorithm_payload = strict_mapping(
        payload,
        field_name="algorithm",
        allowed=DIFFUSION_OPD_ALGORITHM_FIELDS,
    )
    raw_teachers = algorithm_payload.pop("teachers")
    if isinstance(raw_teachers, (str, bytes)) or not isinstance(raw_teachers, (list, tuple)):
        raise TypeError("algorithm.teachers must be a sequence")
    teachers = tuple(
        DiffusionOPDTeacherSpec(
            **strict_mapping(
                item,
                field_name=f"algorithm.teachers[{index}]",
                allowed=DIFFUSION_OPD_TEACHER_FIELDS,
            )
        )
        for index, item in enumerate(raw_teachers)
    )
    return DiffusionOPDAlgorithmSpec(**algorithm_payload, teachers=teachers)


__all__ = [
    "DiffusionOPDAlgorithmSpec",
    "DiffusionOPDTeacherSpec",
    "parse_diffusion_opd_algorithm",
]
