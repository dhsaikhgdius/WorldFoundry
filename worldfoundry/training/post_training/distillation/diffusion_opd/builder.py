"""Construction of a complete WorldFoundry-native DiffusionOPD stack."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.recipes.post_training.algorithms.diffusion_opd import (
    DiffusionOPDAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ...rl.rollout_strategies.transition import VariancePreservingFlowTransition
from ...shared.building import (
    build_post_training_optimizer,
    prediction_module,
    require_checkpoint_identity,
    require_disjoint_trainable_parameters,
    resolve_tensor_dtype,
    validate_post_training_recipe,
)
from ...shared.contracts import FlowPredictionAdapter
from ...shared.distributed import PostTrainingParallelContext
from .adapters import BranchClassifierFreeGuidance
from .engine import NativeDiffusionOPDEngine
from .trajectory import (
    DiffusionOPDTrajectorySampler,
    NativeDiffusionOPDTrajectoryReplay,
)


@dataclass(frozen=True, slots=True)
class NativeDiffusionOPDTrainingStack:
    """Student sampler, domain teachers, objective, optimizer, and engine."""

    recipe: PostTrainingRecipe
    sampler: DiffusionOPDTrajectorySampler
    student_replay: NativeDiffusionOPDTrajectoryReplay
    teacher_replays: Mapping[str, NativeDiffusionOPDTrajectoryReplay]
    optimizer: torch.optim.Optimizer
    engine: NativeDiffusionOPDEngine

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {"lr_scheduler": None, "ema": None, "algorithm_state": None}


def build_native_diffusion_opd_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: FlowPredictionAdapter,
    teachers: Mapping[str, FlowPredictionAdapter],
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeDiffusionOPDTrainingStack:
    """Bind current-student rollout and frozen domain teachers to one optimizer."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, DiffusionOPDAlgorithmSpec):
        raise TypeError("DiffusionOPD stack requires DiffusionOPDAlgorithmSpec")
    spec = recipe.algorithm
    student_module = prediction_module(student, role="DiffusionOPD student")
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="DiffusionOPD student",
    )
    teacher_registry = dict(teachers)
    expected_names = {teacher.name for teacher in spec.teachers}
    if set(teacher_registry) != expected_names:
        raise ValueError("DiffusionOPD teacher registry must exactly match recipe teacher names")
    if recipe.optimizer.gradient_accumulation_steps % len(spec.teachers):
        raise ValueError("DiffusionOPD gradient accumulation must contain complete teacher-domain cycles")

    role_modules: dict[str, torch.nn.Module] = {"student": student_module}
    for teacher_spec in spec.teachers:
        teacher = teacher_registry[teacher_spec.name]
        module = prediction_module(
            teacher,
            role=f"DiffusionOPD teacher {teacher_spec.name}",
        )
        require_checkpoint_identity(
            teacher,
            teacher_spec.checkpoint,
            role=f"DiffusionOPD teacher {teacher_spec.name}",
        )
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError("DiffusionOPD teacher adapter parameters must be frozen")
        module.eval()
        role_modules[f"teacher:{teacher_spec.name}"] = module
    require_disjoint_trainable_parameters(role_modules)

    transition_strategy = VariancePreservingFlowTransition(
        eta=spec.eta,
        sigma_max=spec.sigmas[1],
    )
    active_student = BranchClassifierFreeGuidance(
        student,
        guidance_scale=spec.guidance_scale,
    )
    student_replay = NativeDiffusionOPDTrajectoryReplay(
        active_student,
        transition_strategy=transition_strategy,
    )
    teacher_replays = {
        teacher_spec.name: NativeDiffusionOPDTrajectoryReplay(
            BranchClassifierFreeGuidance(
                teacher_registry[teacher_spec.name],
                guidance_scale=(
                    spec.guidance_scale if teacher_spec.guidance_scale is None else teacher_spec.guidance_scale
                ),
            ),
            transition_strategy=transition_strategy,
        )
        for teacher_spec in spec.teachers
    }
    sampler = DiffusionOPDTrajectorySampler(
        active_student,
        transition_strategy=transition_strategy,
        sigmas=spec.sigmas,
        step_indices=spec.sde_step_indices,
        trajectory_dtype=resolve_tensor_dtype(spec.trajectory_dtype),
    )
    optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="DiffusionOPD student",
    )
    engine = NativeDiffusionOPDEngine(
        student_module=student_module,
        student_replay=student_replay,
        teacher_replays=teacher_replays,
        optimizer=optimizer,
        add_kl_coefficient=spec.add_kl_coefficient,
        gradient_accumulation_steps=recipe.optimizer.gradient_accumulation_steps,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        parallel_context=parallel_context,
    )
    return NativeDiffusionOPDTrainingStack(
        recipe=recipe,
        sampler=sampler,
        student_replay=student_replay,
        teacher_replays=teacher_replays,
        optimizer=optimizer,
        engine=engine,
    )


__all__ = [
    "NativeDiffusionOPDTrainingStack",
    "build_native_diffusion_opd_training_stack",
]
