"""Construction of the native two-stage diagonal distillation stack."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import nn

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.diagonal import (
    DiagonalAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ...shared.building import (
    build_post_training_optimizer,
    named_stateful_collection,
    prediction_module,
    require_checkpoint_identity,
    require_independent_modules,
    validate_post_training_recipe,
)
from ...shared.contracts import FlowPredictionAdapter
from ...shared.distributed import PostTrainingParallelContext
from ...shared.ema import DelayedModuleEMA
from ..dmd.engine import NativeDMDTrainEngine
from .config import DiagonalObjectiveConfig, DiagonalScheduleConfig
from .contracts import DiagonalCausalAdapter
from .engine import NativeDiagonalTrainEngine
from .motion import SpatialMotionHead, register_motion_head
from .objective import DiagonalDMDLossAdapter
from .rollout import DiagonalFixedTeacherSampler, DiagonalRolloutSampler


@dataclass(frozen=True, slots=True)
class NativeDiagonalTrainingStack:
    """All model roles, stateful math, optimizers, and execution cadence."""

    recipe: PostTrainingRecipe
    schedule: DiagonalScheduleConfig
    fixed_teacher_schedule: DiagonalScheduleConfig
    objective_config: DiagonalObjectiveConfig
    sampler: DiagonalRolloutSampler
    fixed_teacher_sampler: DiagonalFixedTeacherSampler
    motion_head_student: SpatialMotionHead
    motion_head_teacher: SpatialMotionHead
    loss_adapter: DiagonalDMDLossAdapter
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer
    engine: NativeDiagonalTrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }


def _causal_module(adapter: object, *, role: str) -> nn.Module:
    if not isinstance(adapter, DiagonalCausalAdapter):
        raise TypeError(f"{role} must implement DiagonalCausalAdapter")
    module = adapter.module
    if not isinstance(module, nn.Module):
        raise TypeError(f"{role}.module must be torch.nn.Module")
    return module


def _student_motion_head(
    student_module: nn.Module,
    *,
    channels: int,
) -> SpatialMotionHead:
    existing = getattr(student_module, "diagonal_motion_head", None)
    if existing is not None:
        if not isinstance(existing, SpatialMotionHead):
            raise TypeError("student.diagonal_motion_head has an incompatible type")
        if existing.channels != channels:
            raise ValueError(
                "student diagonal motion-head channels differ from the recipe"
            )
        return existing
    try:
        parameter = next(student_module.parameters())
    except StopIteration as error:
        raise ValueError("diagonal student has no parameters") from error
    head = SpatialMotionHead(channels).to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    registered = register_motion_head(student_module, head)
    if not isinstance(registered, SpatialMotionHead):
        raise RuntimeError("diagonal motion-head registration changed its type")
    return registered


def build_native_diagonal_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: DiagonalCausalAdapter,
    real_score: FlowPredictionAdapter,
    fake_score: FlowPredictionAdapter,
    fixed_teacher: DiagonalCausalAdapter,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeDiagonalTrainingStack:
    """Build the released diagonal objective over native model adapters."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, DiagonalAlgorithmSpec):
        raise TypeError(
            "diagonal stack requires a DiagonalAlgorithmSpec recipe"
        )
    if recipe.fake_score_optimizer is None:
        raise ValueError("diagonal distillation requires fake_score_optimizer")
    algorithm = recipe.algorithm
    accumulation_steps = recipe.optimizer.gradient_accumulation_steps
    if recipe.fake_score_optimizer.gradient_accumulation_steps != accumulation_steps:
        raise ValueError(
            "diagonal student and fake-score accumulation steps must match"
        )

    student_module = _causal_module(student, role="diagonal student")
    fixed_teacher_module = _causal_module(
        fixed_teacher,
        role="diagonal fixed teacher",
    )
    real_score_module = prediction_module(
        real_score,
        role="diagonal real-score teacher",
    )
    fake_score_module = prediction_module(
        fake_score,
        role="diagonal fake-score critic",
    )
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="diagonal student",
    )
    require_checkpoint_identity(
        real_score,
        algorithm.real_score_checkpoint,
        role="diagonal real-score teacher",
    )
    require_checkpoint_identity(
        fake_score,
        algorithm.fake_score_checkpoint,
        role="diagonal fake-score critic",
    )
    require_checkpoint_identity(
        fixed_teacher,
        algorithm.fixed_teacher_checkpoint,
        role="diagonal fixed teacher",
    )
    require_independent_modules(
        {
            "student": student_module,
            "real-score": real_score_module,
            "fake-score": fake_score_module,
            "fixed-teacher": fixed_teacher_module,
        }
    )
    for name, module in (
        ("real-score teacher", real_score_module),
        ("fixed teacher", fixed_teacher_module),
    ):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError(f"diagonal {name} must be frozen")
        module.eval()

    schedule_factory = (
        DiagonalScheduleConfig.stage_one
        if algorithm.stage == "stage-one"
        else DiagonalScheduleConfig.stage_two
    )
    schedule = schedule_factory(
        frames_per_block=algorithm.frames_per_block,
        frame_dim=algorithm.frame_dim,
    )
    fixed_teacher_schedule = DiagonalScheduleConfig.fixed_teacher(
        frames_per_block=algorithm.frames_per_block,
        frame_dim=algorithm.frame_dim,
    )
    objective_config = DiagonalObjectiveConfig.released(schedule)
    context = parallel_context or PostTrainingParallelContext.current()
    sampler = DiagonalRolloutSampler(
        student,
        schedule,
        parallel_context=context,
    )
    fixed_teacher_sampler = DiagonalFixedTeacherSampler(
        fixed_teacher,
        fixed_teacher_schedule,
        parallel_context=context,
    )
    motion_head_student = _student_motion_head(
        student_module,
        channels=algorithm.latent_channels,
    )
    motion_head_teacher = copy.deepcopy(motion_head_student)
    motion_head_teacher.requires_grad_(False)
    motion_head_teacher.eval()
    loss_adapter = DiagonalDMDLossAdapter(
        real_score,
        fake_score,
        objective_config,
        student_sampler=sampler,
        fixed_teacher_sampler=fixed_teacher_sampler,
        motion_head_student=motion_head_student,
        motion_head_teacher=motion_head_teacher,
    )
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="diagonal student",
    )
    fake_score_optimizer = build_post_training_optimizer(
        replace(recipe.fake_score_optimizer, gradient_accumulation_steps=1),
        fake_score_module,
        fused=fused_adamw,
        role="diagonal fake-score critic",
    )
    student_ema = DelayedModuleEMA(student_module, decay=algorithm.ema_decay)
    dmd_engine = NativeDMDTrainEngine(
        student_module=student_module,
        real_score_module=real_score_module,
        fake_score_module=fake_score_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        generator_update_interval=algorithm.generator_update_interval,
        student_max_grad_norm=recipe.optimizer.max_grad_norm,
        fake_score_max_grad_norm=recipe.fake_score_optimizer.max_grad_norm,
        gradient_accumulation_steps=accumulation_steps,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        student_scheduler_cadence=algorithm.student_scheduler_cadence,
        student_ema=student_ema,
        student_ema_start_step=algorithm.ema_start_step,
        parallel_context=context,
    )
    engine = NativeDiagonalTrainEngine(dmd_engine, sampler, loss_adapter)
    ema_state = named_stateful_collection(student=student_ema)
    if ema_state is None:
        raise RuntimeError("diagonal student EMA was not registered")
    return NativeDiagonalTrainingStack(
        recipe=recipe,
        schedule=schedule,
        fixed_teacher_schedule=fixed_teacher_schedule,
        objective_config=objective_config,
        sampler=sampler,
        fixed_teacher_sampler=fixed_teacher_sampler,
        motion_head_student=motion_head_student,
        motion_head_teacher=motion_head_teacher,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        engine=engine,
        scheduler_state=named_stateful_collection(
            student=student_scheduler,
            fake_score=fake_score_scheduler,
        ),
        ema_state=ema_state,
    )


__all__ = [
    "NativeDiagonalTrainingStack",
    "build_native_diagonal_training_stack",
]
