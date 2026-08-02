"""Recipe-owned construction of progressive DDIM distillation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import nn

from worldfoundry.training.recipes.post_training.algorithms.progressive import (
    ProgressiveDistillationAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ...shared.building import (
    build_post_training_optimizer,
    require_checkpoint_identity,
    require_independent_modules,
    validate_post_training_recipe,
)
from ...shared.distributed import PostTrainingParallelContext
from .config import ProgressiveDistillationConfig
from .contracts import ProgressivePredictionAdapter
from .engine import NativeProgressiveDistillationTrainEngine
from .objective import ProgressiveDistillationObjective


def _module(
    adapter: ProgressivePredictionAdapter,
    *,
    role: str,
) -> nn.Module:
    if not isinstance(adapter, ProgressivePredictionAdapter):
        raise TypeError(f"{role} must implement ProgressivePredictionAdapter")
    module = adapter.module
    if not isinstance(module, nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    return module


@dataclass(frozen=True, slots=True)
class NativeProgressiveDistillationTrainingStack:
    recipe: PostTrainingRecipe
    config: ProgressiveDistillationConfig
    objective: ProgressiveDistillationObjective
    optimizer: torch.optim.Optimizer
    engine: NativeProgressiveDistillationTrainEngine
    model: nn.ModuleDict

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": None,
            "ema": None,
            "algorithm_state": None,
        }


def build_native_progressive_distillation_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: ProgressivePredictionAdapter,
    teacher: ProgressivePredictionAdapter,
    ema_target: ProgressivePredictionAdapter,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeProgressiveDistillationTrainingStack:
    """Build one strict repeated-halving stack without an external trainer."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, ProgressiveDistillationAlgorithmSpec):
        raise TypeError(
            "progressive stack requires ProgressiveDistillationAlgorithmSpec"
        )
    if recipe.optimizer.type != "adamw":
        raise ValueError("progressive distillation requires optimizer.type='adamw'")
    spec = recipe.algorithm
    if recipe.model.checkpoint != spec.teacher_checkpoint:
        raise ValueError(
            "progressive student must initialize from teacher_checkpoint"
        )
    modules = {
        "student": _module(student, role="progressive student"),
        "teacher": _module(teacher, role="progressive teacher"),
        "ema_target": _module(ema_target, role="progressive EMA target"),
    }
    require_independent_modules(modules)
    for name, adapter in (
        ("student", student),
        ("teacher", teacher),
        ("EMA target", ema_target),
    ):
        require_checkpoint_identity(
            adapter,
            spec.teacher_checkpoint,
            role=f"progressive {name}",
        )
    modules["teacher"].requires_grad_(False)
    modules["ema_target"].requires_grad_(False)
    modules["teacher"].eval()
    modules["ema_target"].eval()
    config = ProgressiveDistillationConfig(
        start_num_steps=spec.start_num_steps,
        end_num_steps=spec.end_num_steps,
        optimizer_steps_per_stage=spec.optimizer_steps_per_stage,
        prediction_type=spec.prediction_type,
        loss_weight=spec.loss_weight,
        logsnr_min=spec.logsnr_min,
        logsnr_max=spec.logsnr_max,
        ema_decay=spec.ema_decay,
        learning_rate_anneal=spec.learning_rate_anneal,
    )
    objective = ProgressiveDistillationObjective(
        student=student,
        teacher=teacher,
        config=config,
    )
    optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        modules["student"],
        fused=fused_adamw,
        role="progressive student",
    )
    engine = NativeProgressiveDistillationTrainEngine(
        student_module=modules["student"],
        teacher_module=modules["teacher"],
        ema_target_module=modules["ema_target"],
        objective=objective,
        optimizer=optimizer,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        gradient_accumulation_steps=(
            recipe.optimizer.gradient_accumulation_steps
        ),
        parallel_context=parallel_context,
        seed=recipe.data.shuffle_seed,
        initialize_student_from_teacher=True,
        initialize_ema_target=True,
    )
    return NativeProgressiveDistillationTrainingStack(
        recipe=recipe,
        config=config,
        objective=objective,
        optimizer=optimizer,
        engine=engine,
        model=nn.ModuleDict(modules),
    )


__all__ = [
    "NativeProgressiveDistillationTrainingStack",
    "build_native_progressive_distillation_training_stack",
]
