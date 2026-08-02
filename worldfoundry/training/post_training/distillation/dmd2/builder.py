"""Fail-closed construction of the native DMD2 training stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.dmd2 import DMD2AlgorithmSpec
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ...shared.building import (
    build_post_training_optimizer,
    named_stateful_collection,
    validate_post_training_recipe,
)
from ...shared.distributed import PostTrainingParallelContext
from .contracts import DMD2GuidanceAdapter, DMD2PredictionAdapter
from .engine import NativeDMD2TrainEngine
from .objective import DMD2Config, NativeDMD2LossAdapter


def _module(adapter: object, protocol: type, *, role: str) -> torch.nn.Module:
    if not isinstance(adapter, protocol):
        raise TypeError(f"{role} does not implement its DMD2 functional adapter")
    module = adapter.module
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    checkpoint_identity = str(adapter.checkpoint_identity).strip()
    if not checkpoint_identity:
        raise ValueError(f"{role}.checkpoint_identity must be non-empty")
    return module


def _require_checkpoint(adapter: object, expected: str, *, role: str) -> None:
    actual = str(adapter.checkpoint_identity).strip()
    if actual != str(expected).strip():
        raise ValueError(
            f"{role} loaded checkpoint identity {actual!r} differs from recipe {expected!r}"
        )


@dataclass(frozen=True, slots=True)
class NativeDMD2TrainingStack:
    recipe: PostTrainingRecipe
    config: DMD2Config
    loss_adapter: NativeDMD2LossAdapter
    student_optimizer: torch.optim.Optimizer
    guidance_optimizer: torch.optim.Optimizer
    engine: NativeDMD2TrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }


def build_native_dmd2_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: DMD2PredictionAdapter,
    real_score: DMD2PredictionAdapter,
    guidance: DMD2GuidanceAdapter,
    student_scheduler: object | None = None,
    guidance_scheduler: object | None = None,
    student_ema: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeDMD2TrainingStack:
    """Build only from real native adapters whose loaded identities are known."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, DMD2AlgorithmSpec):
        raise TypeError("recipe.algorithm must be DMD2AlgorithmSpec")
    if recipe.guidance_optimizer is None:
        raise ValueError("DMD2 requires guidance_optimizer")
    accumulation_steps = recipe.optimizer.gradient_accumulation_steps
    if recipe.guidance_optimizer.gradient_accumulation_steps != accumulation_steps:
        raise ValueError("DMD2 student and guidance gradient accumulation steps must match")
    student_module = _module(student, DMD2PredictionAdapter, role="DMD2 student")
    teacher_module = _module(real_score, DMD2PredictionAdapter, role="DMD2 real-score teacher")
    guidance_module = _module(guidance, DMD2GuidanceAdapter, role="DMD2 guidance")
    _require_checkpoint(student, recipe.model.checkpoint, role="DMD2 student")
    _require_checkpoint(
        real_score,
        recipe.algorithm.real_score_checkpoint,
        role="DMD2 real-score teacher",
    )
    _require_checkpoint(
        guidance,
        recipe.algorithm.guidance_checkpoint,
        role="DMD2 guidance",
    )
    if any(parameter.requires_grad for parameter in teacher_module.parameters()):
        raise ValueError("DMD2 teacher parameters must be frozen before stack construction")
    teacher_module.eval()
    config = DMD2Config.from_recipe(recipe.algorithm)
    loss_adapter = NativeDMD2LossAdapter(
        student,
        real_score,
        guidance,
        config,
        config_digest=recipe.digest,
    )
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="DMD2 student",
    )
    guidance_optimizer = build_post_training_optimizer(
        replace(recipe.guidance_optimizer, gradient_accumulation_steps=1),
        guidance_module,
        fused=fused_adamw,
        role="DMD2 guidance",
    )
    engine = NativeDMD2TrainEngine(
        student_module=student_module,
        teacher_module=teacher_module,
        guidance_module=guidance_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        guidance_optimizer=guidance_optimizer,
        generator_update_interval=recipe.algorithm.generator_update_interval,
        student_max_grad_norm=recipe.optimizer.max_grad_norm,
        guidance_max_grad_norm=recipe.guidance_optimizer.max_grad_norm,
        gradient_accumulation_steps=accumulation_steps,
        student_scheduler=student_scheduler,
        guidance_scheduler=guidance_scheduler,
        student_scheduler_cadence=recipe.algorithm.student_scheduler_cadence,
        student_ema=student_ema,
        parallel_context=parallel_context,
    )
    return NativeDMD2TrainingStack(
        recipe=recipe,
        config=config,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        guidance_optimizer=guidance_optimizer,
        engine=engine,
        scheduler_state=named_stateful_collection(
            student=student_scheduler,
            guidance=guidance_scheduler,
        ),
        ema_state=named_stateful_collection(student=student_ema),
    )


__all__ = ["NativeDMD2TrainingStack", "build_native_dmd2_training_stack"]
