"""Fail-closed construction of the native SiD training stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.sid import SIDAlgorithmSpec
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ...shared.building import (
    build_post_training_optimizer,
    named_stateful_collection,
    validate_post_training_recipe,
)
from ...shared.distributed import PostTrainingParallelContext
from .contracts import SIDPredictionAdapter
from .engine import NativeSIDTrainEngine
from .objective import NativeSIDLossAdapter, SIDConfig


def _module(adapter: object, *, role: str) -> torch.nn.Module:
    if not isinstance(adapter, SIDPredictionAdapter):
        raise TypeError(f"{role} does not implement SIDPredictionAdapter")
    module = adapter.module
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    if not str(adapter.checkpoint_identity).strip():
        raise ValueError(f"{role}.checkpoint_identity must be non-empty")
    return module


def _require_checkpoint(adapter: object, expected: str, *, role: str) -> None:
    actual = str(adapter.checkpoint_identity).strip()
    if actual != str(expected).strip():
        raise ValueError(
            f"{role} loaded checkpoint identity {actual!r} differs from recipe {expected!r}"
        )


@dataclass(frozen=True, slots=True)
class NativeSIDTrainingStack:
    recipe: PostTrainingRecipe
    config: SIDConfig
    loss_adapter: NativeSIDLossAdapter
    student_optimizer: torch.optim.AdamW
    fake_score_optimizer: torch.optim.AdamW
    engine: NativeSIDTrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }


def build_native_sid_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: SIDPredictionAdapter,
    teacher: SIDPredictionAdapter,
    fake_score: SIDPredictionAdapter,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    student_ema: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeSIDTrainingStack:
    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, SIDAlgorithmSpec):
        raise TypeError("recipe.algorithm must be SIDAlgorithmSpec")
    if recipe.fake_score_optimizer is None:
        raise ValueError("SiD requires fake_score_optimizer")
    accumulation = recipe.optimizer.gradient_accumulation_steps
    if recipe.fake_score_optimizer.gradient_accumulation_steps != accumulation:
        raise ValueError("SiD student and fake-score gradient accumulation steps must match")
    student_module = _module(student, role="SiD student")
    teacher_module = _module(teacher, role="SiD teacher")
    fake_score_module = _module(fake_score, role="SiD fake-score")
    _require_checkpoint(student, recipe.model.checkpoint, role="SiD student")
    _require_checkpoint(teacher, recipe.algorithm.teacher_checkpoint, role="SiD teacher")
    _require_checkpoint(
        fake_score,
        recipe.algorithm.fake_score_checkpoint,
        role="SiD fake-score",
    )
    if any(parameter.requires_grad for parameter in teacher_module.parameters()):
        raise ValueError("SiD teacher parameters must be frozen before stack construction")
    teacher_module.eval()
    config = SIDConfig.from_recipe(recipe.algorithm)
    loss_adapter = NativeSIDLossAdapter(
        student,
        teacher,
        fake_score,
        config,
        config_digest=recipe.digest,
    )
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="SiD student",
    )
    fake_score_optimizer = build_post_training_optimizer(
        replace(recipe.fake_score_optimizer, gradient_accumulation_steps=1),
        fake_score_module,
        fused=fused_adamw,
        role="SiD fake-score",
    )
    engine = NativeSIDTrainEngine(
        student_module=student_module,
        teacher_module=teacher_module,
        fake_score_module=fake_score_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        student_max_grad_norm=recipe.optimizer.max_grad_norm,
        fake_score_max_grad_norm=recipe.fake_score_optimizer.max_grad_norm,
        gradient_accumulation_steps=accumulation,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        student_ema=student_ema,
        parallel_context=parallel_context,
    )
    return NativeSIDTrainingStack(
        recipe=recipe,
        config=config,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        engine=engine,
        scheduler_state=named_stateful_collection(
            student=student_scheduler,
            fake_score=fake_score_scheduler,
        ),
        ema_state=named_stateful_collection(student=student_ema),
    )


__all__ = ["NativeSIDTrainingStack", "build_native_sid_training_stack"]
