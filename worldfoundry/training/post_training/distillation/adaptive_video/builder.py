"""Fail-closed construction of native adaptive video distillation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.adaptive_video import (
    AdaptiveVideoAlgorithmSpec,
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
from ..dmd.objective import DMDStudentSampler
from .config import AdaptiveVideoConfig
from .engine import NativeAdaptiveVideoTrainEngine
from .objective import FlowAdaptiveVideoLossAdapter


@dataclass(frozen=True, slots=True)
class NativeAdaptiveVideoTrainingStack:
    recipe: PostTrainingRecipe
    config: AdaptiveVideoConfig
    loss_adapter: FlowAdaptiveVideoLossAdapter
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer
    engine: NativeAdaptiveVideoTrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }


def build_native_adaptive_video_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: FlowPredictionAdapter,
    real_score: FlowPredictionAdapter,
    fake_score: FlowPredictionAdapter,
    student_sampler: DMDStudentSampler | None = None,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    student_ema: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeAdaptiveVideoTrainingStack:
    """Build every trainable role from the WorldFoundry recipe and adapters."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, AdaptiveVideoAlgorithmSpec):
        raise TypeError(
            "adaptive video stack requires AdaptiveVideoAlgorithmSpec"
        )
    if recipe.fake_score_optimizer is None:
        raise ValueError("adaptive video distillation requires fake_score_optimizer")
    accumulation_steps = recipe.optimizer.gradient_accumulation_steps
    if recipe.fake_score_optimizer.gradient_accumulation_steps != accumulation_steps:
        raise ValueError(
            "adaptive video student and fake-score accumulation steps must match"
        )
    student_module = prediction_module(student, role="adaptive video student")
    real_score_module = prediction_module(
        real_score,
        role="adaptive video real-score teacher",
    )
    fake_score_module = prediction_module(
        fake_score,
        role="adaptive video fake-score critic",
    )
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="adaptive video student",
    )
    require_checkpoint_identity(
        real_score,
        recipe.algorithm.real_score_checkpoint,
        role="adaptive video real-score teacher",
    )
    require_checkpoint_identity(
        fake_score,
        recipe.algorithm.fake_score_checkpoint,
        role="adaptive video fake-score critic",
    )
    require_independent_modules(
        {
            "student": student_module,
            "real-score": real_score_module,
            "fake-score": fake_score_module,
        }
    )
    if any(parameter.requires_grad for parameter in real_score_module.parameters()):
        raise ValueError(
            "adaptive video real-score teacher must be frozen before stack construction"
        )
    real_score_module.eval()
    config = AdaptiveVideoConfig.from_recipe(recipe.algorithm)
    loss_adapter = FlowAdaptiveVideoLossAdapter(
        student,
        real_score,
        fake_score,
        config,
        student_sampler=student_sampler,
        config_digest=recipe.digest,
    )
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="adaptive video student",
    )
    fake_score_optimizer = build_post_training_optimizer(
        replace(recipe.fake_score_optimizer, gradient_accumulation_steps=1),
        fake_score_module,
        fused=fused_adamw,
        role="adaptive video fake-score critic",
    )
    engine = NativeAdaptiveVideoTrainEngine(
        student_module=student_module,
        real_score_module=real_score_module,
        fake_score_module=fake_score_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        generator_update_interval=recipe.algorithm.generator_update_interval,
        student_max_grad_norm=recipe.optimizer.max_grad_norm,
        fake_score_max_grad_norm=recipe.fake_score_optimizer.max_grad_norm,
        gradient_accumulation_steps=accumulation_steps,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        student_scheduler_cadence=recipe.algorithm.student_scheduler_cadence,
        student_ema=student_ema,
        parallel_context=parallel_context,
    )
    return NativeAdaptiveVideoTrainingStack(
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


__all__ = [
    "NativeAdaptiveVideoTrainingStack",
    "build_native_adaptive_video_training_stack",
]
