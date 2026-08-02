"""Construction of the complete WorldFoundry-native DMD training stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.dmd import DMDAlgorithmSpec
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
from .engine import NativeDMDTrainEngine
from .objective import DMDConfig, FewStepSchedule, FlowDMDLossAdapter


@dataclass(frozen=True, slots=True)
class NativeDMDTrainingStack:
    """Fully constructed DMD math and optimizer execution plane."""

    recipe: PostTrainingRecipe
    config: DMDConfig
    loss_adapter: FlowDMDLossAdapter
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer
    engine: NativeDMDTrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }


def build_native_dmd_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: FlowPredictionAdapter,
    real_score: FlowPredictionAdapter,
    fake_score: FlowPredictionAdapter,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    student_ema: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeDMDTrainingStack:
    """Build DMD directly from loaded native model roles and a strict recipe."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, DMDAlgorithmSpec):
        raise TypeError("DMD stack requires a DMDAlgorithmSpec recipe")
    if recipe.fake_score_optimizer is None:
        raise ValueError("DMD requires fake_score_optimizer")
    accumulation_steps = recipe.optimizer.gradient_accumulation_steps
    if recipe.fake_score_optimizer.gradient_accumulation_steps != accumulation_steps:
        raise ValueError("DMD student and fake-score gradient accumulation steps must match")
    student_module = prediction_module(student, role="DMD student")
    real_score_module = prediction_module(real_score, role="DMD real-score teacher")
    fake_score_module = prediction_module(fake_score, role="DMD fake-score critic")
    algorithm = recipe.algorithm
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="DMD student",
    )
    require_checkpoint_identity(
        real_score,
        algorithm.real_score_checkpoint,
        role="DMD real-score teacher",
    )
    require_checkpoint_identity(
        fake_score,
        algorithm.fake_score_checkpoint,
        role="DMD fake-score critic",
    )
    require_independent_modules(
        {
            "student": student_module,
            "real-score": real_score_module,
            "fake-score": fake_score_module,
        }
    )
    if any(parameter.requires_grad for parameter in real_score_module.parameters()):
        raise ValueError("DMD real-score teacher must be frozen before stack construction")
    real_score_module.eval()
    config = DMDConfig(
        schedule=FewStepSchedule(
            algorithm.student_timesteps,
            algorithm.student_sigmas,
        ),
        num_train_timesteps=algorithm.num_train_timesteps,
        score_min_sigma=algorithm.score_min_sigma,
        score_max_sigma=algorithm.score_max_sigma,
        score_flow_shift=algorithm.score_flow_shift,
        teacher_guidance_scale=algorithm.teacher_guidance_scale,
        normalization_epsilon=algorithm.normalization_epsilon,
        shared_score_timestep=algorithm.shared_score_timestep,
    )
    loss_adapter = FlowDMDLossAdapter(student, real_score, fake_score, config)
    # Accumulation is an engine cadence, not optimizer state.  Construct each
    # optimizer from the otherwise identical single-microbatch view.
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="DMD student",
    )
    fake_score_optimizer = build_post_training_optimizer(
        replace(recipe.fake_score_optimizer, gradient_accumulation_steps=1),
        fake_score_module,
        fused=fused_adamw,
        role="DMD fake-score critic",
    )
    engine = NativeDMDTrainEngine(
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
        parallel_context=parallel_context,
    )
    return NativeDMDTrainingStack(
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


__all__ = ["NativeDMDTrainingStack", "build_native_dmd_training_stack"]
