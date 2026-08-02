"""Recipe-owned construction for online causal consistency distillation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.causal_consistency import (
    CausalConsistencyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ...shared.building import (
    build_post_training_optimizer,
    named_stateful_collection,
    require_checkpoint_identity,
    require_independent_modules,
    validate_post_training_recipe,
)
from ...shared.distributed import PostTrainingParallelContext
from ..causal.contracts import (
    CausalCleanPredictionAdapter,
    CausalVelocityPredictionAdapter,
)
from .config import CausalConsistencyConfig
from .engine import NativeCausalConsistencyTrainEngine
from .objective import CausalConsistencyObjective


@dataclass(frozen=True, slots=True)
class NativeCausalConsistencyTrainingStack:
    recipe: PostTrainingRecipe
    config: CausalConsistencyConfig
    objective: CausalConsistencyObjective
    optimizer: torch.optim.Optimizer
    engine: NativeCausalConsistencyTrainEngine
    scheduler_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": None,
            "algorithm_state": None,
        }


def build_native_causal_consistency_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: CausalCleanPredictionAdapter,
    teacher: CausalVelocityPredictionAdapter,
    ema_student: CausalCleanPredictionAdapter,
    scheduler: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeCausalConsistencyTrainingStack:
    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, CausalConsistencyAlgorithmSpec):
        raise TypeError(
            "causal consistency stack requires CausalConsistencyAlgorithmSpec"
        )
    if not isinstance(student, CausalCleanPredictionAdapter):
        raise TypeError("student must implement CausalCleanPredictionAdapter")
    if not isinstance(teacher, CausalVelocityPredictionAdapter):
        raise TypeError("teacher must implement CausalVelocityPredictionAdapter")
    if not isinstance(ema_student, CausalCleanPredictionAdapter):
        raise TypeError("ema_student must implement CausalCleanPredictionAdapter")
    modules = {
        "student": student.module,
        "teacher": teacher.module,
        "ema-student": ema_student.module,
    }
    require_independent_modules(modules)
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="causal consistency student",
    )
    require_checkpoint_identity(
        teacher,
        recipe.algorithm.teacher_checkpoint,
        role="causal consistency teacher",
    )
    require_checkpoint_identity(
        ema_student,
        recipe.model.checkpoint,
        role="causal consistency EMA student",
    )
    config = CausalConsistencyConfig(
        num_levels=recipe.algorithm.num_levels,
        num_train_timesteps=recipe.algorithm.num_train_timesteps,
        flow_shift=recipe.algorithm.flow_shift,
        extra_terminal_step=recipe.algorithm.extra_terminal_step,
        guidance_scale=recipe.algorithm.guidance_scale,
        ema_decay=recipe.algorithm.ema_decay,
        frame_dim=recipe.algorithm.frame_dim,
    )
    objective = CausalConsistencyObjective(
        student=student,
        teacher=teacher,
        ema_student=ema_student,
        config=config,
    )
    student_module = modules["student"]
    assert isinstance(student_module, torch.nn.Module)
    optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="causal consistency student",
    )
    engine = NativeCausalConsistencyTrainEngine(
        student_module=student_module,
        teacher_module=modules["teacher"],
        ema_student_module=modules["ema-student"],
        objective=objective,
        optimizer=optimizer,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        gradient_accumulation_steps=recipe.optimizer.gradient_accumulation_steps,
        scheduler=scheduler,
        parallel_context=parallel_context,
        seed=recipe.data.shuffle_seed,
        initialize_ema_target=True,
    )
    return NativeCausalConsistencyTrainingStack(
        recipe=recipe,
        config=config,
        objective=objective,
        optimizer=optimizer,
        engine=engine,
        scheduler_state=named_stateful_collection(student=scheduler),
    )


__all__ = [
    "NativeCausalConsistencyTrainingStack",
    "build_native_causal_consistency_training_stack",
]
