"""Recipe-owned construction for causal ODE distillation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.causal_ode import (
    CausalODEAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ...shared.building import (
    build_post_training_optimizer,
    named_stateful_collection,
    require_checkpoint_identity,
    validate_post_training_recipe,
)
from ...shared.distributed import PostTrainingParallelContext
from ..causal.contracts import CausalCleanPredictionAdapter
from .config import CausalODEConfig, warped_causal_ode_timesteps
from .engine import NativeCausalODETrainEngine
from .objective import CausalODEObjective


@dataclass(frozen=True, slots=True)
class NativeCausalODETrainingStack:
    recipe: PostTrainingRecipe
    config: CausalODEConfig
    objective: CausalODEObjective
    optimizer: torch.optim.Optimizer
    engine: NativeCausalODETrainEngine
    scheduler_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": None,
            "algorithm_state": None,
        }


def build_native_causal_ode_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: CausalCleanPredictionAdapter,
    scheduler: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeCausalODETrainingStack:
    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, CausalODEAlgorithmSpec):
        raise TypeError("causal ODE stack requires CausalODEAlgorithmSpec")
    if not isinstance(student, CausalCleanPredictionAdapter):
        raise TypeError("student must implement CausalCleanPredictionAdapter")
    module = student.module
    if not isinstance(module, torch.nn.Module):
        raise TypeError("causal ODE student.module must be an nn.Module")
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="causal ODE student",
    )
    algorithm = recipe.algorithm
    config = CausalODEConfig(
        trajectory_timesteps=warped_causal_ode_timesteps(
            algorithm.raw_denoising_steps,
            num_train_timesteps=algorithm.num_train_timesteps,
            flow_shift=algorithm.flow_shift,
            extra_terminal_step=algorithm.extra_terminal_step,
        ),
        frame_dim=algorithm.frame_dim,
    )
    objective = CausalODEObjective(student, config)
    optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        module,
        fused=fused_adamw,
        role="causal ODE student",
    )
    engine = NativeCausalODETrainEngine(
        student_module=module,
        objective=objective,
        optimizer=optimizer,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        gradient_accumulation_steps=recipe.optimizer.gradient_accumulation_steps,
        scheduler=scheduler,
        parallel_context=parallel_context,
        seed=recipe.data.shuffle_seed,
    )
    return NativeCausalODETrainingStack(
        recipe=recipe,
        config=config,
        objective=objective,
        optimizer=optimizer,
        engine=engine,
        scheduler_state=named_stateful_collection(student=scheduler),
    )


__all__ = [
    "NativeCausalODETrainingStack",
    "build_native_causal_ode_training_stack",
]
