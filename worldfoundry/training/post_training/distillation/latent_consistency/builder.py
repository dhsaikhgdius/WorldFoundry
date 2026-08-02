"""Recipe-owned construction of native latent consistency distillation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import nn

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.latent_consistency import (
    LatentConsistencyAlgorithmSpec,
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
from .config import (
    LatentConsistencyConfig,
    LatentConsistencyNoiseSchedule,
)
from .contracts import LatentConsistencyPredictionAdapter
from .engine import NativeLatentConsistencyTrainEngine
from .objective import LatentConsistencyObjective


def _module(
    adapter: LatentConsistencyPredictionAdapter,
    *,
    role: str,
) -> nn.Module:
    if not isinstance(adapter, LatentConsistencyPredictionAdapter):
        raise TypeError(f"{role} must implement LatentConsistencyPredictionAdapter")
    module = adapter.module
    if not isinstance(module, nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    return module


@dataclass(frozen=True, slots=True)
class NativeLatentConsistencyTrainingStack:
    recipe: PostTrainingRecipe
    config: LatentConsistencyConfig
    noise_schedule: LatentConsistencyNoiseSchedule
    objective: LatentConsistencyObjective
    optimizer: torch.optim.Optimizer
    engine: NativeLatentConsistencyTrainEngine
    model: nn.ModuleDict
    scheduler_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": None,
            "algorithm_state": None,
        }


def build_native_latent_consistency_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: LatentConsistencyPredictionAdapter,
    teacher: LatentConsistencyPredictionAdapter,
    ema_target: LatentConsistencyPredictionAdapter,
    noise_schedule: LatentConsistencyNoiseSchedule,
    scheduler_factory: Callable[[torch.optim.Optimizer], object] | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeLatentConsistencyTrainingStack:
    """Build independent roles, objective, optimizer, EMA, and checkpoint state."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, LatentConsistencyAlgorithmSpec):
        raise TypeError(
            "latent consistency stack requires LatentConsistencyAlgorithmSpec"
        )
    if not isinstance(noise_schedule, LatentConsistencyNoiseSchedule):
        raise TypeError("noise_schedule must be LatentConsistencyNoiseSchedule")
    algorithm = recipe.algorithm
    if noise_schedule.num_train_timesteps != algorithm.num_train_timesteps:
        raise ValueError(
            "latent consistency noise schedule length differs from num_train_timesteps"
        )
    if recipe.optimizer.type != "adamw":
        raise ValueError("latent consistency requires optimizer.type='adamw'")
    modules = {
        "student": _module(student, role="latent consistency student"),
        "teacher": _module(teacher, role="latent consistency teacher"),
        "ema_target": _module(ema_target, role="latent consistency EMA target"),
    }
    require_independent_modules(modules)
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="latent consistency student",
    )
    require_checkpoint_identity(
        teacher,
        algorithm.teacher_checkpoint,
        role="latent consistency teacher",
    )
    require_checkpoint_identity(
        ema_target,
        recipe.model.checkpoint,
        role="latent consistency EMA target",
    )
    modules["teacher"].requires_grad_(False)
    modules["ema_target"].requires_grad_(False)
    modules["teacher"].eval()
    modules["ema_target"].eval()
    config = LatentConsistencyConfig(
        num_ddim_timesteps=algorithm.num_ddim_timesteps,
        prediction_type=algorithm.prediction_type,
        guidance_coefficient_min=algorithm.guidance_coefficient_min,
        guidance_coefficient_max=algorithm.guidance_coefficient_max,
        guidance_embedding_dim=algorithm.guidance_embedding_dim,
        guidance_embedding_scale=algorithm.guidance_embedding_scale,
        guidance_embedding_max_period=(
            algorithm.guidance_embedding_max_period
        ),
        sigma_data=algorithm.sigma_data,
        timestep_scaling=algorithm.timestep_scaling,
        loss_type=algorithm.loss_type,
        pseudo_huber_c=algorithm.pseudo_huber_c,
        ema_decay=algorithm.ema_decay,
    )
    objective = LatentConsistencyObjective(
        student=student,
        teacher=teacher,
        ema_target=ema_target,
        noise_schedule=noise_schedule,
        config=config,
    )
    optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        modules["student"],
        fused=fused_adamw,
        role="latent consistency student",
    )
    if scheduler_factory is not None and not callable(scheduler_factory):
        raise TypeError("scheduler_factory must be callable or None")
    scheduler = None if scheduler_factory is None else scheduler_factory(optimizer)
    engine = NativeLatentConsistencyTrainEngine(
        student_module=modules["student"],
        teacher_module=modules["teacher"],
        ema_target_module=modules["ema_target"],
        objective=objective,
        optimizer=optimizer,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        gradient_accumulation_steps=(
            recipe.optimizer.gradient_accumulation_steps
        ),
        scheduler=scheduler,
        parallel_context=parallel_context,
        seed=recipe.data.shuffle_seed,
        initialize_ema_target=True,
    )
    model = nn.ModuleDict(modules)
    return NativeLatentConsistencyTrainingStack(
        recipe=recipe,
        config=config,
        noise_schedule=noise_schedule,
        objective=objective,
        optimizer=optimizer,
        engine=engine,
        model=model,
        scheduler_state=named_stateful_collection(student=scheduler),
    )


__all__ = [
    "NativeLatentConsistencyTrainingStack",
    "build_native_latent_consistency_training_stack",
]
