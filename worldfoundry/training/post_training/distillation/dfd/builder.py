"""Recipe-owned construction of native Data-Forcing Distillation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.dfd import (
    DFDAlgorithmSpec,
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
from .config import DFDConfig
from .contracts import (
    DFDDiscriminatorAdapter,
    DFDFakeScoreAdapter,
    DFDPredictionAdapter,
)
from .engine import NativeDFDTrainEngine
from .objective import NativeDFDLossAdapter


def _prediction_module(
    adapter: DFDPredictionAdapter,
    *,
    role: str,
) -> torch.nn.Module:
    if not isinstance(adapter, DFDPredictionAdapter):
        raise TypeError(f"{role} must implement DFDPredictionAdapter")
    if not isinstance(adapter.module, torch.nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    return adapter.module


def _discriminator_module(
    adapter: DFDDiscriminatorAdapter,
) -> torch.nn.Module:
    if not isinstance(adapter, DFDDiscriminatorAdapter):
        raise TypeError("discriminator must implement DFDDiscriminatorAdapter")
    if not isinstance(adapter.module, torch.nn.Module):
        raise TypeError("discriminator.module must be an nn.Module")
    return adapter.module


@dataclass(frozen=True, slots=True)
class NativeDFDTrainingStack:
    recipe: PostTrainingRecipe
    config: DFDConfig
    loss_adapter: NativeDFDLossAdapter
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer
    discriminator_optimizer: torch.optim.Optimizer | None
    engine: NativeDFDTrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection | None

    @property
    def optimizers(self) -> tuple[torch.optim.Optimizer, ...]:
        values = [self.student_optimizer, self.fake_score_optimizer]
        if self.discriminator_optimizer is not None:
            values.append(self.discriminator_optimizer)
        return tuple(values)

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }


def build_native_dfd_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: DFDPredictionAdapter,
    teacher: DFDPredictionAdapter,
    fake_score: DFDFakeScoreAdapter,
    discriminator: DFDDiscriminatorAdapter | None = None,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    discriminator_scheduler: object | None = None,
    student_ema: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeDFDTrainingStack:
    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, DFDAlgorithmSpec):
        raise TypeError("DFD stack requires DFDAlgorithmSpec")
    if recipe.fake_score_optimizer is None:
        raise ValueError("DFD requires fake_score_optimizer")
    if not isinstance(fake_score, DFDFakeScoreAdapter):
        raise TypeError("fake_score must implement DFDFakeScoreAdapter")
    algorithm = recipe.algorithm
    if algorithm.adversarial_enabled != (discriminator is not None):
        raise ValueError("DFD discriminator topology differs from the recipe")
    if algorithm.adversarial_enabled != (recipe.discriminator_optimizer is not None):
        raise ValueError("DFD discriminator optimizer topology differs from the recipe")
    if discriminator is None and discriminator_scheduler is not None:
        raise ValueError("discriminator_scheduler requires adversarial DFD")

    student_module = _prediction_module(student, role="DFD student")
    teacher_module = _prediction_module(teacher, role="DFD teacher")
    fake_score_module = _prediction_module(fake_score, role="DFD fake score")
    discriminator_module = (
        None if discriminator is None else _discriminator_module(discriminator)
    )
    modules = {
        "student": student_module,
        "teacher": teacher_module,
        "fake-score": fake_score_module,
    }
    if discriminator_module is not None:
        modules["discriminator"] = discriminator_module
    require_independent_modules(modules)
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="DFD student",
    )
    require_checkpoint_identity(
        teacher,
        algorithm.teacher_checkpoint,
        role="DFD teacher",
    )
    require_checkpoint_identity(
        fake_score,
        algorithm.fake_score_checkpoint,
        role="DFD fake score",
    )
    if discriminator is not None:
        assert algorithm.discriminator_checkpoint is not None
        require_checkpoint_identity(
            discriminator,
            algorithm.discriminator_checkpoint,
            role="DFD discriminator",
        )
    adapters = (student, teacher, fake_score)
    kinds = {
        str(adapter.noise_process_kind).strip().lower().replace("_", "-")
        for adapter in adapters
    }
    digests = {str(adapter.noise_process_digest).strip() for adapter in adapters}
    if kinds != {"flow-matching"} or "" in digests or len(digests) != 1:
        raise ValueError("DFD prediction roles must expose one matching flow process")
    if any(parameter.requires_grad for parameter in teacher_module.parameters()):
        raise ValueError("DFD teacher parameters must be frozen before stack construction")
    teacher_module.eval()

    accumulation = recipe.optimizer.gradient_accumulation_steps
    optimizer_specs = [recipe.fake_score_optimizer]
    if recipe.discriminator_optimizer is not None:
        optimizer_specs.append(recipe.discriminator_optimizer)
    if any(spec.gradient_accumulation_steps != accumulation for spec in optimizer_specs):
        raise ValueError("DFD optimizer gradient accumulation steps must match")
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="DFD student",
    )
    fake_score_optimizer = build_post_training_optimizer(
        replace(recipe.fake_score_optimizer, gradient_accumulation_steps=1),
        fake_score_module,
        fused=fused_adamw,
        role="DFD fake score",
    )
    discriminator_optimizer = (
        None
        if discriminator_module is None or recipe.discriminator_optimizer is None
        else build_post_training_optimizer(
            replace(
                recipe.discriminator_optimizer,
                gradient_accumulation_steps=1,
            ),
            discriminator_module,
            fused=fused_adamw,
            role="DFD discriminator",
        )
    )
    config = DFDConfig.from_recipe(algorithm)
    loss_adapter = NativeDFDLossAdapter(
        student,
        teacher,
        fake_score,
        config,
        discriminator=discriminator,
    )
    engine = NativeDFDTrainEngine(
        student_module=student_module,
        teacher_module=teacher_module,
        fake_score_module=fake_score_module,
        discriminator_module=discriminator_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        student_max_grad_norm=recipe.optimizer.max_grad_norm,
        fake_score_max_grad_norm=recipe.fake_score_optimizer.max_grad_norm,
        discriminator_max_grad_norm=(
            None
            if recipe.discriminator_optimizer is None
            else recipe.discriminator_optimizer.max_grad_norm
        ),
        gradient_accumulation_steps=accumulation,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        student_ema=student_ema,
        parallel_context=parallel_context,
    )
    return NativeDFDTrainingStack(
        recipe=recipe,
        config=config,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        engine=engine,
        scheduler_state=named_stateful_collection(
            student=student_scheduler,
            fake_score=fake_score_scheduler,
            discriminator=discriminator_scheduler,
        ),
        ema_state=named_stateful_collection(student=student_ema),
    )


__all__ = ["NativeDFDTrainingStack", "build_native_dfd_training_stack"]
