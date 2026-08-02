"""Construction of the WorldFoundry-native sCM-LADD training stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.scm_ladd import (
    SCMLADDAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ...shared.building import (
    build_post_training_optimizer,
    named_stateful_collection,
    require_checkpoint_identity,
    require_disjoint_trainable_parameters,
    require_independent_modules,
    validate_post_training_recipe,
)
from ...shared.distributed import PostTrainingParallelContext
from .contracts import SCMLADDDiscriminatorAdapter, TrigFlowPredictionAdapter
from .engine import NativeSCMLADDTrainEngine
from .objective import NativeSCMLADDLossAdapter


def _prediction_module(adapter: TrigFlowPredictionAdapter, *, role: str) -> torch.nn.Module:
    if not isinstance(adapter, TrigFlowPredictionAdapter):
        raise TypeError(f"{role} must implement TrigFlowPredictionAdapter")
    if not isinstance(adapter.module, torch.nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    return adapter.module


@dataclass(frozen=True, slots=True)
class NativeSCMLADDTrainingStack:
    recipe: PostTrainingRecipe
    loss_adapter: NativeSCMLADDLossAdapter
    student_optimizer: torch.optim.Optimizer
    discriminator_optimizer: torch.optim.Optimizer
    engine: NativeSCMLADDTrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }


def build_native_scm_ladd_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: TrigFlowPredictionAdapter,
    teacher: TrigFlowPredictionAdapter,
    discriminator: SCMLADDDiscriminatorAdapter,
    student_scheduler: object | None = None,
    discriminator_scheduler: object | None = None,
    student_ema: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeSCMLADDTrainingStack:
    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, SCMLADDAlgorithmSpec):
        raise TypeError("recipe.algorithm must be SCMLADDAlgorithmSpec")
    if recipe.discriminator_optimizer is None:
        raise ValueError("SCM-LADD requires discriminator_optimizer")
    accumulation_steps = recipe.optimizer.gradient_accumulation_steps
    if recipe.discriminator_optimizer.gradient_accumulation_steps != accumulation_steps:
        raise ValueError("SCM-LADD student and discriminator gradient accumulation steps must match")
    student_module = _prediction_module(student, role="SCM-LADD student")
    teacher_module = _prediction_module(teacher, role="SCM-LADD teacher")
    if not isinstance(discriminator, SCMLADDDiscriminatorAdapter):
        raise TypeError("discriminator must implement SCMLADDDiscriminatorAdapter")
    discriminator_module = discriminator.module
    feature_module = discriminator.feature_module
    if not isinstance(discriminator_module, torch.nn.Module):
        raise TypeError("discriminator.module must be an nn.Module")
    if not isinstance(feature_module, torch.nn.Module):
        raise TypeError("discriminator.feature_module must be an nn.Module")
    if feature_module is not teacher_module:
        raise ValueError("LADD discriminator must use the exact frozen teacher as its feature backbone")
    head_block_ids = tuple(int(value) for value in discriminator.head_block_ids)
    if head_block_ids != recipe.algorithm.discriminator_head_block_ids:
        raise ValueError("LADD discriminator head blocks differ from the active SCM-LADD recipe")
    if any(parameter.requires_grad for parameter in teacher_module.parameters()):
        raise ValueError("SCM-LADD teacher parameters must be frozen before stack construction")
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="SCM-LADD student",
    )
    require_checkpoint_identity(
        teacher,
        recipe.algorithm.teacher_checkpoint,
        role="SCM-LADD teacher",
    )
    require_independent_modules(
        {
            "student": student_module,
            "teacher": teacher_module,
        }
    )
    # Some adapters expose the frozen teacher feature backbone as a child of
    # the discriminator view.  That sharing is intentional; mutable optimizer
    # state must still remain disjoint from the student.
    require_disjoint_trainable_parameters(
        {
            "student": student_module,
            "discriminator": discriminator_module,
        }
    )
    teacher_module.eval()
    feature_module.eval()

    loss_adapter = NativeSCMLADDLossAdapter(
        student,
        teacher,
        discriminator,
        recipe.algorithm,
        config_digest=recipe.digest,
    )
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="SCM-LADD student",
    )
    discriminator_optimizer = build_post_training_optimizer(
        replace(recipe.discriminator_optimizer, gradient_accumulation_steps=1),
        discriminator_module,
        fused=fused_adamw,
        role="SCM-LADD discriminator",
    )
    engine = NativeSCMLADDTrainEngine(
        student_module=student_module,
        teacher_module=teacher_module,
        discriminator_module=discriminator_module,
        discriminator_feature_module=feature_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        student_max_grad_norm=recipe.optimizer.max_grad_norm,
        discriminator_max_grad_norm=recipe.discriminator_optimizer.max_grad_norm,
        gradient_accumulation_steps=accumulation_steps,
        student_scheduler=student_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        student_ema=student_ema,
        parallel_context=parallel_context,
    )
    return NativeSCMLADDTrainingStack(
        recipe=recipe,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        engine=engine,
        scheduler_state=named_stateful_collection(
            student=student_scheduler,
            discriminator=discriminator_scheduler,
        ),
        ema_state=named_stateful_collection(student=student_ema),
    )


__all__ = ["NativeSCMLADDTrainingStack", "build_native_scm_ladd_training_stack"]
