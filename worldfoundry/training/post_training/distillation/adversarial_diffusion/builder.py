"""Recipe-owned construction of the native ADD training stack."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.adversarial_diffusion import (
    AdversarialDiffusionAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ...shared.building import (
    build_post_training_optimizer,
    named_stateful_collection,
    require_checkpoint_identity,
    validate_post_training_recipe,
)
from ...shared.distributed import PostTrainingParallelContext
from .adapters import ADDTrainableRoles
from .config import ADDConfig, ADDNoiseSchedule
from .contracts import ADDDecoderAdapter, ADDDiscriminatorAdapter, ADDPredictionAdapter
from .engine import NativeADDTrainEngine
from .objective import NativeADDLossAdapter


def _scheduler(
    factory: Callable[[torch.optim.Optimizer], object] | None,
    optimizer: torch.optim.Optimizer,
    *,
    role: str,
) -> object | None:
    if factory is None:
        return None
    if not callable(factory):
        raise TypeError(f"{role} scheduler_factory must be callable or None")
    scheduler = factory(optimizer)
    for method_name in ("step", "state_dict", "load_state_dict"):
        if not callable(getattr(scheduler, method_name, None)):
            raise TypeError(f"{role} scheduler must expose step/state_dict/load_state_dict")
    bound_optimizer = getattr(scheduler, "optimizer", optimizer)
    if bound_optimizer is not optimizer:
        raise ValueError(f"{role} scheduler is bound to a different optimizer")
    return scheduler


@dataclass(frozen=True, slots=True)
class NativeADDTrainingStack:
    """Complete executable and checkpointable ADD role graph."""

    recipe: PostTrainingRecipe
    config: ADDConfig
    student_schedule: ADDNoiseSchedule
    teacher_schedule: ADDNoiseSchedule
    loss_adapter: NativeADDLossAdapter
    student_optimizer: torch.optim.Optimizer
    discriminator_optimizer: torch.optim.Optimizer
    checkpoint_model: ADDTrainableRoles
    engine: NativeADDTrainEngine
    scheduler_state: NamedStatefulCollection | None

    @property
    def optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        return self.student_optimizer, self.discriminator_optimizer

    def checkpoint_state_kwargs(self) -> dict[str, object]:
        """Arguments consumed directly by ``TrainingState`` for exact DCP resume."""

        return {
            "model": self.checkpoint_model,
            "optimizer": self.optimizers,
            "engine": self.engine,
            "ignore_frozen_parameters": True,
            "lr_scheduler": self.scheduler_state,
        }


def build_native_add_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: ADDPredictionAdapter,
    teacher: ADDPredictionAdapter,
    decoder: ADDDecoderAdapter,
    discriminator: ADDDiscriminatorAdapter,
    student_scheduler_factory: Callable[[torch.optim.Optimizer], object] | None = None,
    discriminator_scheduler_factory: Callable[[torch.optim.Optimizer], object] | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeADDTrainingStack:
    """Build schedules, roles, optimizers, and state from one strict recipe."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, AdversarialDiffusionAlgorithmSpec):
        raise TypeError("ADD stack requires AdversarialDiffusionAlgorithmSpec")
    if recipe.discriminator_optimizer is None:
        raise ValueError("ADD requires discriminator_optimizer")
    if recipe.fake_score_optimizer is not None or recipe.guidance_optimizer is not None:
        raise ValueError("ADD accepts only the primary and discriminator optimizers")
    accumulation_steps = recipe.optimizer.gradient_accumulation_steps
    if recipe.discriminator_optimizer.gradient_accumulation_steps != accumulation_steps:
        raise ValueError("ADD student and discriminator gradient accumulation steps must match")
    if not isinstance(student, ADDPredictionAdapter):
        raise TypeError("student must implement ADDPredictionAdapter")
    if not isinstance(teacher, ADDPredictionAdapter):
        raise TypeError("teacher must implement ADDPredictionAdapter")
    if not isinstance(decoder, ADDDecoderAdapter):
        raise TypeError("decoder must implement ADDDecoderAdapter")
    if not isinstance(discriminator, ADDDiscriminatorAdapter):
        raise TypeError("discriminator must implement ADDDiscriminatorAdapter")

    spec = recipe.algorithm
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="ADD student",
    )
    require_checkpoint_identity(
        teacher,
        spec.teacher_checkpoint,
        role="ADD teacher",
    )
    require_checkpoint_identity(
        decoder,
        spec.decoder_checkpoint,
        role="ADD decoder",
    )
    require_checkpoint_identity(
        discriminator,
        spec.feature_checkpoint,
        role="ADD discriminator feature network",
    )
    student_schedule = ADDNoiseSchedule(spec.student_alpha_cumprods)
    teacher_schedule = ADDNoiseSchedule(
        spec.teacher_alpha_cumprods,
        training_loss_weights=spec.teacher_training_loss_weights,
    )
    config = ADDConfig.from_recipe(spec)
    loss_adapter = NativeADDLossAdapter(
        student=student,
        teacher=teacher,
        decoder=decoder,
        discriminator=discriminator,
        student_schedule=student_schedule,
        teacher_schedule=teacher_schedule,
        config=config,
    )
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student.module,
        fused=fused_adamw,
        role="ADD student",
    )
    discriminator_optimizer = build_post_training_optimizer(
        replace(
            recipe.discriminator_optimizer,
            gradient_accumulation_steps=1,
        ),
        discriminator.module,
        fused=fused_adamw,
        role="ADD discriminator",
    )
    student_scheduler = _scheduler(
        student_scheduler_factory,
        student_optimizer,
        role="ADD student",
    )
    discriminator_scheduler = _scheduler(
        discriminator_scheduler_factory,
        discriminator_optimizer,
        role="ADD discriminator",
    )
    engine = NativeADDTrainEngine(
        student_module=student.module,
        teacher_module=teacher.module,
        decoder_module=decoder.module,
        discriminator_module=discriminator.module,
        discriminator_feature_module=discriminator.feature_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        discriminator_updates_per_generator=(config.discriminator_updates_per_generator),
        student_max_grad_norm=recipe.optimizer.max_grad_norm,
        discriminator_max_grad_norm=(recipe.discriminator_optimizer.max_grad_norm),
        gradient_accumulation_steps=accumulation_steps,
        student_scheduler=student_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        parallel_context=parallel_context,
    )
    checkpoint_model = ADDTrainableRoles(student.module, discriminator.module)
    return NativeADDTrainingStack(
        recipe=recipe,
        config=config,
        student_schedule=student_schedule,
        teacher_schedule=teacher_schedule,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        checkpoint_model=checkpoint_model,
        engine=engine,
        scheduler_state=named_stateful_collection(
            student=student_scheduler,
            discriminator=discriminator_scheduler,
        ),
    )


__all__ = ["NativeADDTrainingStack", "build_native_add_training_stack"]
