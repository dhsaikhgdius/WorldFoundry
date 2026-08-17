"""Fail-closed construction of native scale-wise distillation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.scale_wise import (
    ScaleWiseAlgorithmSpec,
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
from .config import ScaleWiseConfig
from .contracts import ScaleWiseCriticAdapter, ScaleWisePredictionAdapter
from .engine import NativeScaleWiseTrainEngine
from .objective import FlowScaleWiseLossAdapter


@dataclass(frozen=True, slots=True)
class NativeScaleWiseTrainingStack:
    recipe: PostTrainingRecipe
    config: ScaleWiseConfig
    loss_adapter: FlowScaleWiseLossAdapter
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer | None
    engine: NativeScaleWiseTrainEngine
    scheduler_state: NamedStatefulCollection | None

    @property
    def optimizers(self) -> tuple[torch.optim.Optimizer, ...]:
        if self.fake_score_optimizer is None:
            return (self.student_optimizer,)
        return (self.student_optimizer, self.fake_score_optimizer)

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": None,
            "algorithm_state": None,
        }


def build_native_scale_wise_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: ScaleWisePredictionAdapter,
    teacher: ScaleWisePredictionAdapter,
    fake_score: ScaleWiseCriticAdapter,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeScaleWiseTrainingStack:
    """Build progressive SwD roles and optimizers from one strict recipe."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, ScaleWiseAlgorithmSpec):
        raise TypeError("scale-wise stack requires ScaleWiseAlgorithmSpec")
    if not isinstance(student, ScaleWisePredictionAdapter):
        raise TypeError("student must implement ScaleWisePredictionAdapter")
    if not isinstance(teacher, ScaleWisePredictionAdapter):
        raise TypeError("teacher must implement ScaleWisePredictionAdapter")
    if not isinstance(fake_score, ScaleWiseCriticAdapter):
        raise TypeError("fake_score must implement ScaleWiseCriticAdapter")
    spec = recipe.algorithm
    if spec.dmd_enabled:
        if recipe.fake_score_optimizer is None:
            raise ValueError("scale-wise DMD requires fake_score_optimizer")
        if (
            recipe.fake_score_optimizer.gradient_accumulation_steps
            != recipe.optimizer.gradient_accumulation_steps
        ):
            raise ValueError(
                "scale-wise student and fake-score accumulation steps must match"
            )
    elif recipe.fake_score_optimizer is not None:
        raise ValueError("MMD-only scale-wise training does not update fake score")
    if not spec.dmd_enabled and fake_score_scheduler is not None:
        raise ValueError("fake_score_scheduler requires scale-wise DMD")

    modules = {
        "student": student.module,
        "teacher": teacher.module,
        "fake-score": fake_score.module,
    }
    if not all(isinstance(module, torch.nn.Module) for module in modules.values()):
        raise TypeError("every scale-wise adapter.module must be an nn.Module")
    student_module = modules["student"]
    teacher_module = modules["teacher"]
    fake_module = modules["fake-score"]
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="scale-wise student",
    )
    require_checkpoint_identity(
        teacher,
        spec.teacher_checkpoint,
        role="scale-wise teacher",
    )
    require_checkpoint_identity(
        fake_score,
        spec.fake_score_checkpoint,
        role="scale-wise fake score",
    )
    require_independent_modules(
        {"student": student_module, "fake-score": fake_module}
    )
    require_disjoint_trainable_parameters(
        {
            "student": student_module,
            "teacher": teacher_module,
            "fake-score": fake_module,
        }
    )
    if any(parameter.requires_grad for parameter in teacher_module.parameters()):
        raise ValueError("scale-wise teacher must expose only frozen parameters")
    if not spec.dmd_enabled and any(
        parameter.requires_grad for parameter in fake_module.parameters()
    ):
        raise ValueError("MMD-only scale-wise feature extractor must be frozen")
    teacher_module.eval()
    fake_score.audit_scale_wise_critic(
        classifier_blocks=spec.classifier_blocks,
        mmd_blocks=spec.mmd_blocks,
        discriminator_layers=spec.discriminator_layers,
    )
    config = ScaleWiseConfig.from_recipe(spec)
    loss_adapter = FlowScaleWiseLossAdapter(
        student,
        teacher,
        fake_score,
        config,
    )
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="scale-wise student",
    )
    if recipe.fake_score_optimizer is None:
        fake_optimizer = None
    else:
        fake_optimizer = build_post_training_optimizer(
            replace(
                recipe.fake_score_optimizer,
                gradient_accumulation_steps=1,
            ),
            fake_module,
            fused=fused_adamw,
            role="scale-wise fake score",
        )
    context = parallel_context or PostTrainingParallelContext.current()
    engine = NativeScaleWiseTrainEngine(
        student_module=student_module,
        teacher_module=teacher_module,
        fake_score_module=fake_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_optimizer,
        student_max_grad_norm=recipe.optimizer.max_grad_norm,
        fake_score_max_grad_norm=(
            recipe.optimizer.max_grad_norm
            if recipe.fake_score_optimizer is None
            else recipe.fake_score_optimizer.max_grad_norm
        ),
        gradient_accumulation_steps=recipe.optimizer.gradient_accumulation_steps,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        parallel_context=context,
    )
    return NativeScaleWiseTrainingStack(
        recipe=recipe,
        config=config,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_optimizer,
        engine=engine,
        scheduler_state=named_stateful_collection(
            student=student_scheduler,
            fake_score=fake_score_scheduler,
        ),
    )


__all__ = [
    "NativeScaleWiseTrainingStack",
    "build_native_scale_wise_training_stack",
]
