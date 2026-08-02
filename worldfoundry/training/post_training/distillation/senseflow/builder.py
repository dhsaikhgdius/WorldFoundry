"""Fail-closed construction of the native SenseFlow training stack."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch
from torch import nn

from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.optimization import build_adamw, trainable_parameters
from worldfoundry.training.recipes.post_training.algorithms.senseflow import (
    SenseFlowAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ...shared.building import (
    require_checkpoint_identity,
    require_disjoint_trainable_parameters,
    require_independent_modules,
    validate_post_training_recipe,
)
from ...shared.distributed import PostTrainingParallelContext
from .config import SenseFlowConfig, SenseFlowOptimizerConfig
from .contracts import (
    SenseFlowDiscriminatorAdapter,
    SenseFlowFakeScoreAdapter,
    SenseFlowPredictionAdapter,
    SenseFlowTeacherAdapter,
    SenseFlowTrainingBatch,
)
from .engine import NativeSenseFlowTrainEngine
from .objective import NativeSenseFlowLossAdapter
from .session import NativeSenseFlowTrainingSession


class LinearWarmupConstantScheduler:
    """Stateful form of SenseFlow's released linear-warmup/constant schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        start_ratio: float,
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        if (
            isinstance(warmup_steps, bool)
            or not isinstance(warmup_steps, int)
            or warmup_steps < 0
        ):
            raise ValueError("warmup_steps must be a non-negative integer")
        ratio = float(start_ratio)
        if not isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("start_ratio must lie in (0,1]")
        base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        if not base_lrs or any(not isfinite(value) or value <= 0 for value in base_lrs):
            raise ValueError("optimizer groups must have finite positive learning rates")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_ratio = ratio
        self.base_lrs = base_lrs
        self.step_count = 0
        self._apply()

    def _ratio(self) -> float:
        if self.warmup_steps == 0:
            return 1.0
        progress = min(float(self.step_count) / float(self.warmup_steps), 1.0)
        return self.start_ratio + (1.0 - self.start_ratio) * progress

    def _apply(self) -> None:
        ratio = self._ratio()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * ratio

    def step(self) -> None:
        self.step_count += 1
        self._apply()

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "worldfoundry-senseflow-linear-warmup-constant-scheduler",
            "step_count": self.step_count,
            "warmup_steps": self.warmup_steps,
            "start_ratio": self.start_ratio,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        expected = {
            "schema",
            "step_count",
            "warmup_steps",
            "start_ratio",
            "base_lrs",
        }
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected:
            raise ValueError("SenseFlow scheduler state fields differ from the active schema")
        if state_dict["schema"] != "worldfoundry-senseflow-linear-warmup-constant-scheduler":
            raise ValueError("unsupported SenseFlow scheduler state schema")
        if int(state_dict["warmup_steps"]) != self.warmup_steps:
            raise ValueError("saved SenseFlow warmup length differs from the active scheduler")
        if float(state_dict["start_ratio"]) != self.start_ratio:
            raise ValueError("saved SenseFlow warmup start differs from the active scheduler")
        saved_base_lrs = tuple(float(value) for value in state_dict["base_lrs"])
        if saved_base_lrs != self.base_lrs:
            raise ValueError("saved SenseFlow base learning rates differ from the active scheduler")
        step_count = state_dict["step_count"]
        if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 0:
            raise ValueError("saved SenseFlow scheduler step must be a non-negative integer")
        self.step_count = step_count
        self._apply()


def _module(adapter: object, protocol: type, *, role: str) -> nn.Module:
    if not isinstance(adapter, protocol):
        raise TypeError(f"{role} does not implement its SenseFlow functional adapter")
    module = getattr(adapter, "module", None)
    if not isinstance(module, nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    return module


def _configure_discriminator_roles(
    adapter: SenseFlowDiscriminatorAdapter,
    discriminator_module: nn.Module,
) -> tuple[nn.Module, ...]:
    frozen_values = getattr(adapter, "frozen_feature_modules", None)
    head_values = getattr(adapter, "trainable_head_modules", None)
    if not isinstance(frozen_values, tuple) or not frozen_values:
        raise ValueError(
            "SenseFlow discriminator must expose frozen VFM/language feature modules"
        )
    if not isinstance(head_values, tuple) or not head_values:
        raise ValueError("SenseFlow discriminator must expose trainable head modules")
    descendants = {id(module) for module in discriminator_module.modules()}
    modules: list[nn.Module] = []
    for value in frozen_values:
        if not isinstance(value, nn.Module) or id(value) not in descendants:
            raise ValueError(
                "SenseFlow discriminator feature modules must belong to discriminator.module"
            )
        value.requires_grad_(False)
        value.eval()
        modules.append(value)
    if len({id(module) for module in modules}) != len(modules):
        raise ValueError("SenseFlow discriminator feature modules cannot contain duplicates")
    heads: list[nn.Module] = []
    for value in head_values:
        if not isinstance(value, nn.Module) or id(value) not in descendants:
            raise ValueError(
                "SenseFlow discriminator head modules must belong to discriminator.module"
            )
        heads.append(value)
    if len({id(module) for module in heads}) != len(heads):
        raise ValueError("SenseFlow discriminator head modules cannot contain duplicates")
    if {id(module) for module in modules} & {id(module) for module in heads}:
        raise ValueError("SenseFlow discriminator feature and head modules must be disjoint")
    actual_trainable = {
        id(parameter)
        for parameter in discriminator_module.parameters()
        if parameter.requires_grad
    }
    declared_trainable = {
        id(parameter)
        for head in heads
        for parameter in head.parameters()
        if parameter.requires_grad
    }
    if not declared_trainable or declared_trainable != actual_trainable:
        raise ValueError(
            "SenseFlow discriminator trainable parameters must belong exactly to its heads"
        )
    return tuple(modules)


@dataclass(frozen=True, slots=True)
class NativeSenseFlowTrainingStack:
    recipe: PostTrainingRecipe
    config: SenseFlowConfig
    optimizer_config: SenseFlowOptimizerConfig
    model: nn.ModuleDict
    loss_adapter: NativeSenseFlowLossAdapter
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer
    discriminator_optimizer: torch.optim.Optimizer
    student_scheduler: LinearWarmupConstantScheduler
    fake_score_scheduler: LinearWarmupConstantScheduler
    discriminator_scheduler: LinearWarmupConstantScheduler
    scheduler_state: NamedStatefulCollection
    engine: NativeSenseFlowTrainEngine

    @property
    def optimizers(self) -> tuple[torch.optim.Optimizer, ...]:
        return (
            self.student_optimizer,
            self.fake_score_optimizer,
            self.discriminator_optimizer,
        )

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": None,
            "algorithm_state": None,
        }

    def create_session(
        self,
        dataloader: Iterable[SenseFlowTrainingBatch],
        progress: TrainingProgress,
        **kwargs: object,
    ) -> NativeSenseFlowTrainingSession:
        return NativeSenseFlowTrainingSession(
            self.engine,
            dataloader,
            progress,
            **kwargs,
        )


def build_native_senseflow_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: SenseFlowPredictionAdapter,
    teacher: SenseFlowTeacherAdapter,
    fake_score: SenseFlowFakeScoreAdapter,
    discriminator: SenseFlowDiscriminatorAdapter,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeSenseFlowTrainingStack:
    """Build all four roles, three optimizers, schedulers, and the native engine."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, SenseFlowAlgorithmSpec):
        raise TypeError("SenseFlow stack requires SenseFlowAlgorithmSpec")
    if recipe.fake_score_optimizer is None or recipe.discriminator_optimizer is None:
        raise ValueError("SenseFlow requires fake_score_optimizer and discriminator_optimizer")
    if recipe.guidance_optimizer is not None:
        raise ValueError("SenseFlow does not accept guidance_optimizer")
    config = SenseFlowConfig.from_recipe(recipe.algorithm)
    optimizer_config = SenseFlowOptimizerConfig.from_recipe(
        recipe.algorithm,
        recipe.optimizer,
        recipe.fake_score_optimizer,
        recipe.discriminator_optimizer,
    )
    student_module = _module(student, SenseFlowPredictionAdapter, role="SenseFlow student")
    teacher_module = _module(teacher, SenseFlowTeacherAdapter, role="SenseFlow teacher")
    fake_score_module = _module(
        fake_score,
        SenseFlowFakeScoreAdapter,
        role="SenseFlow fake score",
    )
    discriminator_module = _module(
        discriminator,
        SenseFlowDiscriminatorAdapter,
        role="SenseFlow discriminator",
    )
    roles = {
        "student": student_module,
        "teacher": teacher_module,
        "fake-score": fake_score_module,
        "discriminator": discriminator_module,
    }
    require_independent_modules(roles)
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="SenseFlow student",
    )
    require_checkpoint_identity(
        teacher,
        recipe.algorithm.teacher_checkpoint,
        role="SenseFlow teacher",
    )
    require_checkpoint_identity(
        fake_score,
        recipe.algorithm.fake_score_checkpoint,
        role="SenseFlow fake score",
    )
    require_checkpoint_identity(
        discriminator,
        recipe.algorithm.discriminator_checkpoint,
        role="SenseFlow discriminator",
    )
    teacher_module.requires_grad_(False)
    teacher_module.eval()
    discriminator_frozen_modules = _configure_discriminator_roles(
        discriminator,
        discriminator_module,
    )
    require_disjoint_trainable_parameters(
        {
            "student": student_module,
            "fake-score": fake_score_module,
            "discriminator": discriminator_module,
        }
    )

    student_optimizer = build_adamw(
        trainable_parameters(student_module),
        learning_rate=optimizer_config.student_learning_rate,
        weight_decay=optimizer_config.weight_decay,
        betas=optimizer_config.betas,
        epsilon=optimizer_config.epsilon,
        fused=fused_adamw,
    )
    fake_score_optimizer = build_adamw(
        trainable_parameters(fake_score_module),
        learning_rate=optimizer_config.fake_score_learning_rate,
        weight_decay=optimizer_config.weight_decay,
        betas=optimizer_config.betas,
        epsilon=optimizer_config.epsilon,
        fused=fused_adamw,
    )
    discriminator_optimizer = build_adamw(
        trainable_parameters(discriminator_module),
        learning_rate=optimizer_config.discriminator_learning_rate,
        weight_decay=optimizer_config.weight_decay,
        betas=optimizer_config.betas,
        epsilon=optimizer_config.epsilon,
        fused=fused_adamw,
    )
    student_scheduler = LinearWarmupConstantScheduler(
        student_optimizer,
        warmup_steps=optimizer_config.warmup_steps,
        start_ratio=optimizer_config.warmup_start_ratio,
    )
    fake_score_scheduler = LinearWarmupConstantScheduler(
        fake_score_optimizer,
        warmup_steps=optimizer_config.warmup_steps,
        start_ratio=optimizer_config.warmup_start_ratio,
    )
    discriminator_scheduler = LinearWarmupConstantScheduler(
        discriminator_optimizer,
        warmup_steps=optimizer_config.warmup_steps,
        start_ratio=optimizer_config.warmup_start_ratio,
    )
    loss_adapter = NativeSenseFlowLossAdapter(
        student,
        teacher,
        fake_score,
        discriminator,
        config,
    )
    execution_config_digest = canonical_sha256(
        {
            "schema": "worldfoundry-senseflow-training-config",
            "objective_digest": loss_adapter.config_digest,
            "optimizer_digest": optimizer_config.digest,
            "recipe_digest": recipe.digest,
        }
    )
    engine = NativeSenseFlowTrainEngine(
        student_module=student_module,
        teacher_module=teacher_module,
        fake_score_module=fake_score_module,
        discriminator_module=discriminator_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        student_max_grad_norm=optimizer_config.student_max_grad_norm,
        fake_score_max_grad_norm=optimizer_config.fake_score_max_grad_norm,
        discriminator_max_grad_norm=optimizer_config.discriminator_max_grad_norm,
        gradient_accumulation_steps=optimizer_config.gradient_accumulation_steps,
        seed=config.seed,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        student_scheduler_cadence=config.student_scheduler_cadence,
        execution_config_digest=execution_config_digest,
        discriminator_frozen_modules=discriminator_frozen_modules,
        parallel_context=parallel_context,
    )
    scheduler_state = NamedStatefulCollection(
        {
            "student": student_scheduler,
            "fake-score": fake_score_scheduler,
            "discriminator": discriminator_scheduler,
        }
    )
    return NativeSenseFlowTrainingStack(
        recipe=recipe,
        config=config,
        optimizer_config=optimizer_config,
        model=nn.ModuleDict(
            {
                "student": student_module,
                "teacher": teacher_module,
                "fake_score": fake_score_module,
                "discriminator": discriminator_module,
            }
        ),
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        scheduler_state=scheduler_state,
        engine=engine,
    )


__all__ = [
    "LinearWarmupConstantScheduler",
    "NativeSenseFlowTrainingStack",
    "build_native_senseflow_training_stack",
]
