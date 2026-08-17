"""Fail-closed construction of the native SGMD stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.sgmd import (
    SGMDAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ...shared.building import (
    build_post_training_optimizer,
    named_stateful_collection,
    validate_post_training_recipe,
)
from ...shared.distributed import PostTrainingParallelContext
from .config import SGMDConfig
from .contracts import SGMDPredictionAdapter
from .engine import NativeSGMDTrainEngine
from .objective import NativeSGMDLossAdapter


def _module(adapter: SGMDPredictionAdapter, *, role: str) -> torch.nn.Module:
    if not isinstance(adapter, SGMDPredictionAdapter):
        raise TypeError(f"{role} must implement SGMDPredictionAdapter")
    if not isinstance(adapter.module, torch.nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    if not str(adapter.checkpoint_identity).strip():
        raise ValueError(f"{role}.checkpoint_identity must be non-empty")
    return adapter.module


def _require_checkpoint(
    adapter: SGMDPredictionAdapter,
    expected: str,
    *,
    role: str,
) -> None:
    actual = str(adapter.checkpoint_identity).strip()
    if actual != str(expected).strip():
        raise ValueError(
            f"{role} loaded checkpoint identity {actual!r} differs from recipe {expected!r}"
        )


def _audit_flow_process(adapters: tuple[SGMDPredictionAdapter, ...]) -> None:
    kinds = {
        str(adapter.noise_process_kind).strip().lower().replace("_", "-")
        for adapter in adapters
    }
    if kinds != {"flow-matching"}:
        raise ValueError("SGMD requires every role to expose the same flow-matching process")


def _audit_independent_roles(modules: tuple[torch.nn.Module, ...]) -> None:
    inventories = [
        {id(parameter) for parameter in module.parameters()} for module in modules
    ]
    for left in range(len(inventories)):
        for right in range(left + 1, len(inventories)):
            if inventories[left] & inventories[right]:
                raise ValueError(
                    "SGMD roles must be independently materialized without shared parameters"
                )


@dataclass(frozen=True, slots=True)
class NativeSGMDTrainingStack:
    recipe: PostTrainingRecipe
    config: SGMDConfig
    loss_adapter: NativeSGMDLossAdapter
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer
    engine: NativeSGMDTrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }


def build_native_sgmd_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: SGMDPredictionAdapter,
    teacher: SGMDPredictionAdapter,
    fake_score: SGMDPredictionAdapter,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    student_ema: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeSGMDTrainingStack:
    """Build SGMD roles, objectives, optimizers, and cadence from one recipe."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, SGMDAlgorithmSpec):
        raise TypeError("SGMD stack requires SGMDAlgorithmSpec")
    if recipe.fake_score_optimizer is None:
        raise ValueError("SGMD requires fake_score_optimizer")
    accumulation = recipe.optimizer.gradient_accumulation_steps
    if recipe.fake_score_optimizer.gradient_accumulation_steps != accumulation:
        raise ValueError("SGMD student and fake-score accumulation steps must match")
    adapters = (student, teacher, fake_score)
    modules = tuple(
        _module(adapter, role=role)
        for adapter, role in zip(
            adapters,
            ("SGMD student", "SGMD teacher", "SGMD fake score"),
            strict=True,
        )
    )
    student_module, teacher_module, fake_score_module = modules
    if len({id(module) for module in modules}) != 3:
        raise ValueError("SGMD roles must be independently materialized as distinct modules")
    _audit_independent_roles(modules)
    _audit_flow_process(adapters)
    _require_checkpoint(student, recipe.model.checkpoint, role="SGMD student")
    _require_checkpoint(
        teacher,
        recipe.algorithm.teacher_checkpoint,
        role="SGMD teacher",
    )
    _require_checkpoint(
        fake_score,
        recipe.algorithm.fake_score_checkpoint,
        role="SGMD fake score",
    )
    if any(parameter.requires_grad for parameter in teacher_module.parameters()):
        raise ValueError("SGMD teacher parameters must be frozen before stack construction")
    teacher_module.eval()
    fake_score_module.eval()
    config = SGMDConfig.from_recipe(recipe.algorithm)
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="SGMD student",
    )
    fake_score_optimizer = build_post_training_optimizer(
        replace(recipe.fake_score_optimizer, gradient_accumulation_steps=1),
        fake_score_module,
        fused=fused_adamw,
        role="SGMD fake score",
    )
    loss_adapter = NativeSGMDLossAdapter(
        student,
        teacher,
        fake_score,
        config,
    )
    engine = NativeSGMDTrainEngine(
        student_module=student_module,
        teacher_module=teacher_module,
        fake_score_module=fake_score_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        student_max_grad_norm=recipe.optimizer.max_grad_norm,
        fake_score_max_grad_norm=recipe.fake_score_optimizer.max_grad_norm,
        gradient_accumulation_steps=accumulation,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        student_ema=student_ema,
        parallel_context=parallel_context,
    )
    return NativeSGMDTrainingStack(
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


__all__ = ["NativeSGMDTrainingStack", "build_native_sgmd_training_stack"]
