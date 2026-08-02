"""Construction of native Self-Gradient-Forcing DMD training."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.self_gradient_forcing import (
    SelfGradientForcingAlgorithmSpec,
)
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
from ...shared.ema import DelayedModuleEMA
from ..dmd.engine import NativeDMDTrainEngine
from ..dmd.objective import DMDConfig, FlowDMDLossAdapter
from .config import SelfGradientForcingConfig
from .contracts import SelfGradientForcingAdapter
from .engine import NativeSelfGradientForcingTrainEngine
from .rollout import SelfGradientForcingSampler


@dataclass(frozen=True, slots=True)
class NativeSelfGradientForcingTrainingStack:
    """Two-pass causal sampler, DMD objective, optimizers, and exact state."""

    recipe: PostTrainingRecipe
    rollout_config: SelfGradientForcingConfig
    dmd_config: DMDConfig
    sampler: SelfGradientForcingSampler
    loss_adapter: FlowDMDLossAdapter
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer
    engine: NativeSelfGradientForcingTrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }


def build_native_self_gradient_forcing_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: SelfGradientForcingAdapter,
    real_score: FlowPredictionAdapter,
    fake_score: FlowPredictionAdapter,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeSelfGradientForcingTrainingStack:
    """Build official two-pass replay over WorldFoundry-native model roles."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, SelfGradientForcingAlgorithmSpec):
        raise TypeError(
            "Self-Gradient-Forcing stack requires its matching algorithm spec"
        )
    if recipe.fake_score_optimizer is None:
        raise ValueError("Self-Gradient-Forcing requires fake_score_optimizer")
    if not isinstance(student, SelfGradientForcingAdapter):
        raise TypeError("student must implement SelfGradientForcingAdapter")
    student_module = student.module
    if not isinstance(student_module, torch.nn.Module):
        raise TypeError("Self-Gradient-Forcing student.module must be torch.nn.Module")
    real_score_module = prediction_module(
        real_score,
        role="Self-Gradient-Forcing real-score teacher",
    )
    fake_score_module = prediction_module(
        fake_score,
        role="Self-Gradient-Forcing fake-score critic",
    )
    algorithm = recipe.algorithm
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="Self-Gradient-Forcing student",
    )
    require_checkpoint_identity(
        real_score,
        algorithm.real_score_checkpoint,
        role="Self-Gradient-Forcing real-score teacher",
    )
    require_checkpoint_identity(
        fake_score,
        algorithm.fake_score_checkpoint,
        role="Self-Gradient-Forcing fake-score critic",
    )
    require_independent_modules(
        {
            "student": student_module,
            "real-score": real_score_module,
            "fake-score": fake_score_module,
        }
    )
    if any(parameter.requires_grad for parameter in real_score_module.parameters()):
        raise ValueError("Self-Gradient-Forcing real-score teacher must be frozen")
    real_score_module.eval()
    accumulation_steps = recipe.optimizer.gradient_accumulation_steps
    if recipe.fake_score_optimizer.gradient_accumulation_steps != accumulation_steps:
        raise ValueError(
            "Self-Gradient-Forcing student and fake-score accumulation steps must match"
        )

    rollout_config = SelfGradientForcingConfig.from_raw_timesteps(
        algorithm.denoising_timesteps,
        num_train_timesteps=algorithm.num_train_timesteps,
        flow_shift=algorithm.denoising_flow_shift,
        frames_per_block=algorithm.frames_per_block,
        frame_dim=algorithm.frame_dim,
        context_timestep=algorithm.context_timestep,
        cache_target_mode=algorithm.cache_target_mode,
        exit_step_rank_mode=algorithm.exit_step_rank_mode,
        match_context=algorithm.match_context,
        last_step_only=algorithm.last_step_only,
    )
    context = parallel_context or PostTrainingParallelContext.current()
    sampler = SelfGradientForcingSampler(
        student,
        rollout_config,
        parallel_context=context,
    )
    dmd_config = DMDConfig(
        schedule=rollout_config.schedule,
        num_train_timesteps=algorithm.num_train_timesteps,
        score_min_sigma=algorithm.score_min_sigma,
        score_max_sigma=algorithm.score_max_sigma,
        score_flow_shift=algorithm.score_flow_shift,
        teacher_guidance_scale=algorithm.teacher_guidance_scale,
        normalization_epsilon=algorithm.normalization_epsilon,
        shared_score_timestep=False,
        per_sample_normalization=True,
    )
    loss_adapter = FlowDMDLossAdapter(
        None,
        real_score,
        fake_score,
        dmd_config,
        student_sampler=sampler,
    )
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="Self-Gradient-Forcing student",
    )
    fake_score_optimizer = build_post_training_optimizer(
        replace(recipe.fake_score_optimizer, gradient_accumulation_steps=1),
        fake_score_module,
        fused=fused_adamw,
        role="Self-Gradient-Forcing fake-score critic",
    )
    student_ema = DelayedModuleEMA(student_module, decay=algorithm.ema_decay)
    dmd_engine = NativeDMDTrainEngine(
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
        student_ema_start_step=algorithm.ema_start_step,
        parallel_context=context,
    )
    engine = NativeSelfGradientForcingTrainEngine(dmd_engine, sampler)
    ema_state = named_stateful_collection(student=student_ema)
    if ema_state is None:
        raise RuntimeError("Self-Gradient-Forcing student EMA was not registered")
    return NativeSelfGradientForcingTrainingStack(
        recipe=recipe,
        rollout_config=rollout_config,
        dmd_config=dmd_config,
        sampler=sampler,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        engine=engine,
        scheduler_state=named_stateful_collection(
            student=student_scheduler,
            fake_score=fake_score_scheduler,
        ),
        ema_state=ema_state,
    )


__all__ = [
    "NativeSelfGradientForcingTrainingStack",
    "build_native_self_gradient_forcing_training_stack",
]
