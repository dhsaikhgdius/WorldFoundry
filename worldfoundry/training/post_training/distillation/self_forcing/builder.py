"""Construction of the WorldFoundry-native Self-Forcing DMD stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.self_forcing import (
    SelfForcingAlgorithmSpec,
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
from ..dmd.engine import NativeDMDTrainEngine
from ..dmd.objective import DMDConfig, FlowDMDLossAdapter
from .config import SelfForcingConfig, shifted_few_step_schedule
from .contracts import CausalChunkAdapter
from .ema import DelayedSelfForcingEMA
from .rollout import SelfForcingRolloutSampler


@dataclass(frozen=True, slots=True)
class NativeSelfForcingTrainingStack:
    """Causal rollout, holistic DMD math, and two-optimizer execution."""

    recipe: PostTrainingRecipe
    rollout_config: SelfForcingConfig
    dmd_config: DMDConfig
    sampler: SelfForcingRolloutSampler
    loss_adapter: FlowDMDLossAdapter
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer
    engine: NativeDMDTrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }


def build_native_self_forcing_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: CausalChunkAdapter,
    real_score: FlowPredictionAdapter,
    fake_score: FlowPredictionAdapter,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeSelfForcingTrainingStack:
    """Build official-semantics Self-Forcing over native model roles."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, SelfForcingAlgorithmSpec):
        raise TypeError("Self-Forcing stack requires a SelfForcingAlgorithmSpec recipe")
    if recipe.algorithm.distribution_objective != "dmd":
        raise ValueError("Self-Forcing builder only executes the DMD objective")
    if recipe.fake_score_optimizer is None:
        raise ValueError("Self-Forcing DMD requires fake_score_optimizer")
    if not isinstance(student, CausalChunkAdapter):
        raise TypeError("student must implement CausalChunkAdapter")
    student_module = student.module
    if not isinstance(student_module, torch.nn.Module):
        raise TypeError("Self-Forcing student.module must be torch.nn.Module")
    real_score_module = prediction_module(real_score, role="Self-Forcing real-score teacher")
    fake_score_module = prediction_module(fake_score, role="Self-Forcing fake-score critic")
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="Self-Forcing student",
    )
    require_checkpoint_identity(
        real_score,
        recipe.algorithm.real_score_checkpoint,
        role="Self-Forcing real-score teacher",
    )
    require_checkpoint_identity(
        fake_score,
        recipe.algorithm.fake_score_checkpoint,
        role="Self-Forcing fake-score critic",
    )
    require_independent_modules(
        {
            "student": student_module,
            "real-score": real_score_module,
            "fake-score": fake_score_module,
        }
    )
    if any(parameter.requires_grad for parameter in real_score_module.parameters()):
        raise ValueError(
            "Self-Forcing real-score teacher must be frozen before stack construction"
        )
    real_score_module.eval()
    accumulation_steps = recipe.optimizer.gradient_accumulation_steps
    if recipe.fake_score_optimizer.gradient_accumulation_steps != accumulation_steps:
        raise ValueError("Self-Forcing student and fake-score accumulation steps must match")

    algorithm = recipe.algorithm
    schedule = shifted_few_step_schedule(
        algorithm.denoising_timesteps,
        num_train_timesteps=algorithm.num_train_timesteps,
        flow_shift=algorithm.denoising_flow_shift,
    )
    rollout_config = SelfForcingConfig(
        schedule=schedule,
        frames_per_block=algorithm.frames_per_block,
        frame_dim=algorithm.frame_dim,
        exit_step_mode=algorithm.exit_step_mode,
    )
    context = parallel_context or PostTrainingParallelContext.current()
    sampler = SelfForcingRolloutSampler(
        student,
        rollout_config,
        parallel_context=context,
    )
    dmd_config = DMDConfig(
        schedule=schedule,
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
        role="Self-Forcing student",
    )
    fake_score_optimizer = build_post_training_optimizer(
        replace(recipe.fake_score_optimizer, gradient_accumulation_steps=1),
        fake_score_module,
        fused=fused_adamw,
        role="Self-Forcing fake-score critic",
    )
    student_ema = DelayedSelfForcingEMA(student_module, decay=algorithm.ema_decay)
    engine = NativeDMDTrainEngine(
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
    return NativeSelfForcingTrainingStack(
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
        ema_state=named_stateful_collection(student=student_ema),
    )


__all__ = [
    "NativeSelfForcingTrainingStack",
    "build_native_self_forcing_training_stack",
]
