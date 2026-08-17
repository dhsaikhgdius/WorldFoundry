"""Construction of the recipe-owned native Reward-Forcing stack."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from math import isclose
from typing import Literal

import torch
from torch import nn

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.reward_forcing import (
    RewardForcingAlgorithmSpec,
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
from ..self_forcing.ema import DelayedSelfForcingEMA
from ..self_forcing.rollout import SelfForcingRolloutSampler
from .config import RewardForcingConfig
from .contracts import (
    MotionQualityRewardAdapter,
    RewardForcingCausalAdapter,
    RewardForcingDecoderAdapter,
    RewardForcingTrainingBatch,
)
from .engine import NativeRewardForcingTrainEngine
from .objective import NativeRewardForcingLossAdapter
from .session import NativeRewardForcingTrainingSession


@dataclass(frozen=True, slots=True)
class NativeRewardForcingTrainingStack:
    """Recipe-bound roles, optimizers, engine, session, and DCP components."""

    recipe: PostTrainingRecipe
    config: RewardForcingConfig
    sampler: SelfForcingRolloutSampler
    loss_adapter: NativeRewardForcingLossAdapter
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer
    engine: NativeRewardForcingTrainEngine
    scheduler_state: NamedStatefulCollection | None
    ema_state: NamedStatefulCollection

    @property
    def optimizers(self) -> tuple[torch.optim.Optimizer, ...]:
        return (self.student_optimizer, self.fake_score_optimizer)

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema_state,
            "algorithm_state": None,
        }

    def build_session(
        self,
        dataloader: Iterable[RewardForcingTrainingBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> NativeRewardForcingTrainingSession:
        """Build the sole session path from this recipe-owned stack."""

        return NativeRewardForcingTrainingSession(
            self.engine,
            dataloader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=save_every_steps,
            asynchronous_checkpoints=asynchronous_checkpoints,
            event_sink=event_sink,
        )


def _motion_reward_module(
    motion_reward: MotionQualityRewardAdapter,
) -> nn.Module | None:
    module = motion_reward.owned_module
    if module is not None and not isinstance(module, nn.Module):
        raise TypeError("Reward-Forcing motion reward owned_module must be an nn.Module or None")
    return module


def _require_frozen(module: nn.Module, *, role: str) -> None:
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError(f"{role} must be frozen before stack construction")
    module.eval()


def _audit_motion_reward_behavior(
    motion_reward: MotionQualityRewardAdapter,
    algorithm: RewardForcingAlgorithmSpec,
) -> None:
    expected = {
        "calibration_mean": algorithm.motion_reward_calibration_mean,
        "calibration_std": algorithm.motion_reward_calibration_std,
        "normalization_epsilon": algorithm.motion_reward_normalization_epsilon,
    }
    for name, value in expected.items():
        try:
            actual = float(getattr(motion_reward, name))
        except (TypeError, ValueError) as error:
            raise TypeError(f"Reward-Forcing motion reward {name} must be numeric") from error
        if not isclose(actual, value, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"Reward-Forcing motion reward {name} differs from the active recipe")


def build_native_reward_forcing_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: RewardForcingCausalAdapter,
    real_score: FlowPredictionAdapter,
    fake_score: FlowPredictionAdapter,
    reward_decoder: RewardForcingDecoderAdapter,
    motion_reward: MotionQualityRewardAdapter,
    student_scheduler: object | None = None,
    fake_score_scheduler: object | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeRewardForcingTrainingStack:
    """Build released Re-DMD behavior from one strict PostTrainingRecipe."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, RewardForcingAlgorithmSpec):
        raise TypeError("Reward-Forcing stack requires a RewardForcingAlgorithmSpec recipe")
    if recipe.fake_score_optimizer is None:
        raise ValueError("Reward-Forcing requires fake_score_optimizer")
    if recipe.guidance_optimizer is not None or recipe.discriminator_optimizer is not None:
        raise ValueError("Reward-Forcing accepts only the primary and fake_score optimizers")
    accumulation_steps = recipe.optimizer.gradient_accumulation_steps
    if recipe.fake_score_optimizer.gradient_accumulation_steps != accumulation_steps:
        raise ValueError("Reward-Forcing student and fake-score accumulation steps must match")
    if not isinstance(student, RewardForcingCausalAdapter):
        raise TypeError("student must implement RewardForcingCausalAdapter")
    if not isinstance(student.module, nn.Module):
        raise TypeError("student.module must be an nn.Module")
    if not isinstance(reward_decoder, RewardForcingDecoderAdapter):
        raise TypeError("reward_decoder must implement RewardForcingDecoderAdapter")
    if not isinstance(reward_decoder.module, nn.Module):
        raise TypeError("reward_decoder.module must be an nn.Module")
    if not isinstance(motion_reward, MotionQualityRewardAdapter):
        raise TypeError("motion_reward must implement MotionQualityRewardAdapter")

    algorithm = recipe.algorithm
    config = RewardForcingConfig.from_recipe(algorithm)
    student.audit_reward_forcing_cache(
        frames_per_block=config.frames_per_block,
        local_attention_frames=config.local_attention_frames,
        ema_sink_frames=config.ema_sink_frames,
        ema_sink_decay=config.ema_sink_decay,
    )
    _audit_motion_reward_behavior(motion_reward, algorithm)

    student_module = student.module
    real_score_module = prediction_module(
        real_score,
        role="Reward-Forcing real-score teacher",
    )
    fake_score_module = prediction_module(
        fake_score,
        role="Reward-Forcing fake-score critic",
    )
    decoder_module = reward_decoder.module
    motion_module = _motion_reward_module(motion_reward)

    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="Reward-Forcing student",
    )
    require_checkpoint_identity(
        real_score,
        algorithm.real_score_checkpoint,
        role="Reward-Forcing real-score teacher",
    )
    require_checkpoint_identity(
        fake_score,
        algorithm.fake_score_checkpoint,
        role="Reward-Forcing fake-score critic",
    )
    require_checkpoint_identity(
        reward_decoder,
        algorithm.reward_decoder_checkpoint,
        role="Reward-Forcing reward decoder",
    )
    require_checkpoint_identity(
        motion_reward,
        algorithm.motion_reward_checkpoint,
        role="Reward-Forcing motion reward",
    )

    roles = {
        "student": student_module,
        "real-score": real_score_module,
        "fake-score": fake_score_module,
        "reward-decoder": decoder_module,
    }
    if motion_module is not None:
        roles["motion-reward"] = motion_module
    require_independent_modules(roles)
    _require_frozen(
        real_score_module,
        role="Reward-Forcing real-score teacher",
    )
    _require_frozen(
        decoder_module,
        role="Reward-Forcing reward decoder",
    )
    if motion_module is not None:
        _require_frozen(
            motion_module,
            role="Reward-Forcing motion reward",
        )
    student_module.eval()
    fake_score_module.eval()

    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        student_module,
        fused=fused_adamw,
        role="Reward-Forcing student",
    )
    fake_score_optimizer = build_post_training_optimizer(
        replace(recipe.fake_score_optimizer, gradient_accumulation_steps=1),
        fake_score_module,
        fused=fused_adamw,
        role="Reward-Forcing fake-score critic",
    )
    context = parallel_context or PostTrainingParallelContext.current()
    sampler = SelfForcingRolloutSampler(
        student,
        config.rollout_config,
        parallel_context=context,
    )
    loss_adapter = NativeRewardForcingLossAdapter(
        real_score,
        fake_score,
        sampler,
        reward_decoder,
        motion_reward,
        config,
    )
    student_ema = DelayedSelfForcingEMA(
        student_module,
        decay=config.ema_decay,
    )
    engine = NativeRewardForcingTrainEngine(
        student_module=student_module,
        real_score_module=real_score_module,
        fake_score_module=fake_score_module,
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        generator_update_interval=config.generator_update_interval,
        student_max_grad_norm=recipe.optimizer.max_grad_norm,
        fake_score_max_grad_norm=recipe.fake_score_optimizer.max_grad_norm,
        gradient_accumulation_steps=accumulation_steps,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        student_scheduler_cadence=config.student_scheduler_cadence,
        student_ema=student_ema,
        student_ema_start_step=config.ema_start_step,
        parallel_context=context,
    )
    ema_state = NamedStatefulCollection({"student": student_ema})
    return NativeRewardForcingTrainingStack(
        recipe=recipe,
        config=config,
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
    "NativeRewardForcingTrainingStack",
    "build_native_reward_forcing_training_stack",
]
