"""Recipe-owned construction of native AnyFlow training stacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import nn

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.recipes.post_training.algorithms.anyflow import (
    AnyFlowBidirectionalOnPolicyAlgorithmSpec,
    AnyFlowBidirectionalPretrainAlgorithmSpec,
    AnyFlowFAROnPolicyAlgorithmSpec,
    AnyFlowFARPretrainAlgorithmSpec,
    AnyFlowFARSpec,
    AnyFlowMapSpec,
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
from .bidirectional_on_policy import NativeAnyFlowBidirectionalOnPolicyLossAdapter
from .bidirectional_pretrain import NativeAnyFlowBidirectionalPretrainLossAdapter
from .config import (
    AnyFlowBidirectionalOnPolicyConfig,
    AnyFlowBidirectionalPretrainConfig,
    AnyFlowFARConfig,
    AnyFlowMapConfig,
    AnyFlowOnPolicyConfig,
    AnyFlowPretrainConfig,
)
from .contracts import (
    AnyFlowBidirectionalAdapter,
    AnyFlowFARAdapter,
    AnyFlowOnPolicyLossAdapter,
    AnyFlowPretrainLossAdapter,
    AnyFlowScoreAdapter,
)
from .ema import AnyFlowEMA
from .engine import NativeAnyFlowOnPolicyEngine, NativeAnyFlowPretrainEngine
from .on_policy import NativeAnyFlowOnPolicyLossAdapter
from .pretrain import NativeAnyFlowPretrainLossAdapter
from .synchronization import (
    AnyFlowDecisionRNG,
    ProcessGroupAnyFlowTensorSynchronizer,
)


def _module(adapter: object, *, role: str) -> nn.Module:
    module = getattr(adapter, "module", None)
    if not isinstance(module, nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    return module


def _map_config(spec: AnyFlowMapSpec) -> AnyFlowMapConfig:
    return AnyFlowMapConfig(
        num_train_timesteps=spec.num_train_timesteps,
        timestep_shift=spec.timestep_shift,
        central_difference_epsilon=spec.central_difference_epsilon,
        diffusion_ratio=spec.diffusion_ratio,
        consistency_ratio=spec.consistency_ratio,
        fused_guidance_scale=spec.fused_guidance_scale,
    )


def _far_config(spec: AnyFlowFARSpec) -> AnyFlowFARConfig:
    return AnyFlowFARConfig(
        chunk_partition=spec.chunk_partition,
        full_chunk_limit=spec.full_chunk_limit,
        patch_size=spec.patch_size,
        compressed_patch_size=spec.compressed_patch_size,
        long_context_training_ratio=spec.long_context_training_ratio,
    )


def _parallel_and_decisions(
    parallel_context: PostTrainingParallelContext | None,
    *,
    seed: int,
) -> tuple[PostTrainingParallelContext, AnyFlowDecisionRNG]:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("AnyFlow synchronized decision seed must be non-negative")
    context = parallel_context or PostTrainingParallelContext.current()
    synchronizer = None if context.world_size == 1 else ProcessGroupAnyFlowTensorSynchronizer(context.process_group)
    return context, AnyFlowDecisionRNG(int(seed), synchronizer=synchronizer)


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
    value = factory(optimizer)
    if not callable(getattr(value, "state_dict", None)) or not callable(getattr(value, "load_state_dict", None)):
        raise TypeError(f"{role} scheduler must expose state_dict/load_state_dict")
    return value


@dataclass(frozen=True, slots=True)
class NativeAnyFlowPretrainingStack:
    recipe: PostTrainingRecipe
    config: AnyFlowPretrainConfig | AnyFlowBidirectionalPretrainConfig
    loss_adapter: AnyFlowPretrainLossAdapter
    decisions: AnyFlowDecisionRNG
    optimizer: torch.optim.Optimizer
    scheduler: object | None
    engine: NativeAnyFlowPretrainEngine
    model: nn.ModuleDict
    scheduler_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": None,
            "algorithm_state": None,
        }


@dataclass(frozen=True, slots=True)
class NativeAnyFlowOnPolicyTrainingStack:
    recipe: PostTrainingRecipe
    config: AnyFlowOnPolicyConfig | AnyFlowBidirectionalOnPolicyConfig
    loss_adapter: AnyFlowOnPolicyLossAdapter
    decisions: AnyFlowDecisionRNG
    student_optimizer: torch.optim.Optimizer
    fake_score_optimizer: torch.optim.Optimizer
    student_scheduler: object | None
    fake_score_scheduler: object | None
    ema: AnyFlowEMA
    engine: NativeAnyFlowOnPolicyEngine
    model: nn.ModuleDict
    scheduler_state: NamedStatefulCollection | None

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": self.scheduler_state,
            "ema": self.ema,
            "algorithm_state": None,
        }


def build_native_anyflow_pretraining_stack(
    recipe: PostTrainingRecipe,
    *,
    student: AnyFlowFARAdapter | AnyFlowBidirectionalAdapter,
    scheduler_factory: Callable[[torch.optim.Optimizer], object] | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeAnyFlowPretrainingStack:
    """Build FAR or bidirectional FlowMap pretraining without an upstream loop."""

    validate_post_training_recipe(recipe)
    algorithm = recipe.algorithm
    if not isinstance(
        algorithm,
        (AnyFlowFARPretrainAlgorithmSpec, AnyFlowBidirectionalPretrainAlgorithmSpec),
    ):
        raise TypeError("AnyFlow pretraining requires an AnyFlow pretrain recipe")
    if recipe.optimizer.type != "adamw":
        raise ValueError("released AnyFlow pretraining requires optimizer.type='adamw'")
    module = _module(student, role="AnyFlow student")
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="AnyFlow student",
    )
    if isinstance(algorithm, AnyFlowFARPretrainAlgorithmSpec):
        if not isinstance(student, AnyFlowFARAdapter):
            raise TypeError("AnyFlow FAR pretraining requires AnyFlowFARAdapter")
        config: AnyFlowPretrainConfig | AnyFlowBidirectionalPretrainConfig = AnyFlowPretrainConfig(
            flow_map=_map_config(algorithm.flow_map),
            far=_far_config(algorithm.far),
            bidirectional_modeling_probability=(algorithm.bidirectional_modeling_probability),
            conditioning_dropout_probability=(algorithm.conditioning_dropout_probability),
        )
    else:
        if not isinstance(student, AnyFlowBidirectionalAdapter):
            raise TypeError("AnyFlow bidirectional pretraining requires AnyFlowBidirectionalAdapter")
        config = AnyFlowBidirectionalPretrainConfig(
            flow_map=_map_config(algorithm.flow_map),
            image_conditioning_probability=(algorithm.image_conditioning_probability),
            conditioning_dropout_probability=(algorithm.conditioning_dropout_probability),
        )
    context, decisions = _parallel_and_decisions(
        parallel_context,
        seed=recipe.data.shuffle_seed,
    )
    loss_adapter: AnyFlowPretrainLossAdapter
    if isinstance(config, AnyFlowPretrainConfig):
        loss_adapter = NativeAnyFlowPretrainLossAdapter(
            student,
            config,
            decisions,
            parallel_context=context,
        )
    else:
        loss_adapter = NativeAnyFlowBidirectionalPretrainLossAdapter(
            student,
            config,
            decisions,
            parallel_context=context,
        )
    optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        module,
        fused=fused_adamw,
        role="AnyFlow student",
    )
    scheduler = _scheduler(
        scheduler_factory,
        optimizer,
        role="AnyFlow student",
    )
    engine = NativeAnyFlowPretrainEngine(
        student_module=module,
        loss_adapter=loss_adapter,
        optimizer=optimizer,
        decisions=decisions,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        gradient_accumulation_steps=(recipe.optimizer.gradient_accumulation_steps),
        scheduler=scheduler,
        parallel_context=context,
    )
    return NativeAnyFlowPretrainingStack(
        recipe=recipe,
        config=config,
        loss_adapter=loss_adapter,
        decisions=decisions,
        optimizer=optimizer,
        scheduler=scheduler,
        engine=engine,
        model=nn.ModuleDict({"student": module}),
        scheduler_state=named_stateful_collection(student=scheduler),
    )


def build_native_anyflow_on_policy_training_stack(
    recipe: PostTrainingRecipe,
    *,
    student: AnyFlowFARAdapter | AnyFlowBidirectionalAdapter,
    real_score: AnyFlowScoreAdapter,
    fake_score: AnyFlowScoreAdapter,
    student_scheduler_factory: Callable[[torch.optim.Optimizer], object] | None = None,
    fake_score_scheduler_factory: Callable[[torch.optim.Optimizer], object] | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeAnyFlowOnPolicyTrainingStack:
    """Build on-policy DMD, fresh fake score, FlowMap cotraining, and EMA."""

    validate_post_training_recipe(recipe)
    algorithm = recipe.algorithm
    if not isinstance(
        algorithm,
        (AnyFlowFAROnPolicyAlgorithmSpec, AnyFlowBidirectionalOnPolicyAlgorithmSpec),
    ):
        raise TypeError("AnyFlow on-policy training requires an on-policy recipe")
    if recipe.fake_score_optimizer is None:
        raise ValueError("AnyFlow on-policy training requires fake_score_optimizer")
    if recipe.optimizer.type != "adamw" or recipe.fake_score_optimizer.type != "adamw":
        raise ValueError("released AnyFlow on-policy roles require optimizer.type='adamw'")
    if not isinstance(real_score, AnyFlowScoreAdapter):
        raise TypeError("real_score must implement AnyFlowScoreAdapter")
    if not isinstance(fake_score, AnyFlowScoreAdapter):
        raise TypeError("fake_score must implement AnyFlowScoreAdapter")
    modules = {
        "student": _module(student, role="AnyFlow student"),
        "real_score": _module(real_score, role="AnyFlow real score"),
        "fake_score": _module(fake_score, role="AnyFlow fake score"),
    }
    require_independent_modules(modules)
    require_checkpoint_identity(
        student,
        recipe.model.checkpoint,
        role="AnyFlow student",
    )
    require_checkpoint_identity(
        real_score,
        algorithm.real_score_checkpoint,
        role="AnyFlow real score",
    )
    require_checkpoint_identity(
        fake_score,
        algorithm.fake_score_checkpoint,
        role="AnyFlow fake score",
    )
    modules["real_score"].requires_grad_(False)
    modules["real_score"].eval()
    common = {
        "flow_map": _map_config(algorithm.flow_map),
        "inference_steps": algorithm.inference_steps,
        "dmd_weight": algorithm.dmd_weight,
        "real_guidance_scale": algorithm.real_guidance_scale,
        "fake_score_logit_mean": algorithm.fake_score_logit_mean,
        "fake_score_logit_std": algorithm.fake_score_logit_std,
        "dmd_batch_size": algorithm.dmd_batch_size,
        "dmd_min_timestep": algorithm.dmd_min_timestep,
        "dmd_max_timestep": algorithm.dmd_max_timestep,
        "conditioning_dropout_probability": (algorithm.conditioning_dropout_probability),
        "cotrain_flowmap": algorithm.cotrain_flowmap,
        "discriminator_update_ratio": algorithm.discriminator_update_ratio,
        "ema_decay": algorithm.ema_decay,
        "ema_warmup_steps": algorithm.ema_warmup_steps,
        "synchronized_seed": algorithm.synchronized_seed,
    }
    if isinstance(algorithm, AnyFlowFAROnPolicyAlgorithmSpec):
        if not isinstance(student, AnyFlowFARAdapter):
            raise TypeError("AnyFlow FAR on-policy training requires AnyFlowFARAdapter")
        config: AnyFlowOnPolicyConfig | AnyFlowBidirectionalOnPolicyConfig = AnyFlowOnPolicyConfig(
            **common,
            far=_far_config(algorithm.far),
            bidirectional_modeling_probability=(algorithm.bidirectional_modeling_probability),
        )
    else:
        if not isinstance(student, AnyFlowBidirectionalAdapter):
            raise TypeError("AnyFlow bidirectional on-policy training requires AnyFlowBidirectionalAdapter")
        config = AnyFlowBidirectionalOnPolicyConfig(
            **common,
            image_conditioning_probability=(algorithm.image_conditioning_probability),
        )
    context, decisions = _parallel_and_decisions(
        parallel_context,
        seed=algorithm.synchronized_seed,
    )
    loss_adapter: AnyFlowOnPolicyLossAdapter
    if isinstance(config, AnyFlowOnPolicyConfig):
        loss_adapter = NativeAnyFlowOnPolicyLossAdapter(
            student,
            real_score,
            fake_score,
            config,
            decisions,
            parallel_context=context,
        )
    else:
        loss_adapter = NativeAnyFlowBidirectionalOnPolicyLossAdapter(
            student,
            real_score,
            fake_score,
            config,
            decisions,
            parallel_context=context,
        )
    student_optimizer = build_post_training_optimizer(
        replace(recipe.optimizer, gradient_accumulation_steps=1),
        modules["student"],
        fused=fused_adamw,
        role="AnyFlow student",
    )
    fake_optimizer = build_post_training_optimizer(
        replace(recipe.fake_score_optimizer, gradient_accumulation_steps=1),
        modules["fake_score"],
        fused=fused_adamw,
        role="AnyFlow fake score",
    )
    if recipe.optimizer.gradient_accumulation_steps != recipe.fake_score_optimizer.gradient_accumulation_steps:
        raise ValueError("AnyFlow student and fake-score gradient accumulation steps must match")
    student_scheduler = _scheduler(
        student_scheduler_factory,
        student_optimizer,
        role="AnyFlow student",
    )
    fake_scheduler = _scheduler(
        fake_score_scheduler_factory,
        fake_optimizer,
        role="AnyFlow fake score",
    )
    ema = AnyFlowEMA(
        modules["student"],
        decay=config.ema_decay,
        warmup_steps=config.ema_warmup_steps,
    )
    engine = NativeAnyFlowOnPolicyEngine(
        student_module=modules["student"],
        real_score_module=modules["real_score"],
        fake_score_module=modules["fake_score"],
        loss_adapter=loss_adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_optimizer,
        decisions=decisions,
        discriminator_update_ratio=config.discriminator_update_ratio,
        student_max_grad_norm=recipe.optimizer.max_grad_norm,
        fake_score_max_grad_norm=recipe.fake_score_optimizer.max_grad_norm,
        gradient_accumulation_steps=(recipe.optimizer.gradient_accumulation_steps),
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_scheduler,
        student_ema=ema,
        parallel_context=context,
    )
    return NativeAnyFlowOnPolicyTrainingStack(
        recipe=recipe,
        config=config,
        loss_adapter=loss_adapter,
        decisions=decisions,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_optimizer,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_scheduler,
        ema=ema,
        engine=engine,
        model=nn.ModuleDict(modules),
        scheduler_state=named_stateful_collection(
            student=student_scheduler,
            fake_score=fake_scheduler,
        ),
    )


__all__ = [
    "NativeAnyFlowOnPolicyTrainingStack",
    "NativeAnyFlowPretrainingStack",
    "build_native_anyflow_on_policy_training_stack",
    "build_native_anyflow_pretraining_stack",
]
