"""Construction of the WorldFoundry-native Diffusion-DPO training stack."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress
from worldfoundry.training.recipes.post_training.algorithms.diffusion_dpo import (
    DiffusionDPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ....shared.building import (
    build_post_training_optimizer,
    prediction_module,
    validate_post_training_recipe,
)
from ....shared.contracts import FlowPredictionAdapter
from ....shared.distributed import PostTrainingParallelContext
from .contracts import DiffusionDPOBatch
from .engine import NativeDiffusionDPOEngine
from .session import NativeDiffusionDPOTrainingSession


@dataclass(frozen=True, slots=True)
class NativeDiffusionDPOTrainingStack:
    """Optimizer, engine, and session factory for paired latent preferences."""

    optimizer: torch.optim.Optimizer
    engine: NativeDiffusionDPOEngine

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        """Return optional ``TrainingState`` components for exact resume."""

        return {
            "lr_scheduler": None,
            "ema": None,
            "algorithm_state": None,
        }

    def build_session(
        self,
        dataloader: Iterable[DiffusionDPOBatch],
        progress: TrainingProgress,
        *,
        checkpoint_state: object | None = None,
        checkpointer: TrainingCheckpointer | None = None,
        save_every_steps: int = 0,
        asynchronous_checkpoints: bool = False,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> NativeDiffusionDPOTrainingSession:
        return NativeDiffusionDPOTrainingSession(
            self.engine,
            dataloader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=save_every_steps,
            asynchronous_checkpoints=asynchronous_checkpoints,
            event_sink=event_sink,
        )


def _validate_reference_role(
    policy_module: torch.nn.Module,
    reference_module: torch.nn.Module,
) -> None:
    if policy_module is reference_module:
        raise ValueError("Diffusion-DPO policy and reference policy must be distinct modules")
    if any(parameter.requires_grad for parameter in reference_module.parameters()):
        raise ValueError("Diffusion-DPO reference policy parameters must be frozen")
    policy_parameter_ids = {id(parameter) for parameter in policy_module.parameters()}
    reference_parameter_ids = {id(parameter) for parameter in reference_module.parameters()}
    if policy_parameter_ids & reference_parameter_ids:
        raise ValueError("Diffusion-DPO policy and reference policy cannot share parameters")


def build_native_diffusion_dpo_training_stack(
    recipe: PostTrainingRecipe,
    *,
    policy: FlowPredictionAdapter,
    reference_policy: FlowPredictionAdapter,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeDiffusionDPOTrainingStack:
    """Build Diffusion-DPO from independently loaded policy and reference roles."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, DiffusionDPOAlgorithmSpec):
        raise TypeError("Diffusion-DPO stack requires a DiffusionDPOAlgorithmSpec recipe")
    policy_module = prediction_module(policy, role="Diffusion-DPO policy")
    reference_module = prediction_module(
        reference_policy,
        role="Diffusion-DPO reference policy",
    )
    _validate_reference_role(policy_module, reference_module)
    optimizer = build_post_training_optimizer(
        recipe.optimizer,
        policy_module,
        fused=fused_adamw,
        role="Diffusion-DPO policy",
    )
    engine = NativeDiffusionDPOEngine(
        policy,
        reference_policy,
        optimizer,
        beta=recipe.algorithm.beta,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        parallel_context=parallel_context,
    )
    return NativeDiffusionDPOTrainingStack(
        optimizer=optimizer,
        engine=engine,
    )


__all__ = [
    "NativeDiffusionDPOTrainingStack",
    "build_native_diffusion_dpo_training_stack",
]
