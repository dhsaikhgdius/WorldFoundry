"""Construction of complete WorldFoundry-native flow-policy stacks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch

from worldfoundry.training.recipes.post_training.algorithms.dance_grpo import (
    DanceGRPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.flow_policy import (
    FlowPolicyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ....rewards.scalarization import WeightedRewardScalarizer
from ....shared.building import (
    build_post_training_optimizer,
    prediction_module,
    resolve_tensor_dtype,
    validate_post_training_recipe,
)
from ....shared.contracts import FlowPredictionAdapter
from ....shared.distributed import PostTrainingParallelContext
from ....shared.prediction import (
    NativeClassifierFreeGuidance,
    NativeFlowPredictionAdapter,
)
from ...contracts import FlowTrajectorySamplingAdapter
from ...rollout_strategies.contracts import FlowSDEIndexResolver
from ...rollout_strategies.sparse_sde_steps import FlowSDEIndexSchedule
from ...rollout_strategies.transition import (
    ConstantDiffusionFlowTransition,
    FlowTransitionStrategy,
    VariancePreservingFlowTransition,
)
from ...rollout_strategies.window_sde_steps import FlowSDEWindowSchedule
from ...trajectory import FlowTrajectorySampler, NativeFlowTrajectoryReplay
from .engine import NativeFlowPolicyEngine
from .runtime import resolve_flow_policy_algorithm_runtime


@dataclass(frozen=True, slots=True)
class NativeFlowPolicyTrainingStack:
    """Algorithm-neutral rollout, scalarization, replay, and update plane."""

    sampler: FlowTrajectorySamplingAdapter
    replay: NativeFlowTrajectoryReplay
    reference_replay: NativeFlowTrajectoryReplay | None
    scalarizer: WeightedRewardScalarizer
    optimizer: torch.optim.Optimizer
    engine: NativeFlowPolicyEngine
    session_type: type[object]
    transition_strategy: FlowTransitionStrategy
    sigmas: tuple[float, ...]
    sde_index_schedule: FlowSDEIndexResolver
    group_size: int
    init_same_noise: bool
    old_log_prob_source: str
    advantage_epsilon: float
    advantage_normalization: str
    advantage_clip_max: float | None
    session_kwargs: Mapping[str, object]

    @property
    def sde_step_indices(self) -> tuple[int, ...] | None:
        value = getattr(self.sde_index_schedule, "static_indices", None)
        return None if value is None else tuple(value)

    def checkpoint_state_kwargs(self) -> dict[str, object | None]:
        return {
            "lr_scheduler": None,
            "ema": None,
            "algorithm_state": self.scalarizer,
        }


def _classifier_free_guidance(
    adapter: FlowPredictionAdapter,
    *,
    guidance_scale: float,
    role: str,
) -> FlowPredictionAdapter:
    if guidance_scale <= 1:
        return adapter
    if not isinstance(adapter, NativeFlowPredictionAdapter):
        raise TypeError(f"native flow-policy CFG requires a NativeFlowPredictionAdapter {role}")
    return NativeClassifierFreeGuidance(adapter, guidance_scale=guidance_scale)


def _build_transition_strategy(
    algorithm: FlowPolicyAlgorithmSpec,
) -> FlowTransitionStrategy:
    if algorithm.transition_strategy == "variance-preserving":
        assert algorithm.sigma_max is not None
        return VariancePreservingFlowTransition(
            eta=algorithm.eta,
            sigma_max=algorithm.sigma_max,
        )
    if algorithm.transition_strategy == "constant-diffusion":
        return ConstantDiffusionFlowTransition(eta=algorithm.eta)
    raise ValueError(f"unsupported flow transition strategy: {algorithm.transition_strategy!r}")


def build_native_flow_policy_training_stack(
    recipe: PostTrainingRecipe,
    *,
    policy: FlowPredictionAdapter,
    initial_policy_revision: str,
    reference_policy: FlowPredictionAdapter | None = None,
    parallel_context: PostTrainingParallelContext | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    rollout_forward_batch_size: int | None = None,
    replay_microbatch_size: int | None = None,
) -> NativeFlowPolicyTrainingStack:
    """Build any registered flow-policy algorithm from native policy roles."""

    validate_post_training_recipe(recipe)
    if not isinstance(recipe.algorithm, FlowPolicyAlgorithmSpec):
        raise TypeError("flow-policy stack requires a FlowPolicyAlgorithmSpec recipe")
    algorithm = recipe.algorithm
    runtime = resolve_flow_policy_algorithm_runtime(algorithm)
    policy_module = prediction_module(policy, role=f"{runtime.display_name} policy")
    if algorithm.requires_reference_policy and reference_policy is None:
        raise ValueError(f"{algorithm.type} requires a frozen reference_policy")
    if not algorithm.requires_reference_policy and reference_policy is not None:
        raise ValueError("reference_policy is unused when reference_kl_weight is zero")

    active_policy = _classifier_free_guidance(
        policy,
        guidance_scale=algorithm.guidance_scale,
        role="policy",
    )
    reference_replay: NativeFlowTrajectoryReplay | None = None
    if reference_policy is not None:
        reference_module = prediction_module(
            reference_policy,
            role=f"{runtime.display_name} reference policy",
        )
        if any(parameter.requires_grad for parameter in reference_module.parameters()):
            raise ValueError("flow-policy reference policy parameters must be frozen")
        active_reference = _classifier_free_guidance(
            reference_policy,
            guidance_scale=algorithm.guidance_scale,
            role="reference policy",
        )
        reference_replay = NativeFlowTrajectoryReplay(active_reference)

    optimizer = build_post_training_optimizer(
        recipe.optimizer,
        policy_module,
        fused=fused_adamw,
        role=f"{runtime.display_name} policy",
    )
    transition_strategy = _build_transition_strategy(algorithm)
    sampler = FlowTrajectorySampler(
        active_policy,
        transition_strategy=transition_strategy,
        trajectory_dtype=resolve_tensor_dtype(algorithm.trajectory_dtype),
        forward_batch_size=rollout_forward_batch_size,
    )
    replay = NativeFlowTrajectoryReplay(active_policy)
    scalarizer = WeightedRewardScalarizer(
        algorithm.reward_weights,
        calibration_mean=getattr(algorithm.reward_model, "calibration_mean", None),
        calibration_std=getattr(algorithm.reward_model, "calibration_std", None),
        normalization_epsilon=float(getattr(algorithm.reward_model, "normalization_epsilon", 0.0)),
    )
    engine = runtime.engine_factory(
        algorithm,
        replay,
        optimizer,
        initial_policy_revision=initial_policy_revision,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        updates_per_trajectory=algorithm.updates_per_trajectory,
        reference_replay_adapter=reference_replay,
        reference_kl_weight=algorithm.reference_kl_weight,
        parallel_context=parallel_context,
        replay_microbatch_size=replay_microbatch_size,
    )
    if algorithm.sde_window is None:
        sde_index_schedule: FlowSDEIndexResolver = FlowSDEIndexSchedule(
            transition_count=len(algorithm.sigmas) - 1,
            static_indices=algorithm.sde_step_indices,
            timestep_fraction=algorithm.sde_timestep_fraction,
            num_sde_steps=algorithm.num_sde_steps,
        )
    else:
        window = algorithm.sde_window
        sde_index_schedule = FlowSDEWindowSchedule(
            transition_count=len(algorithm.sigmas) - 1,
            window_size=window.window_size,
            iterations_per_window=window.iterations_per_window,
            stride=window.stride,
            initial_index=window.initial_index,
            rollback=window.rollback,
        )
    session_kwargs: dict[str, object] = {}
    if isinstance(algorithm, DanceGRPOAlgorithmSpec):
        session_kwargs["update_timestep_fraction"] = algorithm.update_timestep_fraction
    return NativeFlowPolicyTrainingStack(
        sampler=sampler,
        replay=replay,
        reference_replay=reference_replay,
        scalarizer=scalarizer,
        optimizer=optimizer,
        engine=engine,
        session_type=runtime.session_type,
        transition_strategy=transition_strategy,
        sigmas=algorithm.sigmas,
        sde_index_schedule=sde_index_schedule,
        group_size=algorithm.group_size,
        init_same_noise=algorithm.init_same_noise,
        old_log_prob_source=algorithm.old_log_prob_source,
        advantage_epsilon=algorithm.advantage_epsilon,
        advantage_normalization=algorithm.advantage_normalization,
        advantage_clip_max=algorithm.advantage_clip_max,
        session_kwargs=MappingProxyType(session_kwargs),
    )


__all__ = [
    "NativeFlowPolicyTrainingStack",
    "build_native_flow_policy_training_stack",
]
