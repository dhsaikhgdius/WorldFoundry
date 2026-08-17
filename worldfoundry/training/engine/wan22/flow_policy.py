"""Wan2.2 A14B profiles and native RL stack materialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch

from worldfoundry.training.data.wan22.rollout_cache import WAN22_T2V_A14B_MODEL
from worldfoundry.training.models.wan22 import (
    WAN22_A14B_BOUNDARY_RATIO,
    WAN22_DUAL_ATTENTION,
    Wan22TrainAdapter,
)
from worldfoundry.training.post_training.rl.algorithms.diffusion_nft.builder import (
    NativeDiffusionNFTTrainingStack,
    build_native_diffusion_nft_training_stack,
)
from worldfoundry.training.post_training.rl.algorithms.diffusion_nft.contracts import (
    DiffusionNFTRewardAdapter,
)
from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (
    NativeFlowPolicyTrainingStack,
    build_native_flow_policy_training_stack,
)
from worldfoundry.training.post_training.rl.transitions.flow_sde import (
    flow_match_sigma_schedule,
)
from worldfoundry.training.post_training.shared.prediction import NativeFlowPredictionAdapter
from worldfoundry.training.recipes.post_training.algorithms.diffusion_nft import (
    DiffusionNFTAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.flow_policy import (
    FlowPolicyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.peft import PeftLoraApplication

from .tuning import apply_wan22_tuning


def wan22_flow_policy_sigmas(
    num_inference_steps: int = 20,
    *,
    shift: float = 5.0,
) -> tuple[float, ...]:
    """Return the shifted linear flow schedule used by the A14B RL profile."""

    if isinstance(num_inference_steps, bool) or int(num_inference_steps) < 2:
        raise ValueError("Wan2.2 policy rollout requires at least two inference steps")
    return flow_match_sigma_schedule(int(num_inference_steps), shift=shift)


@dataclass(frozen=True, slots=True)
class Wan22FlowPolicyProfile:
    """Practical trainside A14B defaults derived from the released RL setup."""

    model_recipe: str
    boundary_ratio: float
    generation: Mapping[str, int]
    sigmas: tuple[float, ...]
    sde_step_indices: tuple[int, ...]
    guidance_scale: float = 1.0
    eta: float = 0.7
    group_size: int = 16
    trajectory_dtype: str = "float16"
    lora_rank: int = 64
    lora_alpha: int = 128
    learning_rate: float = 3.0e-4
    old_log_prob_source: str = "rollout"
    global_prompt_batch_size: int = 48

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", MappingProxyType(dict(self.generation)))

    @property
    def sigma_max(self) -> float:
        return self.sigmas[1]


def wan22_flow_policy_profile() -> Wan22FlowPolicyProfile:
    sigmas = wan22_flow_policy_sigmas()
    return Wan22FlowPolicyProfile(
        model_recipe=WAN22_T2V_A14B_MODEL,
        boundary_ratio=WAN22_A14B_BOUNDARY_RATIO,
        generation={"height": 480, "width": 640, "num_frames": 1},
        sigmas=sigmas,
        sde_step_indices=tuple(range(4)),
    )


@dataclass(frozen=True, slots=True)
class Wan22FlowPolicyDataPlan:
    generation: Mapping[str, int]
    boundary_ratio: float
    global_prompt_batch_size: int
    rollout_forward_batch_size: int | None
    replay_microbatch_size: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", MappingProxyType(dict(self.generation)))


def _optional_positive_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _data_plan(recipe: PostTrainingRecipe) -> Wan22FlowPolicyDataPlan:
    raw_generation = recipe.data.options.get("generation")
    if not isinstance(raw_generation, Mapping):
        raise TypeError("Wan2.2 policy data.options.generation must be a mapping")
    generation = {name: int(raw_generation[name]) for name in ("height", "width", "num_frames")}
    if generation["height"] % 16 or generation["width"] % 16:
        raise ValueError("Wan2.2 policy height and width must be divisible by 16")
    if generation["num_frames"] < 1 or (generation["num_frames"] - 1) % 4:
        raise ValueError("Wan2.2 policy num_frames must satisfy 1 + 4k")
    boundary = float(recipe.model.options.get("boundary_ratio", WAN22_A14B_BOUNDARY_RATIO))
    if not 0.0 < boundary < 1.0:
        raise ValueError("Wan2.2 model.options.boundary_ratio must be in (0, 1)")
    return Wan22FlowPolicyDataPlan(
        generation=generation,
        boundary_ratio=boundary,
        global_prompt_batch_size=_positive_int(
            recipe.data.options.get("global_prompt_batch_size", 1),
            field_name="data.options.global_prompt_batch_size",
        ),
        rollout_forward_batch_size=_optional_positive_int(
            recipe.data.options.get("rollout_forward_batch_size"),
            field_name="data.options.rollout_forward_batch_size",
        ),
        replay_microbatch_size=_optional_positive_int(
            recipe.data.options.get("replay_microbatch_size"),
            field_name="data.options.replay_microbatch_size",
        ),
    )


def _validate_common(recipe: PostTrainingRecipe) -> Wan22FlowPolicyDataPlan:
    if recipe.model.recipe != WAN22_T2V_A14B_MODEL:
        raise ValueError("Wan2.2 A14B policy training cannot use the TI2V-5B recipe")
    if recipe.distributed.backend not in {"single", "fsdp2"}:
        raise ValueError("Wan2.2 policy training supports single-device and FSDP2 execution")
    if recipe.distributed.cp != 1 or recipe.distributed.tp != 1:
        raise ValueError("Wan2.2 policy context and tensor parallelism are not implemented")
    if recipe.tuning.mode == "lora" and recipe.tuning.preset != WAN22_DUAL_ATTENTION:
        raise ValueError(f"Wan2.2 LoRA requires tuning.preset={WAN22_DUAL_ATTENTION!r}")
    return _data_plan(recipe)


def validate_wan22_flow_policy_recipe(
    recipe: PostTrainingRecipe,
) -> tuple[FlowPolicyAlgorithmSpec, Wan22FlowPolicyDataPlan]:
    if not isinstance(recipe.algorithm, FlowPolicyAlgorithmSpec):
        raise TypeError("Wan2.2 flow policy requires a flow-policy algorithm")
    return recipe.algorithm, _validate_common(recipe)


def validate_wan22_diffusion_nft_recipe(
    recipe: PostTrainingRecipe,
) -> tuple[DiffusionNFTAlgorithmSpec, Wan22FlowPolicyDataPlan]:
    if not isinstance(recipe.algorithm, DiffusionNFTAlgorithmSpec):
        raise TypeError("Wan2.2 DiffusionNFT requires a diffusion-nft algorithm")
    return recipe.algorithm, _validate_common(recipe)


def _latent_shape(
    adapter: Wan22TrainAdapter,
    plan: Wan22FlowPolicyDataPlan,
) -> tuple[int, int, int, int]:
    generation = plan.generation
    return (
        adapter.expected_latent_channels,
        1 + (generation["num_frames"] - 1) // adapter.temporal_compression,
        generation["height"] // adapter.spatial_compression,
        generation["width"] // adapter.spatial_compression,
    )


def _prediction(
    adapter: Wan22TrainAdapter,
    *,
    checkpoint_identity: str | None = None,
) -> NativeFlowPredictionAdapter:
    parameter = next(adapter.trainable_module.parameters())
    return NativeFlowPredictionAdapter(
        adapter,
        autocast_dtype=None if parameter.dtype is torch.float32 else parameter.dtype,
        checkpoint_identity=checkpoint_identity,
    )


@dataclass(frozen=True, slots=True)
class Wan22FlowPolicyRuntime:
    prediction: NativeFlowPredictionAdapter
    stack: NativeFlowPolicyTrainingStack
    data_plan: Wan22FlowPolicyDataPlan
    latent_shape: tuple[int, int, int, int]
    policy_tuning: PeftLoraApplication | None


@dataclass(frozen=True, slots=True)
class Wan22DiffusionNFTRuntime:
    policy_prediction: NativeFlowPredictionAdapter
    old_policy_prediction: NativeFlowPredictionAdapter
    stack: NativeDiffusionNFTTrainingStack
    data_plan: Wan22FlowPolicyDataPlan
    latent_shape: tuple[int, int, int, int]
    policy_tuning: PeftLoraApplication | None
    old_policy_tuning: PeftLoraApplication | None


def materialize_wan22_flow_policy_stack(
    recipe: PostTrainingRecipe,
    *,
    policy: Wan22TrainAdapter,
    initial_policy_revision: str,
    reference_policy: Wan22TrainAdapter | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> Wan22FlowPolicyRuntime:
    """Connect A14B to FlowGRPO, FlowDPPO, DanceGRPO, MixGRPO, and guards."""

    _, plan = validate_wan22_flow_policy_recipe(recipe)
    if policy.boundary_ratio != plan.boundary_ratio:
        raise ValueError("loaded Wan2.2 policy uses another expert boundary")
    tuning = apply_wan22_tuning(recipe, policy)
    prediction = _prediction(policy, checkpoint_identity=recipe.model.checkpoint)
    reference_prediction = None
    if reference_policy is not None:
        if reference_policy.boundary_ratio != plan.boundary_ratio:
            raise ValueError("loaded Wan2.2 reference uses another expert boundary")
        reference_policy.trainable_module.requires_grad_(False)
        reference_policy.trainable_module.eval()
        reference_prediction = _prediction(
            reference_policy,
            checkpoint_identity=recipe.algorithm.reference_checkpoint,
        )
    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=prediction,
        initial_policy_revision=initial_policy_revision,
        reference_policy=reference_prediction,
        fused_adamw=fused_adamw,
        rollout_forward_batch_size=plan.rollout_forward_batch_size,
        replay_microbatch_size=plan.replay_microbatch_size,
    )
    return Wan22FlowPolicyRuntime(
        prediction=prediction,
        stack=stack,
        data_plan=plan,
        latent_shape=_latent_shape(policy, plan),
        policy_tuning=tuning,
    )


def materialize_wan22_diffusion_nft_stack(
    recipe: PostTrainingRecipe,
    *,
    policy: Wan22TrainAdapter,
    old_policy: Wan22TrainAdapter,
    initial_old_policy_revision: str,
    reward_adapter: DiffusionNFTRewardAdapter,
    reference_policy: Wan22TrainAdapter | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> Wan22DiffusionNFTRuntime:
    """Connect independently loaded A14B roles to native DiffusionNFT."""

    _, plan = validate_wan22_diffusion_nft_recipe(recipe)
    for role in (policy, old_policy, reference_policy):
        if role is not None and role.boundary_ratio != plan.boundary_ratio:
            raise ValueError("loaded Wan2.2 role uses another expert boundary")
    policy_tuning = apply_wan22_tuning(recipe, policy)
    old_tuning = apply_wan22_tuning(recipe, old_policy)
    old_policy.trainable_module.requires_grad_(False)
    old_policy.trainable_module.eval()
    if reference_policy is not None:
        reference_policy.trainable_module.requires_grad_(False)
        reference_policy.trainable_module.eval()
    policy_prediction = _prediction(policy)
    old_prediction = _prediction(old_policy)
    reference_prediction = None if reference_policy is None else _prediction(reference_policy)
    stack = build_native_diffusion_nft_training_stack(
        recipe,
        policy=policy_prediction,
        old_policy=old_prediction,
        initial_old_policy_revision=initial_old_policy_revision,
        reward_adapter=reward_adapter,
        reference_policy=reference_prediction,
        fused_adamw=fused_adamw,
    )
    return Wan22DiffusionNFTRuntime(
        policy_prediction=policy_prediction,
        old_policy_prediction=old_prediction,
        stack=stack,
        data_plan=plan,
        latent_shape=_latent_shape(policy, plan),
        policy_tuning=policy_tuning,
        old_policy_tuning=old_tuning,
    )


__all__ = [
    "WAN22_T2V_A14B_MODEL",
    "Wan22DiffusionNFTRuntime",
    "Wan22FlowPolicyDataPlan",
    "Wan22FlowPolicyProfile",
    "Wan22FlowPolicyRuntime",
    "materialize_wan22_diffusion_nft_stack",
    "materialize_wan22_flow_policy_stack",
    "validate_wan22_diffusion_nft_recipe",
    "validate_wan22_flow_policy_recipe",
    "wan22_flow_policy_profile",
    "wan22_flow_policy_sigmas",
]
