"""Native HunyuanVideo flow-policy and DiffusionNFT materialization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch

from worldfoundry.training.distributed.fsdp import (
    FSDP2Application,
    apply_fsdp2,
    apply_fsdp2_frozen_reference,
)
from worldfoundry.training.distributed.parallel import ParallelPlan
from worldfoundry.training.models.hunyuan_video import (
    HUNYUAN_VIDEO_MODEL_RECIPES,
    HunyuanVideoTrainAdapter,
    hunyuan_video_model_contract,
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
from worldfoundry.training.post_training.rl.batching import NativeFlowPolicyDataLoader
from worldfoundry.training.post_training.shared.distributed import PostTrainingParallelContext
from worldfoundry.training.post_training.shared.prediction import NativeFlowPredictionAdapter
from worldfoundry.training.recipes.post_training.algorithms.dance_grpo import (
    DanceGRPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.diffusion_nft import (
    DiffusionNFTAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.flow_dppo import (
    FlowDPPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.flow_grpo import (
    FlowGRPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.flow_policy import (
    FlowPolicyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.mix_grpo import (
    MixGRPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.peft import PeftLoraApplication

from .roles import (
    apply_hunyuan_video_activation_checkpointing,
    apply_hunyuan_video_tuning,
    load_hunyuan_video_role_adapter,
    torch_dtype,
    validate_hunyuan_video_dtype,
)

_DATA_OPTIONS = frozenset(
    {
        "generation",
        "global_prompt_batch_size",
        "rollout_forward_batch_size",
        "replay_microbatch_size",
    }
)


@dataclass(frozen=True, slots=True)
class HunyuanVideoRLDataPlan:
    generation: Mapping[str, int]
    global_prompt_batch_size: int = 1
    rollout_forward_batch_size: int | None = None
    replay_microbatch_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", MappingProxyType(dict(self.generation)))


@dataclass(frozen=True, slots=True)
class HunyuanVideoFlowPolicyMaterialization:
    policy: HunyuanVideoTrainAdapter
    reference_policy: HunyuanVideoTrainAdapter | None
    policy_tuning: PeftLoraApplication | None
    policy_fsdp: FSDP2Application | None
    reference_fsdp: FSDP2Application | None
    data_plan: HunyuanVideoRLDataPlan
    prediction: NativeFlowPredictionAdapter
    reference_prediction: NativeFlowPredictionAdapter | None
    stack: NativeFlowPolicyTrainingStack

    def build_rollout_loader(
        self,
        source: Iterable[object],
        *,
        generator: torch.Generator,
        device: str | torch.device,
        group_namespace: str | None = None,
    ) -> NativeFlowPolicyDataLoader:
        return NativeFlowPolicyDataLoader(
            source,
            group_size=self.stack.group_size,
            policy_revision=lambda: self.stack.engine.current_policy_revision,
            latent_shape=self.policy.latent_shape(self.data_plan.generation),
            sigmas=self.stack.sigmas,
            device=device,
            dtype=self.stack.sampler.trajectory_dtype,
            generator=generator,
            generation_defaults=self.data_plan.generation,
            group_namespace=group_namespace,
            init_same_noise=self.stack.init_same_noise,
        )


@dataclass(frozen=True, slots=True)
class HunyuanVideoDiffusionNFTRuntime:
    policy: HunyuanVideoTrainAdapter
    old_policy: HunyuanVideoTrainAdapter
    reference_policy: HunyuanVideoTrainAdapter | None
    policy_prediction: NativeFlowPredictionAdapter
    old_policy_prediction: NativeFlowPredictionAdapter
    reference_prediction: NativeFlowPredictionAdapter | None
    policy_tuning: PeftLoraApplication | None
    old_policy_tuning: PeftLoraApplication | None
    data_plan: HunyuanVideoRLDataPlan
    latent_shape: tuple[int, int, int, int]
    stack: NativeDiffusionNFTTrainingStack


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def build_hunyuan_video_data_plan(recipe: PostTrainingRecipe) -> HunyuanVideoRLDataPlan:
    options = dict(recipe.data.options)
    unknown = sorted(set(options) - _DATA_OPTIONS)
    if unknown:
        raise ValueError(f"unknown HunyuanVideo RL data options: {unknown}")
    raw_generation = options.get("generation")
    if not isinstance(raw_generation, Mapping) or set(raw_generation) != {
        "height",
        "width",
        "num_frames",
    }:
        raise ValueError("HunyuanVideo RL generation requires height, width, and num_frames")
    generation = {
        name: _positive_int(raw_generation[name], field=f"generation.{name}")
        for name in ("height", "width", "num_frames")
    }
    contract = hunyuan_video_model_contract(recipe.model.recipe)
    if generation["height"] % contract.spatial_compression or generation["width"] % contract.spatial_compression:
        raise ValueError(
            f"{recipe.model.recipe} spatial dimensions must be divisible by {contract.spatial_compression}"
        )
    if (generation["num_frames"] - 1) % contract.temporal_compression:
        raise ValueError("HunyuanVideo RL num_frames must satisfy 1 + 4k")
    return HunyuanVideoRLDataPlan(
        generation=generation,
        global_prompt_batch_size=_positive_int(
            options.get("global_prompt_batch_size", 1),
            field="global_prompt_batch_size",
        ),
        rollout_forward_batch_size=(
            None
            if options.get("rollout_forward_batch_size") is None
            else _positive_int(options["rollout_forward_batch_size"], field="rollout_forward_batch_size")
        ),
        replay_microbatch_size=(
            None
            if options.get("replay_microbatch_size") is None
            else _positive_int(options["replay_microbatch_size"], field="replay_microbatch_size")
        ),
    )


def _validate_common(recipe: PostTrainingRecipe) -> HunyuanVideoRLDataPlan:
    if recipe.model.recipe not in HUNYUAN_VIDEO_MODEL_RECIPES:
        raise ValueError("HunyuanVideo RL supports the original and 1.5 T2V graphs only")
    if recipe.data.cache is None:
        raise ValueError("HunyuanVideo RL requires cached text conditioning")
    if recipe.distributed.backend not in {"single", "fsdp2"}:
        raise ValueError("HunyuanVideo RL supports single-device or FSDP2 execution")
    if recipe.distributed.cp != 1 or recipe.distributed.tp != 1:
        raise ValueError("HunyuanVideo RL context/tensor parallel execution is not connected yet")
    if recipe.runtime.activation_checkpoint not in {"none", "full"}:
        raise ValueError("HunyuanVideo activation_checkpoint must be none or full")
    return build_hunyuan_video_data_plan(recipe)


def validate_hunyuan_video_flow_policy_recipe(
    recipe: PostTrainingRecipe,
) -> tuple[FlowPolicyAlgorithmSpec, HunyuanVideoRLDataPlan]:
    if not isinstance(
        recipe.algorithm,
        (
            FlowGRPOAlgorithmSpec,
            FlowDPPOAlgorithmSpec,
            DanceGRPOAlgorithmSpec,
            MixGRPOAlgorithmSpec,
        ),
    ):
        raise TypeError("HunyuanVideo supports FlowGRPO, FlowDPPO, DanceGRPO, or MixGRPO")
    if recipe.algorithm.guidance_scale > 1.0:
        raise ValueError("HunyuanVideo native RL uses one conditional branch; external two-branch CFG is not enabled")
    return recipe.algorithm, _validate_common(recipe)


def validate_hunyuan_video_diffusion_nft_recipe(
    recipe: PostTrainingRecipe,
) -> tuple[DiffusionNFTAlgorithmSpec, HunyuanVideoRLDataPlan]:
    if not isinstance(recipe.algorithm, DiffusionNFTAlgorithmSpec):
        raise TypeError("HunyuanVideo DiffusionNFT materialization requires diffusion-nft")
    if recipe.distributed.backend != "single":
        raise ValueError("DiffusionNFT old-policy refresh currently requires single-device execution")
    if recipe.algorithm.collection.guidance_scale != 1.0:
        raise ValueError("HunyuanVideo DiffusionNFT collection currently requires guidance_scale=1")
    plan = _validate_common(recipe)
    if plan.replay_microbatch_size is not None:
        raise ValueError("DiffusionNFT does not replay stochastic trajectories")
    return recipe.algorithm, plan


def _prepare_policy(
    recipe: PostTrainingRecipe,
    adapter: HunyuanVideoTrainAdapter,
) -> PeftLoraApplication | None:
    if adapter.model_recipe != recipe.model.recipe:
        raise ValueError("HunyuanVideo adapter graph differs from the recipe")
    expected_dtype = torch_dtype(recipe.runtime.param_dtype)
    validate_hunyuan_video_dtype(adapter, expected_dtype)
    application = apply_hunyuan_video_tuning(recipe, adapter)
    if recipe.runtime.activation_checkpoint == "full":
        apply_hunyuan_video_activation_checkpointing(adapter)
    return application


def build_hunyuan_video_flow_policy_materialization(
    recipe: PostTrainingRecipe,
    *,
    policy: HunyuanVideoTrainAdapter,
    reference_policy: HunyuanVideoTrainAdapter | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> HunyuanVideoFlowPolicyMaterialization:
    algorithm, data_plan = validate_hunyuan_video_flow_policy_recipe(recipe)
    policy_tuning = _prepare_policy(recipe, policy)
    dtype = torch_dtype(recipe.runtime.param_dtype)
    parallel = PostTrainingParallelContext.current()
    policy_fsdp: FSDP2Application | None = None
    reference_fsdp: FSDP2Application | None = None
    if reference_policy is not None:
        if reference_policy.model_recipe != recipe.model.recipe:
            raise ValueError("HunyuanVideo reference graph differs from the policy graph")
        validate_hunyuan_video_dtype(reference_policy, dtype)
        reference_policy.trainable_module.requires_grad_(False)
        reference_policy.trainable_module.eval()

    if recipe.distributed.backend == "fsdp2" and parallel.world_size > 1:
        plan = ParallelPlan.resolve(recipe.distributed, world_size=parallel.world_size)
        mesh = plan.build_device_mesh(next(policy.trainable_module.parameters()).device.type)
        policy_fsdp = apply_fsdp2(
            policy,
            plan=plan,
            mesh=mesh,
            param_dtype=dtype,
            reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
        )
        if reference_policy is not None:
            reference_fsdp = apply_fsdp2_frozen_reference(
                reference_policy,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
            )

    autocast_dtype = None if dtype is torch.float32 else dtype
    prediction = NativeFlowPredictionAdapter(
        policy,
        autocast_dtype=autocast_dtype,
        checkpoint_identity=recipe.model.checkpoint,
    )
    reference_prediction = (
        None
        if reference_policy is None
        else NativeFlowPredictionAdapter(
            reference_policy,
            autocast_dtype=autocast_dtype,
            checkpoint_identity=algorithm.reference_checkpoint,
        )
    )
    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=prediction,
        initial_policy_revision=recipe.model.checkpoint,
        reference_policy=reference_prediction,
        parallel_context=parallel,
        fused_adamw=fused_adamw,
        rollout_forward_batch_size=data_plan.rollout_forward_batch_size,
        replay_microbatch_size=data_plan.replay_microbatch_size,
    )
    return HunyuanVideoFlowPolicyMaterialization(
        policy=policy,
        reference_policy=reference_policy,
        policy_tuning=policy_tuning,
        policy_fsdp=policy_fsdp,
        reference_fsdp=reference_fsdp,
        data_plan=data_plan,
        prediction=prediction,
        reference_prediction=reference_prediction,
        stack=stack,
    )


def materialize_hunyuan_video_flow_policy(
    recipe: PostTrainingRecipe,
    *,
    device: str | torch.device = "cuda",
    role_checkpoint_overrides: Mapping[str, object] | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> HunyuanVideoFlowPolicyMaterialization:
    """Load independent native roles and construct the executable RL stack."""

    algorithm, _ = validate_hunyuan_video_flow_policy_recipe(recipe)
    overrides = dict(role_checkpoint_overrides or {})
    unknown = sorted(set(overrides) - {"policy", "reference"})
    if unknown:
        raise ValueError(f"unknown HunyuanVideo role checkpoint overrides: {unknown}")
    if recipe.model.checkpoint != "default" and "policy" not in overrides:
        raise ValueError("a non-default policy checkpoint requires an explicit native checkpoint override")
    dtype = torch_dtype(recipe.runtime.param_dtype)
    policy = load_hunyuan_video_role_adapter(
        model_recipe=recipe.model.recipe,
        checkpoint=overrides.get("policy"),
        device=device,
        dtype=dtype,
    )
    reference = None
    if algorithm.requires_reference_policy:
        if algorithm.reference_checkpoint != "default" and "reference" not in overrides:
            raise ValueError("a non-default reference checkpoint requires an explicit native checkpoint override")
        reference = load_hunyuan_video_role_adapter(
            model_recipe=recipe.model.recipe,
            checkpoint=overrides.get("reference"),
            device=device,
            dtype=dtype,
        )
    return build_hunyuan_video_flow_policy_materialization(
        recipe,
        policy=policy,
        reference_policy=reference,
        fused_adamw=fused_adamw,
    )


def build_hunyuan_video_diffusion_nft_stack(
    recipe: PostTrainingRecipe,
    *,
    policy: HunyuanVideoTrainAdapter,
    old_policy: HunyuanVideoTrainAdapter,
    reward_adapter: DiffusionNFTRewardAdapter,
    reference_policy: HunyuanVideoTrainAdapter | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> HunyuanVideoDiffusionNFTRuntime:
    """Bind HunyuanVideo roles to the shared DiffusionNFT implementation."""

    _, data_plan = validate_hunyuan_video_diffusion_nft_recipe(recipe)
    dtype = torch_dtype(recipe.runtime.param_dtype)
    for adapter in (policy, old_policy, reference_policy):
        if adapter is None:
            continue
        if adapter.model_recipe != recipe.model.recipe:
            raise ValueError("all HunyuanVideo DiffusionNFT roles must use the same graph")
        validate_hunyuan_video_dtype(adapter, dtype)
    policy_tuning = _prepare_policy(recipe, policy)
    old_policy_tuning = _prepare_policy(recipe, old_policy)
    old_policy.trainable_module.requires_grad_(False)
    old_policy.trainable_module.eval()
    if reference_policy is not None:
        reference_policy.trainable_module.requires_grad_(False)
        reference_policy.trainable_module.eval()
    autocast = None if dtype is torch.float32 else dtype
    prediction = NativeFlowPredictionAdapter(
        policy,
        autocast_dtype=autocast,
        checkpoint_identity=recipe.model.checkpoint,
    )
    old_prediction = NativeFlowPredictionAdapter(
        old_policy,
        autocast_dtype=autocast,
        checkpoint_identity=recipe.model.checkpoint,
    )
    reference_prediction = (
        None
        if reference_policy is None
        else NativeFlowPredictionAdapter(
            reference_policy,
            autocast_dtype=autocast,
            checkpoint_identity=recipe.algorithm.reference_checkpoint,
        )
    )
    stack = build_native_diffusion_nft_training_stack(
        recipe,
        policy=prediction,
        old_policy=old_prediction,
        initial_old_policy_revision=recipe.model.checkpoint,
        reward_adapter=reward_adapter,
        reference_policy=reference_prediction,
        parallel_context=PostTrainingParallelContext.current(),
        fused_adamw=fused_adamw,
    )
    return HunyuanVideoDiffusionNFTRuntime(
        policy=policy,
        old_policy=old_policy,
        reference_policy=reference_policy,
        policy_prediction=prediction,
        old_policy_prediction=old_prediction,
        reference_prediction=reference_prediction,
        policy_tuning=policy_tuning,
        old_policy_tuning=old_policy_tuning,
        data_plan=data_plan,
        latent_shape=policy.latent_shape(data_plan.generation),
        stack=stack,
    )


def materialize_hunyuan_video_diffusion_nft(
    recipe: PostTrainingRecipe,
    *,
    reward_adapter: DiffusionNFTRewardAdapter,
    device: str | torch.device = "cuda",
    role_checkpoint_overrides: Mapping[str, object] | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> HunyuanVideoDiffusionNFTRuntime:
    """Load independent native policy roles and construct DiffusionNFT."""

    algorithm, _ = validate_hunyuan_video_diffusion_nft_recipe(recipe)
    overrides = dict(role_checkpoint_overrides or {})
    unknown = sorted(set(overrides) - {"policy", "old_policy", "reference"})
    if unknown:
        raise ValueError(f"unknown HunyuanVideo DiffusionNFT role overrides: {unknown}")
    if recipe.model.checkpoint != "default" and "policy" not in overrides:
        raise ValueError("a non-default policy checkpoint requires a native role override")
    if (
        algorithm.reference_mse_weight > 0
        and algorithm.reference_checkpoint != "default"
        and "reference" not in overrides
    ):
        raise ValueError("a non-default reference checkpoint requires a native role override")
    dtype = torch_dtype(recipe.runtime.param_dtype)
    policy = load_hunyuan_video_role_adapter(
        model_recipe=recipe.model.recipe,
        checkpoint=overrides.get("policy"),
        device=device,
        dtype=dtype,
    )
    old_policy = load_hunyuan_video_role_adapter(
        model_recipe=recipe.model.recipe,
        checkpoint=overrides.get("old_policy", overrides.get("policy")),
        device=device,
        dtype=dtype,
    )
    reference = None
    if algorithm.reference_mse_weight > 0:
        reference = load_hunyuan_video_role_adapter(
            model_recipe=recipe.model.recipe,
            checkpoint=overrides.get("reference"),
            device=device,
            dtype=dtype,
        )
    return build_hunyuan_video_diffusion_nft_stack(
        recipe,
        policy=policy,
        old_policy=old_policy,
        reference_policy=reference,
        reward_adapter=reward_adapter,
        fused_adamw=fused_adamw,
    )


__all__ = [
    "HunyuanVideoDiffusionNFTRuntime",
    "HunyuanVideoFlowPolicyMaterialization",
    "HunyuanVideoRLDataPlan",
    "build_hunyuan_video_data_plan",
    "build_hunyuan_video_diffusion_nft_stack",
    "build_hunyuan_video_flow_policy_materialization",
    "materialize_hunyuan_video_flow_policy",
    "materialize_hunyuan_video_diffusion_nft",
    "validate_hunyuan_video_diffusion_nft_recipe",
    "validate_hunyuan_video_flow_policy_recipe",
]
