"""Executable flow-policy runs shared by native video model families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import torch

from worldfoundry.core.utils.torch_utils import set_seed_everywhere
from worldfoundry.training.data.loader import build_stateful_dataloader
from worldfoundry.training.data.rollout_cache import (
    RolloutConditioningDataset,
    collate_rollout_conditioned_prompts,
    resolve_rollout_generation_geometry,
)
from worldfoundry.training.data.rollout_manifest import RolloutPromptDataset
from worldfoundry.training.data.sampler import DeterministicDistributedSampler
from worldfoundry.training.distributed.flow_rollout import (
    attach_ray_flow_policy_rollout,
)
from worldfoundry.training.distributed.fsdp import (
    FSDP2Application,
    apply_fsdp2,
    apply_fsdp2_frozen_reference,
)
from worldfoundry.training.distributed.parallel import (
    DistributedTrainingContext,
    ParallelPlan,
)
from worldfoundry.training.distributed.rollout_runtime import (
    RayPostTrainingRuntime,
    RayPostTrainingRuntimeConfig,
    ray_runtime_config_from_rollout_spec,
)
from worldfoundry.training.post_training.rewards.contracts import RewardEvaluator
from worldfoundry.training.post_training.rewards.http import HTTPRewardEvaluator
from worldfoundry.training.post_training.rewards.videoalign import (
    build_videoalign_reward_evaluator,
)
from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (
    NativeFlowPolicyTrainingStack,
    build_native_flow_policy_training_stack,
)
from worldfoundry.training.post_training.rl.batching import NativeFlowPolicyDataLoader
from worldfoundry.training.post_training.rl.remote_rewards import (
    HTTPTerminalRewardAdapter,
)
from worldfoundry.training.post_training.rl.run import (
    NativeFlowPolicyTrainingRun,
    build_native_flow_policy_training_run,
)
from worldfoundry.training.post_training.rl.trajectory_rewards import (
    DecodedTerminalRewardAdapter,
)
from worldfoundry.training.post_training.shared.distributed import (
    PostTrainingParallelContext,
)
from worldfoundry.training.post_training.shared.prediction import (
    NativeFlowPredictionAdapter,
)
from worldfoundry.training.recipes.post_training.algorithms.flow_policy import (
    FlowPolicyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.post_training.rewards.remote import (
    RemoteRewardSpec,
)
from worldfoundry.training.recipes.post_training.rollout import RayRolloutSpec
from worldfoundry.training.tuning.peft import PeftLoraApplication

from .artifacts import create_run_directory

_HUNYUAN_MODELS = frozenset({"hunyuanvideo-t2v", "hunyuanvideo-1.5-t2v"})
_LTX_MODELS = frozenset({"ltx-2-i2v", "ltx-2.3-i2v"})
_WAN22_MODEL = "wan2.2-t2v-a14b"


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _ray_video_runtime_config(
    recipe: PostTrainingRecipe,
) -> RayPostTrainingRuntimeConfig:
    rollout = recipe.rollout
    if not isinstance(rollout, RayRolloutSpec):
        raise TypeError("Ray video rollout requires a RayRolloutSpec")
    if rollout.trainer_binding != "external" or rollout.placement != "separate":
        raise ValueError("video Ray rollout currently requires trainer_binding='external' and placement='separate'")
    if recipe.distributed.backend != "single":
        raise ValueError(
            "video Ray rollout currently requires distributed.backend='single'; "
            "a per-rank FSDP2 materializer cannot safely share one Ray rollout group"
        )
    if recipe.tuning.mode == "full" and rollout.weight_kind == "lora":
        raise ValueError("full policy tuning cannot use LoRA-only rollout weight sync")
    return ray_runtime_config_from_rollout_spec(rollout)


@dataclass(frozen=True, slots=True)
class VideoFlowPolicyMaterialization:
    """The common execution fields produced by one model-family loader."""

    policy: object
    reference_policy: object | None
    stack: NativeFlowPolicyTrainingStack
    generation: Mapping[str, int]
    latent_shape: tuple[int, int, int, int]
    policy_tuning: PeftLoraApplication | None
    policy_fsdp: FSDP2Application | None = None
    reference_fsdp: FSDP2Application | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", MappingProxyType(dict(self.generation)))

    @property
    def policy_module(self) -> torch.nn.Module:
        module = getattr(self.policy, "trainable_module", None)
        if not isinstance(module, torch.nn.Module):
            raise TypeError("video policy adapter must expose a trainable_module")
        return module

    def build_rollout_loader(
        self,
        source: object,
        *,
        generator: torch.Generator,
        device: str | torch.device,
        group_namespace: str | None,
    ) -> NativeFlowPolicyDataLoader:
        return NativeFlowPolicyDataLoader(
            source,  # type: ignore[arg-type]
            group_size=self.stack.group_size,
            policy_revision=lambda: self.stack.engine.current_policy_revision,
            latent_shape=self.latent_shape,
            sigmas=self.stack.sigmas,
            device=device,
            dtype=self.stack.sampler.trajectory_dtype,
            generator=generator,
            generation_defaults=self.generation,
            group_namespace=group_namespace,
            init_same_noise=self.stack.init_same_noise,
        )


def _apply_distributed_wrapping(
    recipe: PostTrainingRecipe,
    *,
    policy: object,
    reference_policy: object | None,
    device: torch.device,
) -> tuple[FSDP2Application | None, FSDP2Application | None]:
    parallel = PostTrainingParallelContext.current()
    if recipe.distributed.backend != "fsdp2" or parallel.world_size == 1:
        return None, None
    plan = ParallelPlan.resolve(recipe.distributed, world_size=parallel.world_size)
    mesh = plan.build_device_mesh(device.type)
    dtype = _torch_dtype(recipe.runtime.param_dtype)
    reduce_dtype = _torch_dtype(recipe.runtime.reduce_dtype)
    policy_fsdp = apply_fsdp2(
        policy,
        plan=plan,
        mesh=mesh,
        param_dtype=dtype,
        reduce_dtype=reduce_dtype,
    )
    reference_fsdp = None
    if reference_policy is not None:
        reference_fsdp = apply_fsdp2_frozen_reference(
            reference_policy,
            plan=plan,
            mesh=mesh,
            param_dtype=dtype,
            reduce_dtype=reduce_dtype,
        )
    return policy_fsdp, reference_fsdp


def _materialize_ltx(
    recipe: PostTrainingRecipe,
    *,
    device: torch.device,
    checkpoint_overrides: Mapping[str, object],
    fused_adamw: bool | Literal["auto"],
) -> VideoFlowPolicyMaterialization:
    from worldfoundry.training.engine.ltx.flow_policy import (
        LTXFlowPredictionAdapter,
        bind_ltx_trajectory_adapters,
        validate_ltx_flow_policy_recipe,
    )
    from worldfoundry.training.engine.ltx.flow_policy_roles import (
        apply_ltx_policy_tuning,
        load_ltx_policy_adapter,
    )

    algorithm, data_plan = validate_ltx_flow_policy_recipe(recipe)
    dtype = _torch_dtype(recipe.runtime.param_dtype)
    role_keys = {"model", "gemma", "tokenizer", "text_encoder"}
    policy_overrides = {key: value for key, value in checkpoint_overrides.items() if key in role_keys}
    policy = load_ltx_policy_adapter(
        recipe,
        device=device,
        dtype=dtype,
        checkpoint_overrides=policy_overrides,
    )
    policy_tuning = apply_ltx_policy_tuning(recipe, policy)
    reference = None
    if algorithm.requires_reference_policy:
        assert algorithm.reference_checkpoint is not None
        reference_recipe = replace(
            recipe,
            model=replace(recipe.model, checkpoint=algorithm.reference_checkpoint),
        )
        reference_overrides = {
            key.removeprefix("reference-"): value
            for key, value in checkpoint_overrides.items()
            if key.startswith("reference-") and key.removeprefix("reference-") in role_keys
        }
        reference = load_ltx_policy_adapter(
            reference_recipe,
            device=device,
            dtype=dtype,
            checkpoint_overrides=reference_overrides,
        )
        reference.trainable_module.requires_grad_(False)
        reference.trainable_module.eval()
    policy_fsdp, reference_fsdp = _apply_distributed_wrapping(
        recipe,
        policy=policy,
        reference_policy=reference,
        device=device,
    )
    autocast = None if dtype is torch.float32 else dtype
    prediction = LTXFlowPredictionAdapter(
        policy,
        target_fps=data_plan.target_fps,
        autocast_dtype=autocast,
        checkpoint_identity=recipe.model.checkpoint,
    )
    reference_prediction = None
    if reference is not None:
        reference_prediction = LTXFlowPredictionAdapter(
            reference,
            target_fps=data_plan.target_fps,
            autocast_dtype=autocast,
            checkpoint_identity=algorithm.reference_checkpoint,
        )
    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=prediction,
        initial_policy_revision=recipe.model.checkpoint,
        reference_policy=reference_prediction,
        parallel_context=PostTrainingParallelContext.current(),
        fused_adamw=fused_adamw,
        rollout_forward_batch_size=data_plan.rollout_forward_batch_size,
        replay_microbatch_size=data_plan.replay_microbatch_size,
    )
    stack = bind_ltx_trajectory_adapters(
        stack,
        policy=prediction,
        reference_policy=reference_prediction,
        data_plan=data_plan,
        trajectory_dtype=algorithm.trajectory_dtype,
    )
    generation = data_plan.generation
    latent_shape = (
        policy.expected_latent_channels,
        (generation["num_frames"] - 1) // policy.temporal_compression + 1,
        generation["height"] // policy.spatial_compression,
        generation["width"] // policy.spatial_compression,
    )
    return VideoFlowPolicyMaterialization(
        policy=policy,
        reference_policy=reference,
        stack=stack,
        generation=generation,
        latent_shape=latent_shape,
        policy_tuning=policy_tuning,
        policy_fsdp=policy_fsdp,
        reference_fsdp=reference_fsdp,
    )


def _wan22_checkpoints(
    recipe: PostTrainingRecipe,
    checkpoint_overrides: Mapping[str, object],
):
    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
    from worldfoundry.training.engine.wan22.roles import wan22_role_checkpoints

    repository = "Wan-AI/Wan2.2-T2V-A14B" if recipe.model.checkpoint == "default" else recipe.model.checkpoint
    defaults = wan22_role_checkpoints(repository=repository)

    def selected(name: str, default: CheckpointSpec) -> CheckpointSpec:
        value = checkpoint_overrides.get(name)
        return default if value is None else CheckpointSpec(source=str(value))

    return type(defaults)(
        high_noise=selected("high-noise", defaults.high_noise),
        low_noise=selected("low-noise", defaults.low_noise),
    )


def _materialize_wan22(
    recipe: PostTrainingRecipe,
    *,
    device: torch.device,
    checkpoint_overrides: Mapping[str, object],
    fused_adamw: bool | Literal["auto"],
) -> VideoFlowPolicyMaterialization:
    from worldfoundry.training.engine.wan22.flow_policy import (
        validate_wan22_flow_policy_recipe,
    )
    from worldfoundry.training.engine.wan22.roles import load_wan22_role_adapter
    from worldfoundry.training.engine.wan22.tuning import apply_wan22_tuning

    algorithm, data_plan = validate_wan22_flow_policy_recipe(recipe)
    dtype = _torch_dtype(recipe.runtime.param_dtype)
    policy_checkpoint_overrides = {
        key: value for key, value in checkpoint_overrides.items() if key in {"high-noise", "low-noise"}
    }
    role_checkpoints = _wan22_checkpoints(recipe, policy_checkpoint_overrides)
    policy = load_wan22_role_adapter(
        checkpoints=role_checkpoints,
        device=device,
        dtype=dtype,
        boundary_ratio=data_plan.boundary_ratio,
        num_train_timesteps=algorithm.num_train_timesteps,
        gradient_checkpointing=recipe.runtime.activation_checkpoint == "full",
    )
    policy_tuning = apply_wan22_tuning(recipe, policy)
    reference = None
    if algorithm.requires_reference_policy:
        assert algorithm.reference_checkpoint is not None
        reference_recipe = replace(
            recipe,
            model=replace(recipe.model, checkpoint=algorithm.reference_checkpoint),
        )
        reference_overrides = {
            key.removeprefix("reference-"): value
            for key, value in checkpoint_overrides.items()
            if key in {"reference-high-noise", "reference-low-noise"}
        }
        reference = load_wan22_role_adapter(
            checkpoints=_wan22_checkpoints(reference_recipe, reference_overrides),
            device=device,
            dtype=dtype,
            boundary_ratio=data_plan.boundary_ratio,
            num_train_timesteps=algorithm.num_train_timesteps,
            gradient_checkpointing=False,
        )
        reference.trainable_module.requires_grad_(False)
        reference.trainable_module.eval()
    policy_fsdp, reference_fsdp = _apply_distributed_wrapping(
        recipe,
        policy=policy,
        reference_policy=reference,
        device=device,
    )
    autocast = None if dtype is torch.float32 else dtype
    prediction = NativeFlowPredictionAdapter(
        policy,
        autocast_dtype=autocast,
        checkpoint_identity=recipe.model.checkpoint,
    )
    reference_prediction = None
    if reference is not None:
        reference_prediction = NativeFlowPredictionAdapter(
            reference,
            autocast_dtype=autocast,
            checkpoint_identity=algorithm.reference_checkpoint,
        )
    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=prediction,
        initial_policy_revision=recipe.model.checkpoint,
        reference_policy=reference_prediction,
        parallel_context=PostTrainingParallelContext.current(),
        fused_adamw=fused_adamw,
        rollout_forward_batch_size=data_plan.rollout_forward_batch_size,
        replay_microbatch_size=data_plan.replay_microbatch_size,
    )
    generation = data_plan.generation
    latent_shape = (
        policy.expected_latent_channels,
        1 + (generation["num_frames"] - 1) // policy.temporal_compression,
        generation["height"] // policy.spatial_compression,
        generation["width"] // policy.spatial_compression,
    )
    return VideoFlowPolicyMaterialization(
        policy=policy,
        reference_policy=reference,
        stack=stack,
        generation=generation,
        latent_shape=latent_shape,
        policy_tuning=policy_tuning,
        policy_fsdp=policy_fsdp,
        reference_fsdp=reference_fsdp,
    )


def _materialize_hunyuan(
    recipe: PostTrainingRecipe,
    *,
    device: torch.device,
    checkpoint_overrides: Mapping[str, object],
    fused_adamw: bool | Literal["auto"],
) -> VideoFlowPolicyMaterialization:
    from worldfoundry.training.engine.hunyuan_video import (
        materialize_hunyuan_video_flow_policy,
    )

    role_overrides = {key: value for key, value in checkpoint_overrides.items() if key in {"policy", "reference"}}
    materialized = materialize_hunyuan_video_flow_policy(
        recipe,
        device=device,
        role_checkpoint_overrides=role_overrides,
        fused_adamw=fused_adamw,
    )
    return VideoFlowPolicyMaterialization(
        policy=materialized.policy,
        reference_policy=materialized.reference_policy,
        stack=materialized.stack,
        generation=materialized.data_plan.generation,
        latent_shape=materialized.policy.latent_shape(materialized.data_plan.generation),
        policy_tuning=materialized.policy_tuning,
        policy_fsdp=materialized.policy_fsdp,
        reference_fsdp=materialized.reference_fsdp,
    )


def materialize_video_flow_policy_roles(
    recipe: PostTrainingRecipe,
    *,
    device: str | torch.device,
    checkpoint_overrides: Mapping[str, object] | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> VideoFlowPolicyMaterialization:
    """Load and wrap trainable roles for every integrated video RL family."""

    if not isinstance(recipe.algorithm, FlowPolicyAlgorithmSpec):
        raise TypeError("video flow-policy roles require a flow-policy recipe")
    resolved_device = torch.device(device)
    overrides = dict(checkpoint_overrides or {})
    if recipe.model.recipe in _LTX_MODELS:
        return _materialize_ltx(
            recipe,
            device=resolved_device,
            checkpoint_overrides=overrides,
            fused_adamw=fused_adamw,
        )
    if recipe.model.recipe == _WAN22_MODEL:
        return _materialize_wan22(
            recipe,
            device=resolved_device,
            checkpoint_overrides=overrides,
            fused_adamw=fused_adamw,
        )
    if recipe.model.recipe in _HUNYUAN_MODELS:
        return _materialize_hunyuan(
            recipe,
            device=resolved_device,
            checkpoint_overrides=overrides,
            fused_adamw=fused_adamw,
        )
    raise ValueError(f"video flow-policy roles do not support {recipe.model.recipe!r}")


def _resolved_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def resolve_video_flow_policy_prompt_batch_size(
    recipe: PostTrainingRecipe,
    *,
    world_size: int,
) -> int:
    """Resolve an exact rank-local share of one logical global prompt batch."""

    if isinstance(world_size, bool) or int(world_size) <= 0:
        raise ValueError("world_size must be a positive integer")
    resolved_world_size = int(world_size)
    if "prompt_batch_size" in recipe.data.options:
        raise ValueError(
            "video flow-policy recipes use data.options.global_prompt_batch_size, not rank-local prompt_batch_size"
        )
    value = recipe.data.options.get("global_prompt_batch_size", 1)
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError("data.options.global_prompt_batch_size must be a positive integer")
    global_prompt_batch_size = int(value)
    local_prompt_batch_size, remainder = divmod(
        global_prompt_batch_size,
        resolved_world_size,
    )
    if remainder:
        raise ValueError(
            "data.options.global_prompt_batch_size must be divisible by the active world_size; "
            f"got {global_prompt_batch_size}/{resolved_world_size}"
        )
    if recipe.data.tail_policy == "uneven":
        raise ValueError("video flow-policy global prompt batches require data.tail_policy='drop' or 'pad'")
    return local_prompt_batch_size


def validate_video_flow_policy_prompt_population(
    recipe: PostTrainingRecipe,
    *,
    prompt_count: int,
    world_size: int,
) -> int:
    """Require enough distinct prompts to form one complete global rollout."""

    local_prompt_batch_size = resolve_video_flow_policy_prompt_batch_size(
        recipe,
        world_size=world_size,
    )
    global_prompt_batch_size = local_prompt_batch_size * int(world_size)
    if int(prompt_count) < global_prompt_batch_size:
        raise ValueError(
            "video flow-policy prompt data must contain at least one complete global prompt batch; "
            f"got {int(prompt_count)} prompts for global_prompt_batch_size={global_prompt_batch_size}"
        )
    return local_prompt_batch_size


def _load_conditioning_dataset(
    recipe: PostTrainingRecipe,
    *,
    root: Path,
) -> tuple[RolloutPromptDataset, RolloutConditioningDataset]:
    if recipe.data.cache is None:
        raise ValueError("video flow-policy training requires data.cache")
    prompts = RolloutPromptDataset.from_file(
        _resolved_path(root, recipe.data.manifest),
        split=recipe.data.split,
    )
    conditioning = RolloutConditioningDataset(
        prompts,
        _resolved_path(root, recipe.data.cache),
    )
    if conditioning.index.model_recipe != recipe.model.recipe:
        raise ValueError("rollout conditioning cache belongs to another model recipe")
    return prompts, conditioning


def _build_conditioned_source(
    recipe: PostTrainingRecipe,
    *,
    materialized: VideoFlowPolicyMaterialization,
    prompts: RolloutPromptDataset,
    conditioning: RolloutConditioningDataset,
    device: torch.device,
    rank: int,
    world_size: int,
    prompt_batch_size: int,
) -> tuple[object, torch.Generator, RolloutConditioningDataset]:
    geometries = {resolve_rollout_generation_geometry(record, materialized.generation) for record in prompts}
    expected = (
        materialized.generation["height"],
        materialized.generation["width"],
        materialized.generation["num_frames"],
    )
    if geometries != {expected}:
        raise ValueError("one video flow-policy run requires the configured fixed geometry")
    sampler = DeterministicDistributedSampler(
        conditioning,
        seed=recipe.data.shuffle_seed,
        shuffle=recipe.data.shuffle,
        rank=rank,
        world_size=world_size,
        tail_policy=recipe.data.tail_policy,
        local_batch_size=prompt_batch_size,
    )
    options = recipe.data.options
    source = build_stateful_dataloader(
        conditioning,
        sampler,
        batch_size=prompt_batch_size,
        collate_fn=collate_rollout_conditioned_prompts,
        num_workers=int(options.get("num_workers", 0)),
        worker_seed=recipe.data.shuffle_seed + rank,
        pin_memory=bool(options.get("pin_memory", device.type == "cuda")),
        drop_last=False,
        persistent_workers=bool(options.get("persistent_workers", False)),
        prefetch_factor=(None if options.get("prefetch_factor") is None else int(options["prefetch_factor"])),
        snapshot_every_n_steps=int(options.get("snapshot_every_n_steps", 1)),
    )
    generator = torch.Generator(device=device)
    generator.manual_seed((recipe.data.shuffle_seed + rank) % (2**63 - 1))
    return source, generator, conditioning


def _build_native_decoder(
    recipe: PostTrainingRecipe,
    *,
    device: torch.device,
    checkpoint_overrides: Mapping[str, object],
    joint_av: bool = False,
) -> object:
    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentBuildContext,
        ComponentKey,
        ComponentKind,
        ComponentSpec,
    )
    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
    from worldfoundry.base_models.diffusion_model.optimizations import (
        AttentionBackend,
        RuntimePolicy,
    )
    from worldfoundry.base_models.diffusion_model.recipes.registry import (
        default_native_diffusion_registry,
    )

    dtype = _torch_dtype(recipe.runtime.param_dtype)
    policy = RuntimePolicy(
        device=device,
        dtype=dtype,
        attention=AttentionBackend.TORCH,
    )
    if recipe.model.recipe == _WAN22_MODEL:
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import (
            build_wan_video_decoder,
        )

        value = checkpoint_overrides.get("vae")
        checkpoint = (
            CheckpointSpec(source=str(value))
            if value is not None
            else CheckpointSpec(
                repo_id=("Wan-AI/Wan2.2-T2V-A14B" if recipe.model.checkpoint == "default" else recipe.model.checkpoint),
                revision="main",
                files=("Wan2.1_VAE.pth",),
            )
        )
        return build_wan_video_decoder(
            ComponentBuildContext(
                model_id=_WAN22_MODEL,
                key=ComponentKey(ComponentKind.LATENT_ENCODER, "codec"),
                purpose=BuildPurpose.REWARD,
                policy=policy,
                checkpoints={"weights": checkpoint},
                component_options={},
            )
        )

    native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
    assembler = NativeDiffusionAssembler()
    overrides = {key: value for key, value in checkpoint_overrides.items() if key in native_recipe.checkpoints}
    if recipe.model.checkpoint != "default":
        checkpoint_key = "model" if recipe.model.recipe in _LTX_MODELS else "transformer"
        overrides.setdefault(checkpoint_key, recipe.model.checkpoint)
    if recipe.model.recipe in _LTX_MODELS:
        from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx import (
            build_ltx_tensor_video_codec,
        )
        from worldfoundry.training.engine.ltx.flow_policy_roles import (
            ltx_policy_default_checkpoint,
        )

        overrides.setdefault("model", ltx_policy_default_checkpoint(recipe.model.recipe))

        if joint_av:
            decoder_key = ComponentKey(ComponentKind.DECODER)
            components = assembler.build_components(
                native_recipe,
                purpose=BuildPurpose.REWARD,
                policy=policy,
                checkpoint_overrides=overrides,
                component_keys=(decoder_key,),
            )
            return components[decoder_key]

        codec_key = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
        native_recipe = replace(
            native_recipe,
            components=(
                *native_recipe.components,
                ComponentSpec(codec_key, build_ltx_tensor_video_codec, {"weights": "model"}),
            ),
        )
    else:
        codec_key = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
    components = assembler.build_components(
        native_recipe,
        purpose=BuildPurpose.REWARD,
        policy=policy,
        checkpoint_overrides=overrides,
        component_keys=(codec_key,),
    )
    return components[codec_key]


def materialize_video_flow_policy_training_run(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    reward_device: str | torch.device | None = None,
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    checkpoint_overrides: Mapping[str, object] | None = None,
    reward_url: str | None = None,
    reward_evaluator: RewardEvaluator | None = None,
    reward_attention_implementation: str = "sdpa",
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
    rollout_policy_factory: Callable[..., object] | None = None,
    rollout_policy_factory_kwargs: Mapping[str, object] | None = None,
    ray_runtime_factory: Callable[[RayPostTrainingRuntimeConfig], RayPostTrainingRuntime] = RayPostTrainingRuntime,
) -> NativeFlowPolicyTrainingRun:
    """Materialize a complete LTX, Wan2.2, or HunyuanVideo policy run."""

    if not isinstance(recipe.algorithm, FlowPolicyAlgorithmSpec):
        raise TypeError("video flow-policy training requires a flow-policy recipe")
    if reward_url is not None and reward_evaluator is not None:
        raise ValueError("provide reward_url or reward_evaluator, not both")
    remote_reward = isinstance(recipe.algorithm.reward_model, RemoteRewardSpec)
    if remote_reward and reward_url is None and reward_evaluator is None:
        raise ValueError("remote reward recipes require reward_url or an injected reward_evaluator")
    joint_av_reward = remote_reward and recipe.model.recipe == "ltx-2.3-i2v"
    ray_runtime_config = _ray_video_runtime_config(recipe) if isinstance(recipe.rollout, RayRolloutSpec) else None
    root = Path(base_dir).expanduser().resolve()
    destination = _resolved_path(root, output_dir or recipe.run.output_dir)
    resolved_device = torch.device(device)
    distributed_context = None
    if recipe.distributed.backend == "fsdp2":
        if resolved_device.type != "cuda":
            raise ValueError("video flow-policy FSDP2 execution requires CUDA")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device
    elif recipe.distributed.backend != "single":
        raise ValueError("video flow-policy training supports single-device or FSDP2 execution")
    resolved_reward_device = torch.device(resolved_device if reward_device is None else reward_device)
    overrides = dict(checkpoint_overrides or {})
    seed = recipe.data.shuffle_seed if initialization_seed is None else int(initialization_seed)
    set_seed_everywhere(seed)
    closeables: list[object] = []
    try:
        rank = 0 if distributed_context is None else distributed_context.rank
        world_size = 1 if distributed_context is None else distributed_context.world_size
        prompts, conditioning = _load_conditioning_dataset(recipe, root=root)
        prompt_batch_size = validate_video_flow_policy_prompt_population(
            recipe,
            prompt_count=len(conditioning),
            world_size=world_size,
        )
        create_run_directory(destination, distributed_context)
        materialized = materialize_video_flow_policy_roles(
            recipe,
            device=resolved_device,
            checkpoint_overrides=overrides,
            fused_adamw=fused_adamw,
        )
        if isinstance(recipe.rollout, RayRolloutSpec):
            assert ray_runtime_config is not None
            runtime = ray_runtime_factory(ray_runtime_config)
            if not isinstance(runtime, RayPostTrainingRuntime):
                raise TypeError("ray_runtime_factory must return RayPostTrainingRuntime")
            closeables.append(runtime)
            policy_factory = rollout_policy_factory
            policy_factory_kwargs = dict(rollout_policy_factory_kwargs or {})
            if policy_factory is None:
                from worldfoundry.training.engine.video_rollout import (
                    materialize_video_ray_rollout_policy,
                )

                accelerator = recipe.rollout.pool.accelerator_resource.upper()
                if accelerator not in {"CPU", "GPU"}:
                    raise ValueError("native video rollout policy construction supports CPU or GPU Ray workers")
                policy_factory = materialize_video_ray_rollout_policy
                policy_factory_kwargs = {
                    "recipe": recipe,
                    "checkpoint_overrides": overrides,
                    "device_type": "cuda" if accelerator == "GPU" else "cpu",
                }
            sampler_factory = None
            sampler_factory_kwargs: dict[str, object] = {}
            if recipe.model.recipe in _LTX_MODELS:
                from worldfoundry.training.engine.ltx.trajectory import (
                    build_ltx_ray_trajectory_sampler,
                )

                sampler_factory = build_ltx_ray_trajectory_sampler
                sampler_factory_kwargs = {
                    "audio_joint_sde": bool(getattr(materialized.stack.sampler, "audio_joint_sde")),
                    "init_same_noise": bool(getattr(materialized.stack.sampler, "init_same_noise")),
                }
            materialized = replace(
                materialized,
                stack=attach_ray_flow_policy_rollout(
                    materialized.stack,
                    runtime,
                    rollout_policy_factory=policy_factory,
                    rollout_policy_factory_kwargs=policy_factory_kwargs,
                    rollout_sampler_factory=sampler_factory,
                    rollout_sampler_factory_kwargs=sampler_factory_kwargs,
                    source_module=materialized.policy_module,
                    weight_kind=recipe.rollout.weight_kind,
                ),
            )
        source, generator, conditioning = _build_conditioned_source(
            recipe,
            materialized=materialized,
            prompts=prompts,
            conditioning=conditioning,
            device=resolved_device,
            rank=rank,
            world_size=world_size,
            prompt_batch_size=prompt_batch_size,
        )
        rollout_loader = materialized.build_rollout_loader(
            source,
            generator=generator,
            device=resolved_device,
            group_namespace=(None if world_size == 1 else f"rank-{rank:08d}"),
        )
        decoder = _build_native_decoder(
            recipe,
            device=resolved_reward_device,
            checkpoint_overrides=overrides,
            joint_av=joint_av_reward,
        )
        if reward_url is not None:
            http_evaluator = HTTPRewardEvaluator(reward_url)
            if joint_av_reward:
                from worldfoundry.training.engine.ltx.rewards import (
                    LTXAVTerminalRewardAdapter,
                )

                reward_adapter = LTXAVTerminalRewardAdapter(
                    decoder,
                    http_evaluator,
                    reward_ids=recipe.algorithm.reward_model.reward_ids,
                    frame_rate=float(recipe.data.options.get("target_fps", 24.0)),
                    evaluator_identity={"transport": "http", "base_url": http_evaluator.base_url},
                )
            else:
                reward_adapter = HTTPTerminalRewardAdapter(
                    decoder,
                    http_evaluator,
                    reward_ids=recipe.algorithm.reward_model.reward_ids,
                )
            closeables.append(http_evaluator)
        else:
            if reward_evaluator is None:
                evaluator = build_videoalign_reward_evaluator(
                    recipe.algorithm.reward_model,
                    device=resolved_reward_device,
                    attention_implementation=reward_attention_implementation,
                )
            else:
                evaluator = reward_evaluator
            evaluator_identity = dict(getattr(evaluator, "identity", {"type": "reward-evaluator"}))
            if joint_av_reward:
                from worldfoundry.training.engine.ltx.rewards import (
                    LTXAVTerminalRewardAdapter,
                )

                reward_adapter = LTXAVTerminalRewardAdapter(
                    decoder,
                    evaluator,
                    reward_ids=recipe.algorithm.reward_model.reward_ids,
                    frame_rate=float(recipe.data.options.get("target_fps", 24.0)),
                    evaluator_identity=evaluator_identity,
                )
            else:
                reward_adapter = DecodedTerminalRewardAdapter(
                    decoder,
                    evaluator,
                    reward_ids=recipe.algorithm.reward_model.reward_ids,
                    evaluator_identity=evaluator_identity,
                )
        return build_native_flow_policy_training_run(
            recipe,
            stack=materialized.stack,
            dataloader=rollout_loader,
            reward_adapter=reward_adapter,
            policy_module=materialized.policy_module,
            policy_tuning=materialized.policy_tuning,
            objective_generator=generator,
            output_dir=destination,
            resume_identity={
                "recipe": recipe.to_dict(),
                "conditioning": conditioning.index.to_dict(),
                "rank_count": world_size,
                "initialization_seed": seed,
            },
            resume_checkpoint=resume_checkpoint,
            distributed_context=distributed_context,
            closeables=tuple(closeables),
        )
    except Exception:
        for resource in reversed(closeables):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        if distributed_context is not None:
            distributed_context.close()
        raise


__all__ = [
    "VideoFlowPolicyMaterialization",
    "materialize_video_flow_policy_roles",
    "materialize_video_flow_policy_training_run",
    "resolve_video_flow_policy_prompt_batch_size",
    "validate_video_flow_policy_prompt_population",
]
