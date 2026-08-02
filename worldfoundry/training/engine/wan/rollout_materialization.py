"""Shared Wan prompt-rollout assets, stateful loading, and terminal rewards."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from worldfoundry.training.data.loader import build_stateful_dataloader
from worldfoundry.training.data.rollout_cache import (
    RolloutConditioningDataset,
    collate_rollout_conditioned_prompts,
    resolve_rollout_generation_geometry,
)
from worldfoundry.training.data.rollout_manifest import RolloutPromptDataset
from worldfoundry.training.data.sampler import DeterministicDistributedSampler
from worldfoundry.training.data.shared_conditioning import (
    SharedConditioningSample,
    SharedConditioningStore,
)
from worldfoundry.training.data.wan.contracts import (
    wan_cache_contract_digest,
    wan_checkpoint_asset_digest,
)
from worldfoundry.training.distributed.parallel import (
    DistributedTrainingContext,
    ParallelPlan,
)
from worldfoundry.training.models.wan import WanTrainAdapter
from worldfoundry.training.post_training.rewards.videoalign import (
    VideoAlignRewardEvaluator,
    build_videoalign_reward_evaluator,
)
from worldfoundry.training.post_training.rl.trajectory_rewards import (
    DecodedTerminalRewardAdapter,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.post_training.rewards.videoalign import (
    VideoAlignRewardSpec,
)

from ..artifacts import create_run_directory
from .flow_policy_recipe import (
    WanFlowPolicyDataPlan,
    audit_component_overrides,
    audit_conditioning_cache,
    audit_unconditional_conditioning,
    validate_generation_geometry,
)
from .roles import seed_initialization, torch_dtype


@dataclass(frozen=True, slots=True)
class WanRolloutAssets:
    """Resolved immutable assets shared by Wan online policy algorithms."""

    root: Path
    output_dir: Path
    cache_path: Path
    manifest_path: Path
    device: torch.device
    reward_device: torch.device
    distributed_context: DistributedTrainingContext | None
    parallel_plan: ParallelPlan
    world_size: int
    rank: int
    prompts: RolloutPromptDataset
    conditioning: RolloutConditioningDataset
    generation_geometry: tuple[int, int, int]
    assembler: object
    native_recipe: object
    component_overrides: Mapping[str, object]
    resolved_component_checkpoints: Mapping[str, object]
    model_contract_digest: str
    conditioner_digest: str
    tokenizer_digest: str
    dtype: torch.dtype
    base_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_overrides",
            MappingProxyType(dict(self.component_overrides)),
        )
        object.__setattr__(
            self,
            "resolved_component_checkpoints",
            MappingProxyType(dict(self.resolved_component_checkpoints)),
        )


@dataclass(frozen=True, slots=True)
class WanRolloutSource:
    """Stateful conditioned prompt source plus rank-local rollout RNG."""

    loader: object
    generator: torch.Generator
    unconditional: SharedConditioningSample | None


def _audit_distributed_rollout_collectives(
    *,
    world_size: int,
    tail_policy: str,
    rollout_forward_batch_size: int | None,
    replay_microbatch_size: int | None,
) -> None:
    """Reject rank-uneven chunking that would issue mismatched FSDP collectives."""

    if isinstance(world_size, bool) or int(world_size) <= 0:
        raise ValueError("rollout world_size must be positive")
    policy = str(tail_policy).strip().lower().replace("_", "-")
    if policy not in {"drop", "pad", "uneven"}:
        raise ValueError("rollout tail_policy must be drop, pad, or uneven")
    if int(world_size) == 1 or policy != "uneven":
        return
    chunked_stages = tuple(
        name
        for name, value in (
            ("rollout forward", rollout_forward_batch_size),
            ("replay backward", replay_microbatch_size),
        )
        if value is not None
    )
    if chunked_stages:
        raise ValueError(
            "distributed rollout/replay microbatching cannot use data.tail_policy='uneven' "
            "because ranks may execute different model collective counts; "
            f"chunked stages: {', '.join(chunked_stages)}"
        )


def _resolved_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def prepare_wan_rollout_assets(
    recipe: PostTrainingRecipe,
    data_plan: WanFlowPolicyDataPlan,
    *,
    frame_factor: int,
    base_dir: str | Path,
    device: str | torch.device,
    reward_device: str | torch.device | None,
    output_dir: str | Path | None,
    audited_component_overrides: Mapping[str, object] | None,
    force_torch_attention: bool,
    initialization_seed: int | None,
) -> WanRolloutAssets:
    """Resolve data, model-family components, devices, and deterministic seeds."""

    if not isinstance(recipe, PostTrainingRecipe):
        raise TypeError("recipe must be PostTrainingRecipe")
    if not isinstance(data_plan, WanFlowPolicyDataPlan):
        raise TypeError("data_plan must be WanFlowPolicyDataPlan")
    root = Path(base_dir).expanduser().resolve()
    cache_path = _resolved_path(root, recipe.data.cache or "")
    manifest_path = _resolved_path(root, recipe.data.manifest)
    destination = _resolved_path(root, output_dir or recipe.run.output_dir)
    if not cache_path.is_dir():
        raise FileNotFoundError(f"rollout conditioning cache does not exist: {cache_path}")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    distributed_context: DistributedTrainingContext | None = None
    if recipe.distributed.backend == "fsdp2":
        if resolved_device.type != "cuda":
            raise ValueError("Wan rollout FSDP2 materialization requires CUDA")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device
    resolved_reward_device = torch.device(resolved_device if reward_device is None else reward_device)
    if distributed_context is not None and resolved_reward_device.type == "cuda":
        if resolved_reward_device.index is None:
            resolved_reward_device = resolved_device
        elif resolved_reward_device.index != resolved_device.index:
            distributed_context.close()
            raise ValueError("distributed reward_device must be the rank-local CUDA device")
    if resolved_reward_device.type == "cuda" and not torch.cuda.is_available():
        if distributed_context is not None:
            distributed_context.close()
        raise RuntimeError("CUDA reward execution was requested but is not available")

    try:
        world_size = 1 if distributed_context is None else distributed_context.world_size
        rank = 0 if distributed_context is None else distributed_context.rank
        parallel_plan = ParallelPlan.resolve(recipe.distributed, world_size=world_size)
        create_run_directory(destination, distributed_context)
        if force_torch_attention:
            for name in (
                "WORLDFOUNDRY_ATTENTION_IMPLEMENTATION",
                "WORLDFOUNDRY_ATTENTION_BACKEND",
            ):
                configured = os.environ.get(name)
                if configured not in {None, "torch"}:
                    raise ValueError(f"correctness-first Wan rollout requires {name}=torch; got {configured!r}")
                os.environ[name] = "torch"

        prompts = RolloutPromptDataset.from_file(
            manifest_path,
            split=recipe.data.split,
        )
        conditioning = RolloutConditioningDataset(prompts, cache_path)
        geometries = {resolve_rollout_generation_geometry(record, data_plan.generation) for record in prompts}
        if len(geometries) != 1:
            raise ValueError("one Wan rollout run requires one fixed generation geometry")
        generation_geometry = next(iter(geometries))
        validate_generation_geometry(
            generation_geometry,
            frame_factor=frame_factor,
        )

        from worldfoundry.base_models.diffusion_model.assembly import (
            NativeDiffusionAssembler,
        )
        from worldfoundry.base_models.diffusion_model.recipes.registry import (
            default_native_diffusion_registry,
        )

        native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
        assembler = NativeDiffusionAssembler()
        component_overrides = audit_component_overrides(audited_component_overrides)
        resolved_component_checkpoints = assembler.resolve_checkpoints(
            native_recipe,
            component_overrides,
        )
        model_contract_digest = wan_cache_contract_digest(recipe.model.recipe)
        conditioner_digest = wan_checkpoint_asset_digest(resolved_component_checkpoints["text-encoder"])
        tokenizer_digest = wan_checkpoint_asset_digest(resolved_component_checkpoints["tokenizer"])
        if isinstance(initialization_seed, bool):
            raise TypeError("initialization_seed must be an integer, not bool")
        base_seed = int(recipe.data.shuffle_seed if initialization_seed is None else initialization_seed)
        seed_initialization(base_seed)
        return WanRolloutAssets(
            root=root,
            output_dir=destination,
            cache_path=cache_path,
            manifest_path=manifest_path,
            device=resolved_device,
            reward_device=resolved_reward_device,
            distributed_context=distributed_context,
            parallel_plan=parallel_plan,
            world_size=world_size,
            rank=rank,
            prompts=prompts,
            conditioning=conditioning,
            generation_geometry=generation_geometry,
            assembler=assembler,
            native_recipe=native_recipe,
            component_overrides=component_overrides,
            resolved_component_checkpoints=resolved_component_checkpoints,
            model_contract_digest=model_contract_digest,
            conditioner_digest=conditioner_digest,
            tokenizer_digest=tokenizer_digest,
            dtype=torch_dtype(recipe.runtime.param_dtype),
            base_seed=base_seed,
        )
    except Exception:
        if distributed_context is not None:
            distributed_context.close()
        raise


def build_wan_rollout_source(
    assets: WanRolloutAssets,
    recipe: PostTrainingRecipe,
    data_plan: WanFlowPolicyDataPlan,
    policy: WanTrainAdapter,
    *,
    guidance_scale: float,
    replay_microbatch_size: int | None = None,
    requires_unconditional: bool = False,
) -> WanRolloutSource:
    """Build a rank-owned, exact-resume conditioned prompt source."""

    audit_conditioning_cache(
        assets.conditioning,
        policy,
        model_recipe=recipe.model.recipe,
        model_recipe_digest=assets.model_contract_digest,
        conditioner_digest=assets.conditioner_digest,
        tokenizer_digest=assets.tokenizer_digest,
    )
    if not isinstance(requires_unconditional, bool):
        raise TypeError("requires_unconditional must be a bool")
    unconditional = None
    if float(guidance_scale) > 1 or requires_unconditional:
        try:
            unconditional = SharedConditioningStore(assets.cache_path).read("unconditional")
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Wan CFG requires the immutable empty-prompt conditioning object; "
                "rebuild this rollout cache with the native training cache command"
            ) from error
        audit_unconditional_conditioning(
            unconditional,
            policy,
            model_recipe_digest=assets.model_contract_digest,
            conditioner_digest=assets.conditioner_digest,
            tokenizer_digest=assets.tokenizer_digest,
        )
    _audit_distributed_rollout_collectives(
        world_size=assets.world_size,
        tail_policy=recipe.data.tail_policy,
        rollout_forward_batch_size=data_plan.rollout_forward_batch_size,
        replay_microbatch_size=replay_microbatch_size,
    )
    sampler = DeterministicDistributedSampler(
        assets.conditioning,
        dataset_digest=assets.conditioning.dataset_digest,
        seed=recipe.data.shuffle_seed,
        shuffle=recipe.data.shuffle,
        rank=assets.rank,
        world_size=assets.world_size,
        tail_policy=recipe.data.tail_policy,
    )
    if len(sampler) == 0:
        raise ValueError(
            "Wan rollout requires at least one prompt on every active rank; "
            "use data.tail_policy='pad' or launch fewer ranks"
        )
    loader = build_stateful_dataloader(
        assets.conditioning,
        sampler,
        batch_size=data_plan.prompt_batch_size,
        collate_fn=collate_rollout_conditioned_prompts,
        num_workers=data_plan.num_workers,
        worker_seed=recipe.data.shuffle_seed + assets.rank,
        pin_memory=(assets.device.type == "cuda" if data_plan.pin_memory is None else data_plan.pin_memory),
        drop_last=False,
        persistent_workers=data_plan.persistent_workers,
        prefetch_factor=data_plan.prefetch_factor,
        multiprocessing_context=data_plan.multiprocessing_context,
        snapshot_every_n_steps=data_plan.snapshot_every_n_steps,
    )
    generator = torch.Generator(device=assets.device)
    generator.manual_seed((assets.base_seed + assets.rank) % (2**63 - 1))
    return WanRolloutSource(
        loader=loader,
        generator=generator,
        unconditional=unconditional,
    )


def build_wan_terminal_reward_adapter(
    assets: WanRolloutAssets,
    data_plan: WanFlowPolicyDataPlan,
    reward_spec: VideoAlignRewardSpec,
    *,
    attention_implementation: str,
) -> DecodedTerminalRewardAdapter:
    """Materialize the Wan decoder and typed terminal VideoAlign evaluator."""

    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentKey,
        ComponentKind,
    )
    from worldfoundry.base_models.diffusion_model.optimizations import (
        AttentionBackend,
        RuntimePolicy,
    )

    codec_key = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
    codec_components = assets.assembler.build_components(
        assets.native_recipe,
        purpose=BuildPurpose.REWARD,
        policy=RuntimePolicy(
            device=assets.reward_device,
            dtype=assets.dtype,
            attention=AttentionBackend.TORCH,
        ),
        checkpoint_overrides=assets.component_overrides,
        component_options={codec_key: data_plan.codec_options},
        component_keys=(codec_key,),
    )
    evaluator = build_videoalign_reward_evaluator(
        reward_spec,
        device=assets.reward_device,
        attention_implementation=attention_implementation,
    )
    if not isinstance(evaluator, VideoAlignRewardEvaluator):
        raise TypeError("VideoAlign builder returned an incompatible evaluator")
    return DecodedTerminalRewardAdapter(
        codec_components[codec_key],
        evaluator,
        reward_ids=reward_spec.reward_ids,
        evaluator_identity=evaluator.identity,
    )


def wan_rollout_latent_shape(
    policy: WanTrainAdapter,
    geometry: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Resolve the exact [C,T,H,W] latent generation shape."""

    height, width, frames = geometry
    return (
        policy.expected_latent_channels,
        1 + (frames - 1) // policy.temporal_compression,
        height // policy.spatial_compression,
        width // policy.spatial_compression,
    )


__all__ = [
    "WanRolloutAssets",
    "WanRolloutSource",
    "build_wan_rollout_source",
    "build_wan_terminal_reward_adapter",
    "prepare_wan_rollout_assets",
    "wan_rollout_latent_shape",
]
