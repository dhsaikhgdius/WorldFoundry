"""Wan prompt-conditioning cache materialization for RL rollouts."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path

import torch

from ..rollout_cache import (
    RolloutConditioningPreparationResult,
    RolloutConditioningStore,
    prepare_rollout_conditioning_cache,
    resolve_rollout_generation_geometry,
)
from ..rollout_manifest import RolloutPromptDataset
from .artifacts import write_wan_unconditional_conditioning
from .contracts import wan_checkpoint_asset_identity
from .encoding import WanTextFeatureEncoder


def materialize_wan_rollout_conditioning_cache(
    recipe: object,
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    device: str | torch.device = "cuda",
    checkpoint_overrides: Mapping[str, object] | None = None,
) -> RolloutConditioningPreparationResult:
    """Encode safe prompt-only RL data with the native Wan conditioner."""

    from worldfoundry.base_models.diffusion_model.assembly import (
        NativeDiffusionAssembler,
    )
    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentKey,
        ComponentKind,
    )
    from worldfoundry.base_models.diffusion_model.optimizations import (
        AttentionBackend,
        RuntimePolicy,
    )
    from worldfoundry.base_models.diffusion_model.recipes.registry import (
        default_native_diffusion_registry,
    )
    from worldfoundry.training.recipes.post_training.algorithms.diffusion_nft import (
        DiffusionNFTAlgorithmSpec,
    )
    from worldfoundry.training.recipes.post_training.algorithms.flow_policy import (
        FlowPolicyAlgorithmSpec,
    )
    from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

    if not isinstance(recipe, PostTrainingRecipe) or not isinstance(
        recipe.algorithm,
        (DiffusionNFTAlgorithmSpec, FlowPolicyAlgorithmSpec),
    ):
        raise TypeError("Wan rollout conditioning requires a rollout-based post-training recipe")
    if recipe.model.recipe != "wan2.1-t2v-1.3b":
        raise ValueError("Wan rollout conditioning currently requires wan2.1-t2v-1.3b")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    store = RolloutConditioningStore(cache_dir)
    if store.index_path.exists():
        raise FileExistsError("Wan rollout conditioning index already exists; materialization will not overwrite it")
    prompts = RolloutPromptDataset.from_file(
        manifest_path,
        split=recipe.data.split,
    )
    raw_generation = recipe.data.options.get("generation")
    if not isinstance(raw_generation, Mapping):
        raise TypeError("rollout data.options.generation must be a mapping")
    generation_defaults = {str(key): value for key, value in raw_generation.items()}

    native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
    assembler = NativeDiffusionAssembler()
    overrides = dict(checkpoint_overrides or {})
    resolved_checkpoints = assembler.resolve_checkpoints(native_recipe, overrides)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[recipe.runtime.param_dtype]
    conditioner_key = ComponentKey(ComponentKind.CONDITIONER)
    components = assembler.build_components(
        native_recipe,
        purpose=BuildPurpose.TRAINING,
        policy=RuntimePolicy(
            device=resolved_device,
            dtype=dtype,
            attention=AttentionBackend.TORCH,
        ),
        checkpoint_overrides=overrides,
        component_keys=(conditioner_key,),
    )
    encoder = WanTextFeatureEncoder(components[conditioner_key])
    try:
        first_record = prompts[0]
        height, width, frames = resolve_rollout_generation_geometry(
            first_record,
            generation_defaults,
        )
        conditioner = wan_checkpoint_asset_identity(resolved_checkpoints["text-encoder"])
        tokenizer = wan_checkpoint_asset_identity(resolved_checkpoints["tokenizer"])
        unconditional = write_wan_unconditional_conditioning(
            store=store,
            context=encoder.encode(
                sample_id="shared-unconditional",
                prompt="",
                frames=frames,
                height=height,
                width=width,
            ),
            model_recipe=recipe.model.recipe,
            conditioner=conditioner,
            tokenizer=tokenizer,
        )
        prepared = prepare_rollout_conditioning_cache(
            prompts,
            cache_root=cache_dir,
            encoder=encoder,
            model_recipe=recipe.model.recipe,
            conditioner=conditioner,
            tokenizer=tokenizer,
            generation_defaults=generation_defaults,
        )
        return RolloutConditioningPreparationResult(
            index=prepared.index,
            entries=prepared.entries,
            unconditional_conditioning=unconditional,
        )
    finally:
        del encoder, components
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()


__all__ = ["materialize_wan_rollout_conditioning_cache"]
