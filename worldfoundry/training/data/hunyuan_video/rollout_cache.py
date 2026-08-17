"""HunyuanVideo prompt-conditioning cache materialization for RL rollouts."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path

import torch

from worldfoundry.training.models.hunyuan_video import HUNYUAN_VIDEO_MODEL_RECIPES

from ..rollout_cache import (
    RolloutConditioningPreparationResult,
    RolloutConditioningStore,
    prepare_rollout_conditioning_cache,
)
from ..rollout_manifest import RolloutPromptDataset
from ..video_precompute import checkpoint_spec_identity
from .encoding import HunyuanVideoTextFeatureEncoder


def materialize_hunyuan_video_rollout_conditioning_cache(
    recipe: object,
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    device: str | torch.device = "cuda",
    checkpoint_overrides: Mapping[str, object] | None = None,
) -> RolloutConditioningPreparationResult:
    """Encode one prompt manifest with the registered native conditioner."""

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentKey,
        ComponentKind,
    )
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
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
        raise TypeError("HunyuanVideo rollout conditioning requires a rollout-based post-training recipe")
    if recipe.model.recipe not in HUNYUAN_VIDEO_MODEL_RECIPES:
        raise ValueError(f"unsupported HunyuanVideo rollout model: {recipe.model.recipe!r}")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    store = RolloutConditioningStore(cache_dir)
    if store.index_path.exists():
        raise FileExistsError("HunyuanVideo rollout conditioning index already exists")
    prompts = RolloutPromptDataset.from_file(manifest_path, split=recipe.data.split)
    generation = recipe.data.options.get("generation")
    if not isinstance(generation, Mapping):
        raise TypeError("rollout data.options.generation must be a mapping")

    native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
    assembler = NativeDiffusionAssembler()
    overrides = dict(checkpoint_overrides or {})
    checkpoints = assembler.resolve_checkpoints(native_recipe, overrides)
    resources = checkpoint_spec_identity(checkpoints["resources"])
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
    encoder = HunyuanVideoTextFeatureEncoder(
        components[conditioner_key],
        model_recipe=recipe.model.recipe,
        device=resolved_device,
        dtype=dtype,
    )
    try:
        return prepare_rollout_conditioning_cache(
            prompts,
            cache_root=cache_dir,
            encoder=encoder,
            model_recipe=recipe.model.recipe,
            conditioner={"resources": resources},
            tokenizer=resources,
            generation_defaults={str(key): value for key, value in generation.items()},
            tensor_layouts=encoder.tensor_layouts,
        )
    finally:
        del encoder, components
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()


__all__ = ["materialize_hunyuan_video_rollout_conditioning_cache"]
