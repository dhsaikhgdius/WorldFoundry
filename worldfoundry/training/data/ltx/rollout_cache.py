"""LTX-2 prompt-conditioning cache materialization for policy rollouts."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path

import torch

from worldfoundry.training.engine.ltx.flow_policy import LTX_POLICY_MODELS
from worldfoundry.training.engine.ltx.flow_policy_roles import (
    ltx_policy_default_checkpoint,
)

from ..rollout_cache import (
    RolloutConditioningPreparationResult,
    RolloutConditioningStore,
    prepare_rollout_conditioning_cache,
)
from ..rollout_manifest import RolloutPromptDataset
from ..video_precompute import checkpoint_spec_identity
from .encoding import LTXTextFeatureEncoder

LTX_ROLLOUT_CONDITIONING_LAYOUTS = {
    "video_context": "sequence-features",
    "audio_context": "sequence-features",
    "context_mask": "sequence",
}


def materialize_ltx_rollout_conditioning_cache(
    recipe: object,
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    device: str | torch.device = "cuda",
    checkpoint_overrides: Mapping[str, object] | None = None,
) -> RolloutConditioningPreparationResult:
    """Encode one prompt manifest with LTX's Gemma and native connector."""

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
        raise TypeError("LTX rollout conditioning requires a rollout-based post-training recipe")
    if recipe.model.recipe not in LTX_POLICY_MODELS:
        raise ValueError(f"unsupported LTX rollout model: {recipe.model.recipe!r}")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    store = RolloutConditioningStore(cache_dir)
    if store.index_path.exists():
        raise FileExistsError("LTX rollout conditioning index already exists")
    prompts = RolloutPromptDataset.from_file(manifest_path, split=recipe.data.split)
    generation = recipe.data.options.get("generation")
    if not isinstance(generation, Mapping):
        raise TypeError("rollout data.options.generation must be a mapping")
    target_fps = float(recipe.data.options.get("target_fps", 24.0))
    if target_fps <= 0:
        raise ValueError("LTX rollout target_fps must be positive")

    native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
    assembler = NativeDiffusionAssembler()
    overrides = dict(checkpoint_overrides or {})
    if recipe.model.checkpoint == "default":
        overrides.setdefault("model", ltx_policy_default_checkpoint(recipe.model.recipe))
    elif "model" not in overrides:
        raise ValueError(
            "a non-default LTX policy checkpoint requires checkpoint_overrides['model'] for cache creation"
        )
    checkpoints = assembler.resolve_checkpoints(native_recipe, overrides)
    conditioner_identity = {
        "gemma": checkpoint_spec_identity(checkpoints["gemma"]),
        "projection": checkpoint_spec_identity(checkpoints["model"]),
    }
    tokenizer_identity = checkpoint_spec_identity(checkpoints["tokenizer"])
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
    encoder = LTXTextFeatureEncoder(
        components[conditioner_key],
        device=resolved_device,
        dtype=dtype,
    )
    try:
        return prepare_rollout_conditioning_cache(
            prompts,
            cache_root=cache_dir,
            encoder=encoder,
            model_recipe=recipe.model.recipe,
            conditioner=conditioner_identity,
            tokenizer=tokenizer_identity,
            generation_defaults={str(key): value for key, value in generation.items()},
            encoder_options={"fps": target_fps},
            tensor_layouts=LTX_ROLLOUT_CONDITIONING_LAYOUTS,
        )
    finally:
        del encoder, components
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()


__all__ = [
    "LTX_ROLLOUT_CONDITIONING_LAYOUTS",
    "materialize_ltx_rollout_conditioning_cache",
]
