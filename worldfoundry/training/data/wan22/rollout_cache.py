"""Wan2.2 A14B prompt-conditioning cache materialization."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path

import torch

from worldfoundry.training.data.wan.encoding import WanTextFeatureEncoder

from ..rollout_cache import (
    RolloutConditioningPreparationResult,
    RolloutConditioningStore,
    prepare_rollout_conditioning_cache,
)
from ..rollout_manifest import RolloutPromptDataset
from ..video_precompute import checkpoint_spec_identity
from .assets import Wan22TextCheckpoints, wan22_text_checkpoints

WAN22_T2V_A14B_MODEL = "wan2.2-t2v-a14b"


def _local_checkpoint(default: object, value: object) -> object:
    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec

    if isinstance(value, CheckpointSpec):
        return value
    path = Path(str(value)).expanduser()
    if path.is_dir():
        return CheckpointSpec(
            source=path,
            files=tuple(getattr(default, "files", ())),
            allow_patterns=tuple(getattr(default, "allow_patterns", ())),
        )
    return CheckpointSpec(source=path.parent, files=(path.name,))


def _resolve_text_checkpoints(
    overrides: Mapping[str, object] | None,
) -> Wan22TextCheckpoints:
    selected = wan22_text_checkpoints()
    values = dict(overrides or {})
    return Wan22TextCheckpoints(
        text_encoder=_local_checkpoint(selected.text_encoder, values["text-encoder"])
        if "text-encoder" in values
        else selected.text_encoder,
        tokenizer=_local_checkpoint(selected.tokenizer, values["tokenizer"])
        if "tokenizer" in values
        else selected.tokenizer,
    )


def materialize_wan22_rollout_conditioning_cache(
    recipe: object,
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    device: str | torch.device = "cuda",
    checkpoint_overrides: Mapping[str, object] | None = None,
) -> RolloutConditioningPreparationResult:
    """Encode prompts with the official A14B UMT5 role only."""

    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentBuildContext,
        ComponentKey,
        ComponentKind,
    )
    from worldfoundry.base_models.diffusion_model.models.encoders.wan import (
        build_wan_text_conditioner,
    )
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
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
        raise TypeError("Wan2.2 rollout conditioning requires a rollout-based post-training recipe")
    if recipe.model.recipe != WAN22_T2V_A14B_MODEL:
        raise ValueError("Wan2.2 rollout conditioning supports T2V-A14B only")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    store = RolloutConditioningStore(cache_dir)
    if store.index_path.exists():
        raise FileExistsError("Wan2.2 rollout conditioning index already exists")
    prompts = RolloutPromptDataset.from_file(manifest_path, split=recipe.data.split)
    generation = recipe.data.options.get("generation")
    if not isinstance(generation, Mapping):
        raise TypeError("rollout data.options.generation must be a mapping")

    checkpoints = _resolve_text_checkpoints(checkpoint_overrides)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[recipe.runtime.param_dtype]
    conditioner = build_wan_text_conditioner(
        ComponentBuildContext(
            model_id=recipe.model.recipe,
            key=ComponentKey(ComponentKind.CONDITIONER),
            purpose=BuildPurpose.TRAINING,
            policy=RuntimePolicy(
                device=resolved_device,
                dtype=dtype,
                attention=AttentionBackend.TORCH,
            ),
            checkpoints={
                "weights": checkpoints.text_encoder,
                "tokenizer": checkpoints.tokenizer,
            },
        )
    )
    encoder = WanTextFeatureEncoder(conditioner)
    try:
        return prepare_rollout_conditioning_cache(
            prompts,
            cache_root=cache_dir,
            encoder=encoder,
            model_recipe=recipe.model.recipe,
            conditioner=checkpoint_spec_identity(checkpoints.text_encoder),
            tokenizer=checkpoint_spec_identity(checkpoints.tokenizer),
            generation_defaults={str(key): value for key, value in generation.items()},
        )
    finally:
        del encoder, conditioner
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()


__all__ = [
    "WAN22_T2V_A14B_MODEL",
    "materialize_wan22_rollout_conditioning_cache",
]
