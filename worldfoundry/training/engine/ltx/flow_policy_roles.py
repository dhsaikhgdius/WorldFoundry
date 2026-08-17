"""LTX policy role loading and RL-specific LoRA targeting."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.training.models.ltx import LTXTrainAdapter, build_cached_ltx_train_adapter
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.peft import (
    LoraTargetAudit,
    PeftLoraApplication,
    apply_peft_lora_with_audit,
)

_LTX_POLICY_TARGET_PATTERN = re.compile(
    r"^(?:.*\.)?velocity_model\.transformer_blocks\.\d+\."
    r"(?:attn[12]\.(?:to_q|to_k|to_v|to_out\.0)|ff\.net\.(?:0\.proj|2))$"
)

_LTX_POLICY_DEV_CHECKPOINTS = {
    "ltx-2-i2v": "ltx-2-19b-dev.safetensors",
    "ltx-2.3-i2v": "ltx-2.3-22b-dev.safetensors",
}


def ltx_policy_default_checkpoint(model_recipe: str) -> CheckpointSpec:
    """Return the released dev checkpoint used for LTX policy optimization."""

    from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry

    recipe = str(model_recipe).strip().lower().replace("_", "-")
    try:
        filename = _LTX_POLICY_DEV_CHECKPOINTS[recipe]
    except KeyError as error:
        raise ValueError(f"unsupported LTX policy model: {model_recipe!r}") from error
    released = default_native_diffusion_registry().resolve(recipe).checkpoints["model"]
    return CheckpointSpec(
        repo_id=released.repo_id,
        revision=released.revision,
        files=(filename,),
    )


def audit_ltx_policy_lora_targets(model: nn.Module) -> LoraTargetAudit:
    """Select the attention and feed-forward projections used by LTX policy RL."""

    velocity_model = getattr(model, "velocity_model", None)
    blocks = getattr(velocity_model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("LTX policy LoRA requires velocity_model.transformer_blocks")
    modules = dict(model.named_modules())
    names: list[str] = []
    shapes: dict[str, tuple[int, int]] = {}
    for name, module in modules.items():
        if _LTX_POLICY_TARGET_PATTERN.fullmatch(name) is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LTX policy LoRA target {name!r} is not linear")
        names.append(name)
        shapes[name] = (int(module.in_features), int(module.out_features))
    expected_per_block = 10
    if len(names) != len(blocks) * expected_per_block:
        raise ValueError("LTX policy attention or feed-forward target graph differs from the loaded model")
    names.sort()
    return LoraTargetAudit(
        preset="ltx-policy",
        target_pattern=_LTX_POLICY_TARGET_PATTERN.pattern,
        module_names=tuple(names),
        module_shapes=shapes,
        block_count=len(blocks),
        base_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def apply_ltx_policy_tuning(
    recipe: PostTrainingRecipe,
    adapter: LTXTrainAdapter,
) -> PeftLoraApplication | None:
    """Apply full tuning or the released LTX policy attention-plus-FFN LoRA."""

    if recipe.tuning.mode == "full":
        adapter.trainable_module.requires_grad_(True)
        return None
    if recipe.tuning.mode != "lora":
        raise ValueError("LTX policy training supports full or LoRA tuning")
    if recipe.tuning.preset != "ltx-policy":
        raise ValueError("LTX policy LoRA requires tuning.preset='ltx-policy'")
    assert recipe.tuning.rank is not None and recipe.tuning.alpha is not None
    application = apply_peft_lora_with_audit(
        adapter.trainable_module,
        audit=audit_ltx_policy_lora_targets(adapter.trainable_module),
        rank=recipe.tuning.rank,
        alpha=recipe.tuning.alpha,
        dropout=recipe.tuning.dropout,
        modules_to_save=recipe.tuning.modules_to_save,
    )
    adapter.denoiser.model = application.model
    adapter.trainable_module = application.model
    return application


def load_ltx_policy_adapter(
    recipe: PostTrainingRecipe,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_overrides: Mapping[str, object] | None = None,
    base_dir: str | Path = ".",
) -> LTXTrainAdapter:
    """Load the native LTX transformer for one policy, old-policy, or reference role."""

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import BuildPurpose, ComponentKey, ComponentKind
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry

    native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
    overrides = dict(checkpoint_overrides or {})
    if recipe.model.checkpoint != "default":
        checkpoint = Path(recipe.model.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = Path(base_dir) / checkpoint
        overrides["model"] = str(checkpoint)
    elif "model" not in overrides:
        overrides["model"] = ltx_policy_default_checkpoint(recipe.model.recipe)
    components = NativeDiffusionAssembler().build_components(
        native_recipe,
        purpose=BuildPurpose.TRAINING,
        policy=RuntimePolicy(
            device=torch.device(device),
            dtype=dtype,
            attention=AttentionBackend.TORCH,
        ),
        checkpoint_overrides=overrides,
        component_keys=(ComponentKey(ComponentKind.DENOISER),),
    )
    return build_cached_ltx_train_adapter(
        components,
        expected_latent_channels=int(native_recipe.options.get("latent_channels", 128)),
        temporal_compression=int(native_recipe.options.get("temporal_compression", 8)),
        spatial_compression=int(native_recipe.options.get("spatial_compression", 32)),
        default_fps=24.0,
        first_frame_conditioning_probability=0.0,
        per_sample_first_frame_conditioning=True,
        causal_positions=True,
        discrete_timesteps=False,
        gradient_checkpointing=recipe.runtime.activation_checkpoint == "full",
    )


__all__ = [
    "apply_ltx_policy_tuning",
    "audit_ltx_policy_lora_targets",
    "load_ltx_policy_adapter",
    "ltx_policy_default_checkpoint",
]
