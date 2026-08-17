"""HunyuanVideo role loading, LoRA targets, and activation checkpointing."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from torch import nn

from worldfoundry.training.models.hunyuan_video import HunyuanVideoTrainAdapter
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.peft import (
    LoraTargetAudit,
    PeftLoraApplication,
    apply_peft_lora_with_audit,
)

_HUNYUAN_VIDEO15_RL_REPOSITORY = "tencent/HunyuanVideo-1.5"
_HUNYUAN_VIDEO15_RL_TRANSFORMER = "transformer/480p_t2v"


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _linear_targets(model: nn.Module, names: Iterable[str], *, preset: str) -> LoraTargetAudit:
    modules = dict(model.named_modules())
    resolved = tuple(sorted(set(names)))
    shapes: dict[str, tuple[int, int]] = {}
    for name in resolved:
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"HunyuanVideo LoRA target {name!r} is missing from the loaded graph")
        shapes[name] = (int(module.in_features), int(module.out_features))
    pattern = r"^(?:" + "|".join(re.escape(name) for name in resolved) + r")$"
    return LoraTargetAudit(
        preset=preset,
        target_pattern=pattern,
        module_names=resolved,
        module_shapes=shapes,
        block_count=len(tuple(getattr(model, "double_blocks", ()))) + len(tuple(getattr(model, "single_blocks", ()))),
        base_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def audit_hunyuan_video_lora_targets(adapter: HunyuanVideoTrainAdapter) -> LoraTargetAudit:
    """Map the released attention/FFN target set onto each native graph."""

    model = adapter.trainable_module
    names: list[str] = []
    if adapter.contract.architecture == "original":
        for index, _ in enumerate(getattr(model, "double_blocks", ())):
            for stream in ("component_a", "component_b"):
                base = f"double_blocks.{index}.{stream}"
                names.extend(
                    (
                        f"{base}.to_qkv",
                        f"{base}.to_out",
                        f"{base}.ff.0",
                        f"{base}.ff.2",
                    )
                )
        for index, _ in enumerate(getattr(model, "single_blocks", ())):
            base = f"single_blocks.{index}"
            names.extend(
                (
                    f"{base}.to_qkv",
                    f"{base}.to_out",
                    f"{base}.ff.0",
                    f"{base}.ff.2",
                )
            )
    else:
        for index, _ in enumerate(getattr(model, "double_blocks", ())):
            base = f"double_blocks.{index}"
            for stream in ("img", "txt"):
                names.extend(f"{base}.{stream}_attn_{projection}" for projection in ("q", "k", "v", "proj"))
                names.extend((f"{base}.{stream}_mlp.fc1", f"{base}.{stream}_mlp.fc2"))
        for index, _ in enumerate(getattr(model, "single_blocks", ())):
            base = f"single_blocks.{index}"
            names.extend(
                (
                    f"{base}.linear1_q",
                    f"{base}.linear1_k",
                    f"{base}.linear1_v",
                    f"{base}.linear1_mlp",
                    f"{base}.linear2.fc",
                )
            )
        token_refiner = getattr(
            getattr(getattr(model, "txt_in", None), "individual_token_refiner", None),
            "blocks",
            (),
        )
        for index, _ in enumerate(token_refiner):
            base = f"txt_in.individual_token_refiner.blocks.{index}"
            names.extend(
                (
                    f"{base}.self_attn_qkv",
                    f"{base}.self_attn_proj",
                    f"{base}.mlp.fc1",
                    f"{base}.mlp.fc2",
                )
            )
    if not names:
        raise ValueError("HunyuanVideo LoRA requires native double/single transformer blocks")
    return _linear_targets(model, names, preset=adapter.lora_target_preset)


def apply_hunyuan_video_tuning(
    recipe: PostTrainingRecipe,
    adapter: HunyuanVideoTrainAdapter,
) -> PeftLoraApplication | None:
    if recipe.tuning.mode == "full":
        adapter.trainable_module.requires_grad_(True)
        return None
    if recipe.tuning.mode != "lora":
        raise ValueError("HunyuanVideo post-training supports full or LoRA tuning")
    if recipe.tuning.preset != adapter.lora_target_preset:
        raise ValueError("HunyuanVideo recipe selected the wrong architecture's LoRA preset")
    assert recipe.tuning.rank is not None
    assert recipe.tuning.alpha is not None
    application = apply_peft_lora_with_audit(
        adapter.trainable_module,
        audit=audit_hunyuan_video_lora_targets(adapter),
        rank=recipe.tuning.rank,
        alpha=recipe.tuning.alpha,
        dropout=recipe.tuning.dropout,
        modules_to_save=recipe.tuning.modules_to_save,
    )
    adapter.denoiser.model = application.model
    adapter.trainable_module = application.model
    return application


def apply_hunyuan_video_activation_checkpointing(adapter: HunyuanVideoTrainAdapter) -> None:
    """Checkpoint every native double/single stream block before FSDP2 wrapping."""

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    block_types = adapter.fsdp_block_classes

    def wrapper(module: nn.Module) -> nn.Module:
        return checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    apply_activation_checkpointing(
        adapter.trainable_module,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, block_types),
    )
    wrapped_blocks = tuple(getattr(adapter.trainable_module, "double_blocks", ())) + tuple(
        getattr(adapter.trainable_module, "single_blocks", ())
    )
    adapter.fsdp_block_classes = tuple(
        dict.fromkeys(type(block) for block in wrapped_blocks if isinstance(block, nn.Module))
    )


def validate_hunyuan_video_dtype(
    adapter: HunyuanVideoTrainAdapter,
    expected: torch.dtype,
) -> None:
    dtypes = {parameter.dtype for parameter in adapter.trainable_module.parameters() if parameter.is_floating_point()}
    if dtypes != {expected}:
        raise ValueError(
            "loaded HunyuanVideo parameter dtype differs from runtime.param_dtype: "
            f"loaded={sorted(map(str, dtypes))}, expected={expected}"
        )


def load_hunyuan_video_role_adapter(
    *,
    model_recipe: str,
    checkpoint: object | None,
    device: str | torch.device,
    dtype: torch.dtype,
) -> HunyuanVideoTrainAdapter:
    """Load one independent native denoiser without constructing inference runtime."""

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentKey,
        ComponentKind,
    )
    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
    from worldfoundry.training.models.hunyuan_video import build_cached_hunyuan_video_train_adapter

    native_recipe = default_native_diffusion_registry().resolve(model_recipe)
    if native_recipe.model_id != model_recipe:
        raise ValueError("HunyuanVideo role loading requires the canonical native model id")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    overrides = {}
    component_options = {}
    if checkpoint is None and model_recipe == "hunyuanvideo-1.5-t2v":
        overrides["transformer"] = CheckpointSpec(
            repo_id=_HUNYUAN_VIDEO15_RL_REPOSITORY,
            files=(f"{_HUNYUAN_VIDEO15_RL_TRANSFORMER}/diffusion_pytorch_model.safetensors",),
            allow_patterns=(f"{_HUNYUAN_VIDEO15_RL_TRANSFORMER}/*",),
        )
        component_options[denoiser_key] = {
            "config_path": f"{_HUNYUAN_VIDEO15_RL_TRANSFORMER}/config.json",
        }
    elif checkpoint is not None:
        if not isinstance(checkpoint, (CheckpointSpec, str)):
            raise TypeError("HunyuanVideo checkpoint override must be CheckpointSpec or a local path")
        overrides["transformer"] = checkpoint
    components = NativeDiffusionAssembler().build_components(
        native_recipe,
        purpose=BuildPurpose.TRAINING,
        policy=RuntimePolicy(
            device=torch.device(device),
            dtype=dtype,
            attention=AttentionBackend.TORCH,
        ),
        checkpoint_overrides=overrides,
        component_options=component_options,
        component_keys=(denoiser_key,),
    )
    return build_cached_hunyuan_video_train_adapter(
        components,
        model_recipe=model_recipe,
    )


__all__ = [
    "apply_hunyuan_video_activation_checkpointing",
    "apply_hunyuan_video_tuning",
    "audit_hunyuan_video_lora_targets",
    "load_hunyuan_video_role_adapter",
    "torch_dtype",
    "validate_hunyuan_video_dtype",
]
