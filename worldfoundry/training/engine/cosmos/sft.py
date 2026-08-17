"""Cache-backed native video SFT for Cosmos Predict2, Predict2.5, and Cosmos3."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.recipes.spec import NativeDiffusionRecipe
from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.ema import PowerEMA
from worldfoundry.training.models.cosmos import (
    Cosmos3TrainAdapter,
    CosmosPredict2TrainAdapter,
    CosmosPredict25TrainAdapter,
    build_cached_cosmos3_train_adapter,
    build_cached_cosmos_predict2_train_adapter,
    build_cached_cosmos_predict25_train_adapter,
)
from worldfoundry.training.objectives.flow_matching import FlowMatchingConfig, FlowMatchingObjective
from worldfoundry.training.recipes.spec import TrainingRecipe
from worldfoundry.training.tuning.peft import (
    LoraTargetAudit,
    PeftLoraApplication,
    apply_peft_lora_with_audit,
)

from ..video_flow import (
    audit_video_cache_against_manifest,
    build_cached_video_flow_fsdp2_session,
    build_cached_video_flow_single_device_session,
    torch_dtype,
)
from .objective import (
    COSMOS3_NANO_CFG_DROPOUT,
    COSMOS3_NANO_CONDITIONING_CONFIG,
    COSMOS_PREDICT_LORA_CONDITIONAL_FRAME_PROBABILITIES,
    COSMOS_PREDICT_LOSS_SCALE,
    Cosmos3VisionFlowMatchingObjective,
    CosmosPredictFlowMatchingObjective,
)
from .precision import promote_trainable_parameters_to_fp32

CosmosTrainAdapter = CosmosPredict2TrainAdapter | CosmosPredict25TrainAdapter | Cosmos3TrainAdapter

COSMOS_PREDICT_LORA_PRESET = "cosmos-predict-attention-mlp"
COSMOS3_LORA_PRESET = "cosmos3-generation-attention"
COSMOS3_NANO_VISION_SFT_PRESET = "cosmos3-nano-vision-sft"

_PREDICT2_RECIPES = frozenset({"cosmos-predict2-2b-video2world", "cosmos-predict2-14b-video2world"})
_PREDICT25_RECIPES = frozenset({"cosmos-predict2.5-2b", "cosmos-predict2.5-14b"})
_COSMOS3_RECIPES = frozenset({"cosmos3-nano", "cosmos3-super"})
_SUPPORTED_RECIPES = _PREDICT2_RECIPES | _PREDICT25_RECIPES | _COSMOS3_RECIPES
_PREDICT25_PRETRAINED_CHECKPOINT = "transformer-pretrained"

_PREDICT_TARGET = re.compile(
    r"^transformer_blocks\.(?P<block>\d+)\."
    r"(?P<role>attn[12]\.(?:to_q|to_k|to_v|to_out\.0)|ff\.(?:net\.0\.proj|net\.2))$"
)
_PREDICT_ROLES = frozenset(
    {
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    }
)
_COSMOS3_TARGET = re.compile(
    r"^layers\.(?P<block>\d+)\.self_attn\."
    r"(?P<role>add_q_proj|add_k_proj|add_v_proj|to_add_out)$"
)
_COSMOS3_ROLES = frozenset({"add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"})
_COSMOS3_NANO_REMAPPED_MOE_GEN_PARTS = (
    ".self_attn.add_q_proj.",
    ".self_attn.add_k_proj.",
    ".self_attn.add_v_proj.",
    ".self_attn.to_add_out.",
    ".self_attn.norm_added_q.",
    ".self_attn.norm_added_k.",
)

_FLOW_OPTIONS = frozenset(
    {
        "flow_shift",
        "logit_mean",
        "logit_std",
        "max_sigma",
        "min_sigma",
        "num_train_timesteps",
        "loss_scale",
        "conditional_frame_probabilities",
        "conditioning_config",
    }
)


def _build_cosmos_power_ema(model: nn.Module) -> PowerEMA:
    return PowerEMA(model, rate=0.1, iteration_shift=0)


def _uses_cosmos_power_ema(adapter: CosmosTrainAdapter) -> bool:
    return isinstance(adapter, (CosmosPredict25TrainAdapter, Cosmos3TrainAdapter))


def cosmos_training_checkpoint_overrides(
    recipe: TrainingRecipe,
    native_recipe: NativeDiffusionRecipe,
    checkpoint_overrides: Mapping[str, object] | None,
    *,
    base_dir: Path,
) -> dict[str, object]:
    """Select the released pre-trained Predict2.5 base for LoRA SFT."""

    overrides = dict(checkpoint_overrides or {})
    if recipe.model.checkpoint != "default":
        if "transformer" in overrides:
            raise ValueError("model.checkpoint and checkpoint_overrides['transformer'] cannot both be set")
        path = Path(recipe.model.checkpoint)
        overrides["transformer"] = str(path if path.is_absolute() else base_dir / path)
        return overrides
    variant = recipe.model.options.get("checkpoint_variant")
    if variant is not None and variant != "pretrained":
        raise ValueError(f"unsupported Cosmos checkpoint_variant: {variant!r}")
    if "transformer" not in overrides and recipe.model.recipe in _PREDICT25_RECIPES and recipe.tuning.mode == "lora":
        overrides["transformer"] = native_recipe.checkpoints[_PREDICT25_PRETRAINED_CHECKPOINT]
    return overrides


def audit_cosmos_lora_targets(model: nn.Module, preset: str) -> LoraTargetAudit:
    """Resolve the exact official generation targets on the local model graph."""

    normalized = str(preset).strip().lower().replace("_", "-")
    if normalized == COSMOS_PREDICT_LORA_PRESET:
        blocks = getattr(model, "transformer_blocks", None)
        pattern, expected = _PREDICT_TARGET, _PREDICT_ROLES
    elif normalized == COSMOS3_LORA_PRESET:
        blocks = getattr(model, "layers", None)
        pattern, expected = _COSMOS3_TARGET, _COSMOS3_ROLES
    else:
        raise ValueError(f"unsupported Cosmos LoRA preset: {preset!r}")
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError(f"{normalized} requires its native block ModuleList")

    names: list[str] = []
    shapes: dict[str, tuple[int, int]] = {}
    roles = {index: set() for index in range(len(blocks))}
    for name, module in model.named_modules():
        match = pattern.fullmatch(name)
        if match is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Cosmos LoRA target {name!r} must be nn.Linear")
        block = int(match.group("block"))
        roles[block].add(match.group("role"))
        names.append(name)
        shapes[name] = (int(module.in_features), int(module.out_features))
    drift = {index: sorted(expected - found) for index, found in roles.items() if found != expected}
    if drift:
        raise ValueError(f"{normalized} target graph is missing official roles: {drift}")
    names.sort()
    return LoraTargetAudit(
        preset=normalized,
        target_pattern=pattern.pattern,
        module_names=tuple(names),
        module_shapes=shapes,
        block_count=len(blocks),
        base_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def apply_cosmos_tuning(
    recipe: TrainingRecipe,
    adapter: CosmosTrainAdapter,
) -> PeftLoraApplication | None:
    if recipe.tuning.mode == "full":
        adapter.trainable_module.requires_grad_(True)
        return None
    if recipe.tuning.mode == "partial":
        if not isinstance(adapter, Cosmos3TrainAdapter) or recipe.tuning.preset != COSMOS3_NANO_VISION_SFT_PRESET:
            raise ValueError("partial Cosmos tuning requires the Cosmos3 Nano vision SFT preset")
        adapter.trainable_module.requires_grad_(False)
        selected_roles: set[str] = set()
        for name, parameter in adapter.trainable_module.named_parameters():
            role: str | None = None
            if "moe_gen" in name or any(part in name for part in _COSMOS3_NANO_REMAPPED_MOE_GEN_PARTS):
                role = "moe_gen"
            elif name.startswith("time_embedder."):
                role = "time_embedder"
            elif name.startswith("proj_in."):
                role = "vae2llm"
            elif name.startswith("proj_out."):
                role = "llm2vae"
            if role is not None:
                parameter.requires_grad_(True)
                selected_roles.add(role)
        expected = {"moe_gen", "time_embedder", "vae2llm", "llm2vae"}
        if selected_roles != expected:
            missing = sorted(expected - selected_roles)
            raise ValueError(f"Cosmos3 Nano vision SFT parameter groups are missing: {missing}")
        return None
    if recipe.tuning.preset != adapter.lora_target_preset:
        raise ValueError("recipe LoRA preset differs from the Cosmos family target preset")
    assert recipe.tuning.rank is not None
    assert recipe.tuning.alpha is not None
    application = apply_peft_lora_with_audit(
        adapter.trainable_module,
        audit=audit_cosmos_lora_targets(adapter.trainable_module, recipe.tuning.preset),
        rank=recipe.tuning.rank,
        alpha=recipe.tuning.alpha,
        dropout=recipe.tuning.dropout,
        modules_to_save=recipe.tuning.modules_to_save,
    )
    adapter.denoiser.model = application.model
    adapter.trainable_module = application.model
    if isinstance(adapter, CosmosPredict25TrainAdapter):
        promote_trainable_parameters_to_fp32(adapter.trainable_module)
    return application


def _flow_objective(recipe: TrainingRecipe, adapter: CosmosTrainAdapter) -> FlowMatchingObjective:
    options = dict(recipe.objective.options)
    unknown = sorted(set(options) - _FLOW_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported Cosmos flow objective options: {unknown}")
    loss_scale = options.pop("loss_scale", COSMOS_PREDICT_LOSS_SCALE)
    conditional_probabilities = options.pop("conditional_frame_probabilities", None)
    conditioning_config = options.pop("conditioning_config", None)
    config = FlowMatchingConfig(timestep_sampler=recipe.objective.timestep_sampler, **options)
    if config.num_train_timesteps is not None and config.num_train_timesteps != adapter.num_train_timesteps:
        raise ValueError("objective num_train_timesteps differs from the Cosmos adapter")
    if isinstance(adapter, (CosmosPredict2TrainAdapter, CosmosPredict25TrainAdapter)):
        if conditioning_config is not None:
            raise ValueError("conditioning_config is only valid for Cosmos3 vision SFT")
        if conditional_probabilities is None and recipe.tuning.mode == "lora":
            conditional_probabilities = COSMOS_PREDICT_LORA_CONDITIONAL_FRAME_PROBABILITIES
        return CosmosPredictFlowMatchingObjective(
            config,
            loss_scale=loss_scale,
            conditional_frame_probabilities=conditional_probabilities,
            conditioning_dropout_probability=recipe.objective.conditioning_dropout,
        )
    if "loss_scale" in recipe.objective.options or "conditional_frame_probabilities" in recipe.objective.options:
        raise ValueError("Predict-only objective options cannot be used for Cosmos3")
    return Cosmos3VisionFlowMatchingObjective(
        config,
        conditioning_config=conditioning_config or COSMOS3_NANO_CONDITIONING_CONFIG,
        conditioning_dropout=recipe.objective.conditioning_dropout,
    )


def _validate_recipe(recipe: TrainingRecipe, adapter: CosmosTrainAdapter) -> None:
    if recipe.model.recipe not in _SUPPORTED_RECIPES:
        raise ValueError(f"native Cosmos training does not support {recipe.model.recipe!r}")
    if recipe.objective.type != "flow-matching" or recipe.objective.prediction_type != "flow_velocity":
        raise ValueError("Cosmos SFT requires flow-matching velocity prediction")
    if isinstance(adapter, CosmosPredict25TrainAdapter) and recipe.tuning.mode == "lora":
        if recipe.objective.conditioning_dropout != 0.2:
            raise ValueError("Cosmos Predict2.5 LoRA text conditioning dropout must be 0.2")
    elif isinstance(adapter, Cosmos3TrainAdapter):
        if recipe.objective.conditioning_dropout != COSMOS3_NANO_CFG_DROPOUT:
            raise ValueError("Cosmos3 vision SFT conditioning_dropout must be 0.1")
        if recipe.model.options.get("use_system_prompt", False) is not False:
            raise ValueError("Cosmos3 vision SFT requires use_system_prompt=false")
        if recipe.model.recipe == "cosmos3-nano" and (
            recipe.tuning.mode != "partial" or recipe.tuning.preset != COSMOS3_NANO_VISION_SFT_PRESET
        ):
            raise ValueError("Cosmos3 Nano vision SFT uses the released selected-key partial tuning policy")
        if recipe.runtime.activation_checkpoint not in {"none", "full"}:
            raise ValueError("Cosmos3 activation_checkpoint must be 'none' or 'full'")
        if adapter.gradient_checkpointing != (recipe.runtime.activation_checkpoint == "full"):
            raise ValueError("Cosmos3 adapter activation checkpointing differs from the recipe")
    elif recipe.objective.conditioning_dropout != 0.0:
        raise ValueError("this Cosmos training path does not enable conditioning dropout")
    if not isinstance(adapter, Cosmos3TrainAdapter) and recipe.runtime.activation_checkpoint != "none":
        raise ValueError("Cosmos activation checkpointing is not wired into the native graph")
    family_matches = (
        (isinstance(adapter, CosmosPredict2TrainAdapter) and recipe.model.recipe in _PREDICT2_RECIPES)
        or (isinstance(adapter, CosmosPredict25TrainAdapter) and recipe.model.recipe in _PREDICT25_RECIPES)
        or (isinstance(adapter, Cosmos3TrainAdapter) and recipe.model.recipe in _COSMOS3_RECIPES)
    )
    if not family_matches:
        raise ValueError("Cosmos recipe and training adapter belong to different model families")


def validate_cosmos_cache_contract(
    recipe: TrainingRecipe,
    adapter: CosmosTrainAdapter,
    dataset: VideoCachedDataset,
) -> dict[str, object]:
    """Check the tensors consumed by the Cosmos training adapter."""

    context_names = (
        ("condition.input_ids", "condition.empty_input_ids")
        if isinstance(adapter, Cosmos3TrainAdapter)
        else ("condition.context",)
    )
    for entry in dataset.index.entries:
        latents = entry.tensors["clean_latents"]
        if entry.provenance.model_recipe != recipe.model.recipe:
            raise ValueError(f"cache entry {entry.sample_id!r} belongs to another model recipe")
        if latents.shape[0] != adapter.expected_latent_channels:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible latent channels")
        missing_context = [name for name in context_names if name not in entry.tensors]
        if missing_context:
            raise ValueError(f"cache entry {entry.sample_id!r} lacks {missing_context!r}")
        patch_size = getattr(adapter, "patch_size", None)
        if patch_size is not None and any(
            axis % patch for axis, patch in zip(latents.shape[-3:], patch_size, strict=True)
        ):
            raise ValueError(f"cache entry {entry.sample_id!r} is incompatible with the Predict DiT patch size")
    return {
        "model_recipe": recipe.model.recipe,
        "latent_channels": adapter.expected_latent_channels,
        "conditioning": [name.removeprefix("condition.") for name in context_names],
        "prediction_type": adapter.prediction_type,
    }


def build_cosmos_single_device_session(
    *,
    recipe: TrainingRecipe,
    adapter: CosmosTrainAdapter,
    dataset: VideoCachedDataset,
    output_dir: str | Path | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
):
    """Build a shared-engine single-device Cosmos SFT session."""

    _validate_recipe(recipe, adapter)
    objective = _flow_objective(recipe, adapter)
    contract = validate_cosmos_cache_contract(recipe, adapter, dataset)
    use_power_ema = _uses_cosmos_power_ema(adapter)
    return build_cached_video_flow_single_device_session(
        recipe=recipe,
        adapter=adapter,
        dataset=dataset,
        objective=objective,
        cache_contract=contract,
        output_dir=output_dir,
        tuning_factory=apply_cosmos_tuning,
        ema_factory=_build_cosmos_power_ema if use_power_ema else None,
        export_ema=use_power_ema,
        ema_update="optimizer-step",
        fused_adamw=fused_adamw,
        initialization_seed=initialization_seed,
    )


def build_cosmos_fsdp2_session(
    *,
    recipe: TrainingRecipe,
    adapter: CosmosTrainAdapter,
    dataset: VideoCachedDataset,
    distributed_context: DistributedTrainingContext,
    output_dir: str | Path | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
):
    """Build the same Cosmos SFT session on the active FSDP2 world size."""

    _validate_recipe(recipe, adapter)
    objective = _flow_objective(recipe, adapter)
    contract = validate_cosmos_cache_contract(recipe, adapter, dataset)
    use_power_ema = _uses_cosmos_power_ema(adapter)
    return build_cached_video_flow_fsdp2_session(
        recipe=recipe,
        adapter=adapter,
        dataset=dataset,
        objective=objective,
        cache_contract=contract,
        distributed_context=distributed_context,
        output_dir=output_dir,
        tuning_factory=apply_cosmos_tuning,
        ema_factory=_build_cosmos_power_ema if use_power_ema else None,
        export_ema=use_power_ema,
        ema_update="optimizer-step",
        fused_adamw=fused_adamw,
        initialization_seed=initialization_seed,
        master_parameter_dtype=torch.float32,
    )


def materialize_cosmos_cached_training_session(
    recipe: TrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    checkpoint_overrides: Mapping[str, object] | None = None,
    verify_media_files: bool = True,
    audit_cache_on_open: bool = True,
    verify_cache_on_read: bool = True,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
):
    """Load a Cosmos video-generation DiT and construct its cached SFT session."""

    if recipe.model.recipe not in _SUPPORTED_RECIPES:
        raise ValueError(f"native Cosmos materialization does not support {recipe.model.recipe!r}")
    if recipe.data.cache is None:
        raise ValueError("cached Cosmos training requires data.cache")
    root = Path(base_dir).expanduser().resolve()
    cache_path = Path(recipe.data.cache)
    manifest_path = Path(recipe.data.manifest)
    destination = Path(output_dir or recipe.run.output_dir)
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not destination.is_absolute():
        destination = root / destination

    manifest = TrainingManifestDataset.from_file(
        manifest_path,
        split=recipe.data.split,
        verify_files=verify_media_files,
    )
    cache = VideoCachedDataset(
        cache_path,
        expected_sample_ids=manifest.sample_ids,
        audit_on_open=audit_cache_on_open,
        verify_on_read=verify_cache_on_read,
    )
    audit_video_cache_against_manifest(cache, manifest)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    distributed_context = None
    if recipe.distributed.backend == "fsdp2":
        if resolved_device.type != "cuda":
            raise ValueError("Cosmos FSDP2 materialization requires CUDA")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device
    elif recipe.distributed.backend != "single":
        raise ValueError(f"Cosmos materialization does not support {recipe.distributed.backend!r}")

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import BuildPurpose, ComponentKey, ComponentKind
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry

    try:
        native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
        overrides = cosmos_training_checkpoint_overrides(
            recipe,
            native_recipe,
            checkpoint_overrides,
            base_dir=root,
        )
        dtype = torch_dtype(recipe.runtime.param_dtype)
        denoiser_key = ComponentKey(ComponentKind.DENOISER)
        components = NativeDiffusionAssembler().build_components(
            native_recipe,
            purpose=BuildPurpose.TRAINING,
            policy=RuntimePolicy(device=resolved_device, dtype=dtype, attention=AttentionBackend.TORCH),
            checkpoint_overrides=overrides,
            component_keys=(denoiser_key,),
        )
        options = {
            "expected_latent_channels": int(native_recipe.options["latent_channels"]),
            "num_train_timesteps": int(recipe.objective.options.get("num_train_timesteps", 1000)),
        }
        if recipe.model.recipe in _PREDICT2_RECIPES:
            adapter = build_cached_cosmos_predict2_train_adapter(
                components,
                temporal_compression=int(native_recipe.options["temporal_compression"]),
                spatial_compression=int(native_recipe.options["spatial_compression"]),
                **options,
            )
        elif recipe.model.recipe in _PREDICT25_RECIPES:
            adapter = build_cached_cosmos_predict25_train_adapter(
                components,
                temporal_compression=int(native_recipe.options["temporal_compression"]),
                spatial_compression=int(native_recipe.options["spatial_compression"]),
                **options,
            )
        else:
            adapter = build_cached_cosmos3_train_adapter(
                components,
                gradient_checkpointing=recipe.runtime.activation_checkpoint == "full",
                **options,
            )
        if distributed_context is not None:
            return build_cosmos_fsdp2_session(
                recipe=recipe,
                adapter=adapter,
                dataset=cache,
                distributed_context=distributed_context,
                output_dir=destination,
                fused_adamw=fused_adamw,
                initialization_seed=initialization_seed,
            )
        return build_cosmos_single_device_session(
            recipe=recipe,
            adapter=adapter,
            dataset=cache,
            output_dir=destination,
            fused_adamw=fused_adamw,
            initialization_seed=initialization_seed,
        )
    except Exception:
        if distributed_context is not None:
            distributed_context.close()
        raise


__all__ = [
    "COSMOS3_LORA_PRESET",
    "COSMOS3_NANO_VISION_SFT_PRESET",
    "COSMOS_PREDICT_LORA_PRESET",
    "audit_cosmos_lora_targets",
    "apply_cosmos_tuning",
    "build_cosmos_fsdp2_session",
    "build_cosmos_single_device_session",
    "cosmos_training_checkpoint_overrides",
    "materialize_cosmos_cached_training_session",
    "validate_cosmos_cache_contract",
]
