"""WorldFoundry-native LTX video flow-matching training sessions."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.networks.ltx.attention import Attention
from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.engine.sessions.fsdp2 import FSDP2TrainingSession
from worldfoundry.training.engine.sessions.single_device import SingleDeviceTrainingSession
from worldfoundry.training.engine.video_flow import (
    audit_video_cache_against_manifest,
    build_cached_video_flow_fsdp2_session,
    build_cached_video_flow_single_device_session,
    torch_dtype,
)
from worldfoundry.training.models.ltx import LTXTrainAdapter, build_cached_ltx_train_adapter
from worldfoundry.training.recipes.spec import TrainingRecipe
from worldfoundry.training.tuning.peft import (
    LoraTargetAudit,
    apply_peft_lora_with_audit,
)

from .cache import LTX_MODEL_RECIPES, validate_ltx_cache_contract
from .lora import LTXLoraApplication
from .objective import LTXFlowMatchingObjective, LTXTimestepSamplingConfig

_LTX_VIDEO_ATTENTION_PATTERN = re.compile(r"^(?:.*\.)?transformer_blocks\.\d+\.attn[12]\.(?:to_q|to_k|to_v|to_out\.0)$")
_LTX_OBJECTIVE_OPTIONS = frozenset(
    {
        "standard_deviation",
        "std",
        "epsilon",
        "eps",
        "uniform_probability",
        "uniform_prob",
        "first_frame_conditioning_probability",
    }
)


@dataclass(frozen=True, slots=True)
class _LTXTrainingProfile:
    stretch_timesteps: bool
    causal_positions: bool
    discrete_timesteps: bool
    per_sample_first_frame_conditioning: bool
    default_first_frame_probability: float


def _training_profile(model_recipe: str) -> _LTXTrainingProfile:
    if model_recipe == "ltx-video-i2v":
        return _LTXTrainingProfile(
            stretch_timesteps=False,
            causal_positions=False,
            discrete_timesteps=True,
            per_sample_first_frame_conditioning=False,
            default_first_frame_probability=0.1,
        )
    if model_recipe in {"ltx-2-i2v", "ltx-2.3-i2v"}:
        return _LTXTrainingProfile(
            stretch_timesteps=True,
            causal_positions=True,
            discrete_timesteps=False,
            per_sample_first_frame_conditioning=True,
            default_first_frame_probability=0.5,
        )
    raise ValueError(f"unsupported LTX model recipe: {model_recipe!r}")


def _uses_fp32_lora_master(recipe: TrainingRecipe) -> bool:
    return recipe.tuning.mode == "lora" and recipe.model.recipe in {"ltx-2-i2v", "ltx-2.3-i2v"}


def _pop_alias(options: dict[str, object], canonical: str, alias: str, default: float) -> float:
    if canonical in options and alias in options:
        raise ValueError(f"LTX objective cannot set both {canonical!r} and {alias!r}")
    return float(options.pop(canonical, options.pop(alias, default)))


def _objective_config(recipe: TrainingRecipe) -> tuple[LTXTimestepSamplingConfig, float]:
    profile = _training_profile(recipe.model.recipe)
    options = dict(recipe.objective.options)
    unknown = sorted(set(options) - _LTX_OBJECTIVE_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported LTX objective options: {unknown}")
    first_frame_probability = float(
        options.pop("first_frame_conditioning_probability", profile.default_first_frame_probability)
    )
    config = LTXTimestepSamplingConfig(
        mode=recipe.objective.timestep_sampler,
        standard_deviation=_pop_alias(options, "standard_deviation", "std", 1.0),
        stretch=profile.stretch_timesteps,
        epsilon=_pop_alias(options, "epsilon", "eps", 1.0e-3),
        uniform_probability=_pop_alias(options, "uniform_probability", "uniform_prob", 0.1),
    )
    return config, first_frame_probability


def build_ltx_flow_objective(recipe: TrainingRecipe) -> LTXFlowMatchingObjective:
    """Build the Lightricks sequence-length-aware flow objective."""

    config, _ = _objective_config(recipe)
    return LTXFlowMatchingObjective(config)


def validate_ltx_cached_recipe(
    recipe: TrainingRecipe,
    adapter: LTXTrainAdapter,
    *,
    backend: str,
) -> None:
    """Check the LTX-specific choices not covered by the shared video engine."""

    if recipe.model.recipe not in LTX_MODEL_RECIPES:
        raise ValueError(f"LTX training cannot train model recipe {recipe.model.recipe!r}")
    if recipe.distributed.backend != backend:
        raise ValueError(f"LTX training requires distributed.backend={backend!r}")
    if recipe.objective.type != "flow-matching" or recipe.objective.prediction_type != adapter.prediction_type:
        raise ValueError("LTX training requires flow-matching velocity prediction")
    if recipe.runtime.activation_checkpoint not in {"none", "full"}:
        raise ValueError("LTX activation_checkpoint must be 'none' or 'full'")
    if adapter.gradient_checkpointing != (recipe.runtime.activation_checkpoint == "full"):
        raise ValueError("LTX adapter activation checkpointing differs from the recipe")
    if recipe.objective.conditioning_dropout != 0.0:
        raise ValueError("the official LTX video training path does not drop text conditioning")
    if recipe.data.max_latent_tokens_per_microbatch is None:
        raise ValueError("LTX cached training requires a latent token budget")
    if recipe.data.tail_policy not in {"drop", "pad"}:
        raise ValueError("LTX distributed token batching requires tail_policy='drop' or 'pad'")

    profile = _training_profile(recipe.model.recipe)
    _, first_frame_probability = _objective_config(recipe)
    if not math.isclose(
        adapter.first_frame_conditioning_probability,
        first_frame_probability,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("LTX adapter first-frame conditioning probability differs from the recipe")
    if (
        adapter.causal_positions != profile.causal_positions
        or adapter.discrete_timesteps != profile.discrete_timesteps
        or adapter.per_sample_first_frame_conditioning != profile.per_sample_first_frame_conditioning
    ):
        raise ValueError("LTX adapter uses the wrong author-trainer compatibility profile")


def _validate_model_dtype(recipe: TrainingRecipe, adapter: LTXTrainAdapter) -> None:
    expected = torch_dtype(recipe.runtime.param_dtype)
    dtypes = {parameter.dtype for parameter in adapter.trainable_module.parameters() if parameter.is_floating_point()}
    if dtypes != {expected}:
        raise ValueError(
            "loaded LTX parameter dtype differs from runtime.param_dtype: "
            f"loaded={sorted(map(str, dtypes))}, expected={expected}"
        )


def audit_ltx_lora_targets(model: nn.Module) -> LoraTargetAudit:
    """Resolve the video attention projections executed by ``LTXTrainAdapter``."""

    velocity_model = getattr(model, "velocity_model", None)
    blocks = getattr(velocity_model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("LTX LoRA requires velocity_model.transformer_blocks")

    modules = dict(model.named_modules())
    blocks_name = next(name for name, module in modules.items() if module is blocks)
    attention_names: list[str] = []
    for index, block in enumerate(blocks):
        for role in ("attn1", "attn2"):
            attention = getattr(block, role, None)
            if not isinstance(attention, Attention):
                raise TypeError(f"LTX video block {index} must expose {role} as Attention")
            attention_names.append(f"{blocks_name}.{index}.{role}")
    expected_names = {
        f"{name}.{projection}" for name in attention_names for projection in ("to_q", "to_k", "to_v", "to_out.0")
    }
    names: list[str] = []
    shapes: dict[str, tuple[int, int]] = {}
    for name, module in modules.items():
        if _LTX_VIDEO_ATTENTION_PATTERN.fullmatch(name) is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LTX LoRA target {name!r} is not linear")
        names.append(name)
        shapes[name] = (int(module.in_features), int(module.out_features))
    if set(names) != expected_names:
        raise ValueError("LTX attention projection graph differs from the native transformer")
    names.sort()
    return LoraTargetAudit(
        preset="ltx-attention",
        target_pattern=_LTX_VIDEO_ATTENTION_PATTERN.pattern,
        module_names=tuple(names),
        module_shapes=shapes,
        block_count=len(blocks),
        base_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def apply_ltx_tuning(recipe: TrainingRecipe, adapter: object) -> LTXLoraApplication | None:
    """Apply full or video attention-only LoRA tuning to an LTX adapter."""

    if not isinstance(adapter, LTXTrainAdapter):
        raise TypeError("LTX tuning requires LTXTrainAdapter")
    if recipe.tuning.mode == "full":
        adapter.trainable_module.requires_grad_(True)
        return None
    if recipe.tuning.mode == "partial":
        raise ValueError("partial LTX tuning needs an explicit parameter-selection policy")
    if recipe.tuning.preset != adapter.lora_target_preset:
        raise ValueError("recipe LoRA preset differs from the LTX adapter")
    if _uses_fp32_lora_master(recipe):
        # The current LTX trainer keeps LoRA optimizer state on FP32 parameters
        # while mixed precision controls forward/backward compute independently.
        adapter.trainable_module.to(dtype=torch.float32)
    assert recipe.tuning.rank is not None
    assert recipe.tuning.alpha is not None
    application = apply_peft_lora_with_audit(
        adapter.trainable_module,
        audit=audit_ltx_lora_targets(adapter.trainable_module),
        rank=recipe.tuning.rank,
        alpha=recipe.tuning.alpha,
        dropout=recipe.tuning.dropout,
        modules_to_save=recipe.tuning.modules_to_save,
    )
    adapter.denoiser.model = application.model
    adapter.trainable_module = application.model
    return LTXLoraApplication(application)


def build_ltx_single_device_session(
    *,
    recipe: TrainingRecipe,
    adapter: LTXTrainAdapter,
    dataset: VideoCachedDataset,
    output_dir: str | Path | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> SingleDeviceTrainingSession:
    """Build a cache-backed single-device LTX training session."""

    validate_ltx_cached_recipe(recipe, adapter, backend="single")
    cache_contract = validate_ltx_cache_contract(recipe, adapter, dataset)
    _validate_model_dtype(recipe, adapter)
    return build_cached_video_flow_single_device_session(
        recipe=recipe,
        adapter=adapter,
        dataset=dataset,
        objective=build_ltx_flow_objective(recipe),
        cache_contract=cache_contract,
        output_dir=output_dir,
        tuning_factory=apply_ltx_tuning,
        fused_adamw=fused_adamw,
        initialization_seed=initialization_seed,
    )


def build_ltx_fsdp2_session(
    *,
    recipe: TrainingRecipe,
    adapter: LTXTrainAdapter,
    dataset: VideoCachedDataset,
    distributed_context: DistributedTrainingContext,
    output_dir: str | Path | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> FSDP2TrainingSession:
    """Build the same LTX session on any valid FSDP2 world size."""

    validate_ltx_cached_recipe(recipe, adapter, backend="fsdp2")
    cache_contract = validate_ltx_cache_contract(recipe, adapter, dataset)
    _validate_model_dtype(recipe, adapter)
    return build_cached_video_flow_fsdp2_session(
        recipe=recipe,
        adapter=adapter,
        dataset=dataset,
        objective=build_ltx_flow_objective(recipe),
        cache_contract=cache_contract,
        distributed_context=distributed_context,
        output_dir=output_dir,
        tuning_factory=apply_ltx_tuning,
        fused_adamw=fused_adamw,
        initialization_seed=initialization_seed,
        master_parameter_dtype=torch.float32 if _uses_fp32_lora_master(recipe) else None,
    )


def _resolved_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def materialize_ltx_cached_training_session(
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
) -> SingleDeviceTrainingSession | FSDP2TrainingSession:
    """Load only the native LTX transformer and construct a cached SFT run."""

    if recipe.model.recipe not in LTX_MODEL_RECIPES:
        raise ValueError(f"LTX materialization cannot train {recipe.model.recipe!r}")
    if recipe.data.cache is None:
        raise ValueError("cached LTX training requires data.cache")
    root = Path(base_dir).expanduser().resolve()
    manifest = TrainingManifestDataset.from_file(
        _resolved_path(root, recipe.data.manifest),
        split=recipe.data.split,
        verify_files=verify_media_files,
    )
    cache = VideoCachedDataset(
        _resolved_path(root, recipe.data.cache),
        expected_sample_ids=manifest.sample_ids,
        audit_on_open=audit_cache_on_open,
        verify_on_read=verify_cache_on_read,
    )
    audit_video_cache_against_manifest(cache, manifest)

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    distributed_context: DistributedTrainingContext | None = None
    if recipe.distributed.backend == "fsdp2":
        if resolved_device.type != "cuda":
            raise ValueError("native FSDP2 materialization requires device='cuda'")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device
    elif recipe.distributed.backend != "single":
        raise NotImplementedError(f"LTX materialization does not implement {recipe.distributed.backend!r}")

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import BuildPurpose, ComponentKey, ComponentKind
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry

    try:
        native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
        overrides = dict(checkpoint_overrides or {})
        if recipe.model.checkpoint != "default":
            if "model" in overrides:
                raise ValueError("model.checkpoint and checkpoint_overrides['model'] cannot both be set")
            overrides["model"] = str(_resolved_path(root, recipe.model.checkpoint))
        denoiser_key = ComponentKey(ComponentKind.DENOISER)
        dtype = torch_dtype(recipe.runtime.param_dtype)
        components = NativeDiffusionAssembler().build_components(
            native_recipe,
            purpose=BuildPurpose.TRAINING,
            policy=RuntimePolicy(
                device=resolved_device,
                dtype=dtype,
                attention=AttentionBackend.TORCH,
            ),
            checkpoint_overrides=overrides,
            component_keys=(denoiser_key,),
        )
        profile = _training_profile(recipe.model.recipe)
        _, first_frame_probability = _objective_config(recipe)
        adapter = build_cached_ltx_train_adapter(
            components,
            expected_latent_channels=int(native_recipe.options.get("latent_channels", 128)),
            temporal_compression=int(native_recipe.options.get("temporal_compression", 8)),
            spatial_compression=int(native_recipe.options.get("spatial_compression", 32)),
            first_frame_conditioning_probability=first_frame_probability,
            per_sample_first_frame_conditioning=profile.per_sample_first_frame_conditioning,
            causal_positions=profile.causal_positions,
            discrete_timesteps=profile.discrete_timesteps,
            gradient_checkpointing=recipe.runtime.activation_checkpoint == "full",
        )
        destination = _resolved_path(root, output_dir or recipe.run.output_dir)
        if distributed_context is not None:
            return build_ltx_fsdp2_session(
                recipe=recipe,
                adapter=adapter,
                dataset=cache,
                distributed_context=distributed_context,
                output_dir=destination,
                fused_adamw=fused_adamw,
                initialization_seed=initialization_seed,
            )
        return build_ltx_single_device_session(
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
    "apply_ltx_tuning",
    "audit_ltx_lora_targets",
    "build_ltx_flow_objective",
    "build_ltx_fsdp2_session",
    "build_ltx_single_device_session",
    "materialize_ltx_cached_training_session",
    "validate_ltx_cached_recipe",
]
