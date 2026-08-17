"""Recipe-driven immutable-cache Wan2.1 training sessions."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.distributed.fsdp import apply_fsdp2
from worldfoundry.training.distributed.parallel import (
    DistributedTrainingContext,
    ParallelPlan,
)
from worldfoundry.training.models.wan import WanTrainAdapter
from worldfoundry.training.objectives.flow_matching import (
    FlowMatchingConfig,
    FlowMatchingObjective,
)
from worldfoundry.training.optimization import build_lr_scheduler
from worldfoundry.training.recipes.spec import TrainingRecipe
from worldfoundry.training.tuning.peft import (
    PeftLoraApplication,
    apply_peft_lora_to_adapter,
)

from ..fsdp import FSDP2TrainEngine
from ..sessions.fsdp2 import FSDP2TrainingSession
from ..sessions.single_device import SingleDeviceTrainingSession
from ..single_device import SingleDeviceTrainEngine, build_adamw, trainable_parameters
from .cache import (
    audit_wan_cache_against_manifest,
    build_wan_cache_loader,
    validate_wan_cache_contract,
)
from .roles import seed_initialization, torch_dtype, validate_model_dtype

_FLOW_OPTIONS = frozenset(
    {
        "flow_shift",
        "logit_mean",
        "logit_std",
        "max_sigma",
        "min_sigma",
        "num_train_timesteps",
    }
)


def _validate_recipe_for_wan(
    recipe: TrainingRecipe,
    adapter: WanTrainAdapter,
    *,
    backend: str,
) -> None:
    if recipe.execution_owner != "worldfoundry-native":
        raise ValueError("cached Wan session requires WorldFoundry execution ownership")
    if recipe.distributed.backend != backend:
        raise ValueError(f"cached Wan session requires distributed.backend={backend!r}")
    if recipe.model.recipe != "wan2.1-t2v-1.3b":
        raise ValueError(f"cached Wan session cannot train model recipe {recipe.model.recipe!r}")
    if recipe.objective.type != "flow-matching":
        raise ValueError("cached Wan session requires objective.type='flow_matching'")
    if recipe.objective.prediction_type != adapter.prediction_type:
        raise ValueError("recipe prediction_type differs from the Wan adapter")
    if recipe.optimizer.type != "adamw":
        raise ValueError("native Wan session currently requires AdamW")
    if recipe.runtime.reduce_dtype != "float32":
        raise ValueError("native Wan objective reduction must use float32")
    if recipe.runtime.activation_checkpoint not in {"none", "full"}:
        raise ValueError("Wan activation_checkpoint must be 'none' or 'full'")
    if recipe.runtime.compile:
        raise ValueError("torch.compile is not supported by the native Wan training session")
    if not math.isclose(
        recipe.objective.conditioning_dropout,
        adapter.conditioning_dropout_probability,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "Wan T2V conditioning dropout is not part of the audited baseline; set objective.conditioning_dropout=0.0"
        )
    if adapter.gradient_checkpointing != (recipe.runtime.activation_checkpoint == "full"):
        raise ValueError("Wan adapter activation-checkpoint policy differs from the recipe")
    if recipe.data.max_latent_tokens_per_microbatch is None:
        raise ValueError("Wan training requires data.max_latent_tokens_per_microbatch")
    if recipe.data.tail_policy not in {"drop", "pad"}:
        raise ValueError("Wan token batching requires data.tail_policy='drop' or 'pad'")


def _flow_objective(recipe: TrainingRecipe, adapter: WanTrainAdapter) -> FlowMatchingObjective:
    options = dict(recipe.objective.options)
    unknown = sorted(set(options) - _FLOW_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported cached Wan objective options: {unknown}")
    config = FlowMatchingConfig(
        timestep_sampler=recipe.objective.timestep_sampler,
        **options,
    )
    if config.num_train_timesteps is not None and config.num_train_timesteps != adapter.num_train_timesteps:
        raise ValueError("objective num_train_timesteps differs from the Wan adapter")
    return FlowMatchingObjective(config)


def _apply_tuning(
    recipe: TrainingRecipe,
    adapter: WanTrainAdapter,
    *,
    initialization_seed: int | None,
) -> PeftLoraApplication | None:
    seed_initialization(initialization_seed)
    if recipe.tuning.mode == "lora":
        assert recipe.tuning.preset is not None
        assert recipe.tuning.rank is not None
        assert recipe.tuning.alpha is not None
        if recipe.tuning.preset != adapter.lora_target_preset:
            raise ValueError("recipe LoRA preset differs from the Wan adapter")
        return apply_peft_lora_to_adapter(
            adapter,
            preset=recipe.tuning.preset,
            rank=recipe.tuning.rank,
            alpha=recipe.tuning.alpha,
            dropout=recipe.tuning.dropout,
            modules_to_save=recipe.tuning.modules_to_save,
        )
    if recipe.tuning.mode == "partial":
        raise ValueError("partial Wan tuning needs an explicit parameter-selection policy")
    return None


def build_wan_single_device_session(
    *,
    recipe: TrainingRecipe,
    adapter: WanTrainAdapter,
    dataset: VideoCachedDataset,
    output_dir: str | Path | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> SingleDeviceTrainingSession:
    """Construct a strict token-budgeted single-device Wan session."""

    if not isinstance(recipe, TrainingRecipe):
        raise TypeError("recipe must be TrainingRecipe")
    if not isinstance(adapter, WanTrainAdapter):
        raise TypeError("adapter must be WanTrainAdapter")
    if not isinstance(dataset, VideoCachedDataset):
        raise TypeError("dataset must be VideoCachedDataset")
    _validate_recipe_for_wan(recipe, adapter, backend="single")
    expected_contract = validate_wan_cache_contract(recipe, adapter, dataset)
    expected_dtype = torch_dtype(recipe.runtime.param_dtype)
    validate_model_dtype(adapter, expected_dtype)
    application = _apply_tuning(
        recipe,
        adapter,
        initialization_seed=initialization_seed,
    )
    parameters = trainable_parameters(adapter.trainable_module)
    optimizer = build_adamw(
        parameters,
        learning_rate=recipe.optimizer.learning_rate,
        weight_decay=recipe.optimizer.weight_decay,
        betas=recipe.optimizer.betas,
        epsilon=recipe.optimizer.epsilon,
        fused=fused_adamw,
    )
    lr_scheduler = build_lr_scheduler(optimizer, recipe.scheduler)
    objective = _flow_objective(recipe, adapter)
    engine = SingleDeviceTrainEngine(
        adapter,
        objective,
        optimizer,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        autocast_dtype=None if expected_dtype is torch.float32 else expected_dtype,
        optimizer_step_end=None if lr_scheduler is None else lr_scheduler.step,
    )
    dataloader, sampler = build_wan_cache_loader(
        recipe=recipe,
        dataset=dataset,
        rank=0,
        world_size=1,
        default_pin_memory=engine.device.type == "cuda",
    )
    return SingleDeviceTrainingSession(
        recipe=recipe,
        engine=engine,
        dataloader=dataloader,
        output_dir=output_dir,
        peft_application=application,
        lr_scheduler=lr_scheduler,
        data_identity={
            "cache_schema": dataset.index.schema,
            "cache_index": dataset.index.to_dict(),
            "cache_contract": dict(expected_contract),
            "sample_ids": list(dataset.sample_ids),
            "sample_count": len(dataset),
            "latent_token_budget": sampler.max_latent_tokens,
            "token_sampler": {
                "sample_ids": list(sampler.sample_ids),
                "bucket_keys": [key.to_dict() for key in sampler.bucket_keys],
                "batch_contracts": [dict(value) for value in sampler.batch_contracts],
                "seed": sampler.seed,
                "shuffle": sampler.shuffle,
                "tail_policy": sampler.tail_policy,
                "rank": sampler.rank,
                "world_size": sampler.world_size,
            },
        },
    )


def build_wan_fsdp2_session(
    *,
    recipe: TrainingRecipe,
    adapter: WanTrainAdapter,
    dataset: VideoCachedDataset,
    distributed_context: DistributedTrainingContext,
    output_dir: str | Path | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> FSDP2TrainingSession:
    """Construct an FSDP2 Wan session using the active torchrun topology."""

    if not isinstance(recipe, TrainingRecipe):
        raise TypeError("recipe must be TrainingRecipe")
    if not isinstance(adapter, WanTrainAdapter):
        raise TypeError("adapter must be WanTrainAdapter")
    if not isinstance(dataset, VideoCachedDataset):
        raise TypeError("dataset must be VideoCachedDataset")
    if not isinstance(distributed_context, DistributedTrainingContext):
        raise TypeError("distributed_context must be DistributedTrainingContext")
    _validate_recipe_for_wan(recipe, adapter, backend="fsdp2")
    if distributed_context.device.type != "cuda":
        raise ValueError("Wan FSDP2 training currently requires CUDA")
    expected_contract = validate_wan_cache_contract(recipe, adapter, dataset)
    expected_dtype = torch_dtype(recipe.runtime.param_dtype)
    validate_model_dtype(adapter, expected_dtype)
    peft_application = _apply_tuning(
        recipe,
        adapter,
        initialization_seed=initialization_seed,
    )
    plan = ParallelPlan.resolve(
        recipe.distributed,
        world_size=distributed_context.world_size,
    )
    mesh = plan.build_device_mesh(distributed_context.device.type)
    fsdp_application = apply_fsdp2(
        adapter,
        plan=plan,
        mesh=mesh,
        param_dtype=expected_dtype,
        reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
    )
    parameters = trainable_parameters(adapter.trainable_module)
    optimizer = build_adamw(
        parameters,
        learning_rate=recipe.optimizer.learning_rate,
        weight_decay=recipe.optimizer.weight_decay,
        betas=recipe.optimizer.betas,
        epsilon=recipe.optimizer.epsilon,
        fused=fused_adamw,
    )
    lr_scheduler = build_lr_scheduler(optimizer, recipe.scheduler)
    objective = _flow_objective(recipe, adapter)
    engine = FSDP2TrainEngine(
        adapter,
        objective,
        optimizer,
        application=fsdp_application,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        autocast_dtype=None if expected_dtype is torch.float32 else expected_dtype,
        optimizer_step_end=None if lr_scheduler is None else lr_scheduler.step,
    )
    dataloader, sampler = build_wan_cache_loader(
        recipe=recipe,
        dataset=dataset,
        rank=distributed_context.rank,
        world_size=distributed_context.world_size,
        default_pin_memory=True,
    )
    return FSDP2TrainingSession(
        recipe=recipe,
        engine=engine,
        dataloader=dataloader,
        distributed_context=distributed_context,
        output_dir=output_dir,
        peft_application=peft_application,
        lr_scheduler=lr_scheduler,
        data_identity={
            "cache_schema": dataset.index.schema,
            "cache_index": dataset.index.to_dict(),
            "cache_contract": dict(expected_contract),
            "sample_ids": list(dataset.sample_ids),
            "sample_count": len(dataset),
            "latent_token_budget": sampler.max_latent_tokens,
            "token_sampler": {
                "sample_ids": list(sampler.sample_ids),
                "bucket_keys": [key.to_dict() for key in sampler.bucket_keys],
                "batch_contracts": [dict(value) for value in sampler.batch_contracts],
                "seed": sampler.seed,
                "shuffle": sampler.shuffle,
                "tail_policy": sampler.tail_policy,
                "rank": sampler.rank,
                "world_size": sampler.world_size,
            },
            "parallel_plan": plan.to_dict(),
            "fsdp2_application": fsdp_application.to_dict(),
        },
    )


def materialize_wan_cached_training_session(
    recipe: TrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    checkpoint_overrides: Mapping[str, object] | None = None,
    verify_media_files: bool = True,
    audit_cache_on_open: bool = True,
    verify_cache_on_read: bool = True,
    force_torch_attention: bool = True,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> SingleDeviceTrainingSession | FSDP2TrainingSession:
    """Load only the Wan DiT and construct an immutable-cache session."""

    if not isinstance(recipe, TrainingRecipe):
        raise TypeError("recipe must be TrainingRecipe")
    root = Path(base_dir).expanduser().resolve()
    if recipe.data.cache is None:
        raise ValueError("cached Wan training requires data.cache")
    cache_path = Path(recipe.data.cache)
    manifest_path = Path(recipe.data.manifest)
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    destination = Path(output_dir or recipe.run.output_dir)
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
    audit_wan_cache_against_manifest(cache, manifest)
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
        raise NotImplementedError(
            f"cached Wan materialization does not implement backend {recipe.distributed.backend!r}"
        )
    if force_torch_attention:
        for name in (
            "WORLDFOUNDRY_ATTENTION_IMPLEMENTATION",
            "WORLDFOUNDRY_ATTENTION_BACKEND",
        ):
            configured = os.environ.get(name)
            if configured not in {None, "torch"}:
                raise ValueError(f"correctness-first Wan training requires {name}=torch; got {configured!r}")
            os.environ[name] = "torch"

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
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
    from worldfoundry.training.models.wan import build_cached_wan_train_adapter

    try:
        native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
        overrides = dict(checkpoint_overrides or {})
        if recipe.model.checkpoint != "default":
            if "dit" in overrides:
                raise ValueError("model.checkpoint and checkpoint_overrides['dit'] cannot both be set")
            checkpoint_path = Path(recipe.model.checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = root / checkpoint_path
            overrides["dit"] = str(checkpoint_path)
        expected_dtype = torch_dtype(recipe.runtime.param_dtype)
        denoiser_key = ComponentKey(ComponentKind.DENOISER)
        components = NativeDiffusionAssembler().build_components(
            native_recipe,
            purpose=BuildPurpose.TRAINING,
            policy=RuntimePolicy(
                device=resolved_device,
                dtype=expected_dtype,
                attention=AttentionBackend.TORCH,
            ),
            checkpoint_overrides=overrides,
            component_options={denoiser_key: {"weight_dtype": expected_dtype}},
            component_keys=(denoiser_key,),
        )
        adapter = build_cached_wan_train_adapter(
            components,
            expected_latent_channels=int(native_recipe.options.get("latent_channels", 16)),
            temporal_compression=int(native_recipe.options.get("temporal_compression", 4)),
            spatial_compression=int(native_recipe.options.get("spatial_compression", 8)),
            num_train_timesteps=int(recipe.objective.options.get("num_train_timesteps", 1000)),
            gradient_checkpointing=recipe.runtime.activation_checkpoint == "full",
            attention_compatibility_mode=force_torch_attention,
        )
        if distributed_context is not None:
            return build_wan_fsdp2_session(
                recipe=recipe,
                adapter=adapter,
                dataset=cache,
                distributed_context=distributed_context,
                output_dir=destination,
                fused_adamw=fused_adamw,
                initialization_seed=initialization_seed,
            )
        return build_wan_single_device_session(
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
    "build_wan_fsdp2_session",
    "build_wan_single_device_session",
    "materialize_wan_cached_training_session",
]
