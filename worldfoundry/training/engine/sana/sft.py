"""Recipe-driven cached SANA training session construction."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.loader import build_stateful_dataloader
from worldfoundry.training.data.sampler import DeterministicDistributedSampler
from worldfoundry.training.data.sana_cache import (
    SanaCachedDataset,
    collate_sana_cached_samples,
)
from worldfoundry.training.distributed.fsdp import apply_fsdp2
from worldfoundry.training.distributed.parallel import (
    DistributedTrainingContext,
    ParallelPlan,
)
from worldfoundry.training.models.sana import SanaTrainAdapter
from worldfoundry.training.objectives.flow_matching import (
    FlowMatchingConfig,
    FlowMatchingObjective,
)
from worldfoundry.training.recipes.spec import TrainingRecipe
from worldfoundry.training.tuning.peft import (
    PeftLoraApplication,
    apply_peft_lora_to_adapter,
)

from ..fsdp import FSDP2TrainEngine
from ..sessions.fsdp2 import FSDP2TrainingSession
from ..sessions.single_device import SingleDeviceTrainingSession
from ..single_device import SingleDeviceTrainEngine, build_adamw, trainable_parameters
from .cache import audit_sana_cache_against_manifest, validate_sana_cache_contract

_CACHE_LOADER_OPTIONS = frozenset(
    {
        "microbatch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "snapshot_every_n_steps",
    }
)
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


def _strict_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _torch_dtype(value: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def _seed_adapter_initialization(seed: int | None) -> None:
    if seed is None:
        return
    if isinstance(seed, bool):
        raise TypeError("initialization_seed must be an integer, not bool")
    resolved = int(seed) % (2**63 - 1)
    random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)


def _validate_recipe_for_sana(
    recipe: TrainingRecipe,
    adapter: SanaTrainAdapter,
    *,
    backend: str,
) -> None:
    if recipe.execution_owner != "worldfoundry-native":
        raise ValueError("cached SANA session requires WorldFoundry execution ownership")
    if recipe.distributed.backend != backend:
        raise ValueError(f"cached SANA session requires distributed.backend={backend!r}")
    if not recipe.model.recipe.startswith("sana-"):
        raise ValueError(f"cached SANA session cannot train model recipe {recipe.model.recipe!r}")
    if recipe.objective.type != "flow-matching":
        raise ValueError("cached SANA session currently requires objective.type='flow_matching'")
    if recipe.objective.prediction_type != adapter.prediction_type:
        raise ValueError("recipe prediction_type differs from the SANA adapter")
    if recipe.optimizer.type != "adamw":
        raise ValueError("native SANA session currently requires AdamW")
    if recipe.runtime.reduce_dtype != "float32":
        raise ValueError("native SANA objective reduction must use float32")
    if recipe.runtime.activation_checkpoint != "none":
        raise ValueError(
            "native SANA activation checkpointing is not implemented; set runtime.activation_checkpoint='none'"
        )
    if recipe.runtime.compile:
        raise ValueError("torch.compile is not supported by the native SANA training session")
    if not math.isclose(
        recipe.objective.conditioning_dropout,
        adapter.conditioning_dropout_probability,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "recipe conditioning_dropout differs from the denoiser-owned probability: "
            f"{recipe.objective.conditioning_dropout} vs "
            f"{adapter.conditioning_dropout_probability}"
        )


def _flow_objective(recipe: TrainingRecipe, adapter: SanaTrainAdapter) -> FlowMatchingObjective:
    options = dict(recipe.objective.options)
    unknown = sorted(set(options) - _FLOW_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported cached SANA objective options: {unknown}")
    config = FlowMatchingConfig(
        timestep_sampler=recipe.objective.timestep_sampler,
        **options,
    )
    if config.num_train_timesteps is not None and config.num_train_timesteps != adapter.num_train_timesteps:
        raise ValueError("objective num_train_timesteps differs from the SANA adapter")
    return FlowMatchingObjective(config)


def build_sana_single_device_session(
    *,
    recipe: TrainingRecipe,
    adapter: SanaTrainAdapter,
    dataset: SanaCachedDataset,
    output_dir: str | Path | None = None,
    microbatch_size: int | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> SingleDeviceTrainingSession:
    """Construct a strict single-device session from loaded SANA/cache objects."""

    if not isinstance(recipe, TrainingRecipe):
        raise TypeError("recipe must be TrainingRecipe")
    if not isinstance(adapter, SanaTrainAdapter):
        raise TypeError("adapter must be SanaTrainAdapter")
    if not isinstance(dataset, SanaCachedDataset):
        raise TypeError("dataset must be SanaCachedDataset")
    _validate_recipe_for_sana(recipe, adapter, backend="single")

    loader_options = dict(recipe.data.options)
    unknown_loader_options = sorted(set(loader_options) - _CACHE_LOADER_OPTIONS)
    if unknown_loader_options:
        raise ValueError(f"unsupported cached SANA data options: {unknown_loader_options}")
    configured_batch_size = _positive_int(
        loader_options.pop("microbatch_size", 1),
        field_name="data.options.microbatch_size",
    )
    if microbatch_size is not None:
        configured_batch_size = _positive_int(microbatch_size, field_name="microbatch_size")
    expected_contract = validate_sana_cache_contract(
        recipe,
        adapter,
        dataset,
        microbatch_size=configured_batch_size,
    )

    expected_dtype = _torch_dtype(recipe.runtime.param_dtype)
    base_dtypes = {
        parameter.dtype for parameter in adapter.trainable_module.parameters() if parameter.is_floating_point()
    }
    if base_dtypes != {expected_dtype}:
        raise ValueError(
            "loaded SANA parameter dtype differs from runtime.param_dtype: "
            f"loaded={sorted(map(str, base_dtypes))}, expected={expected_dtype}"
        )

    _seed_adapter_initialization(initialization_seed)
    application: PeftLoraApplication | None = None
    if recipe.tuning.mode == "lora":
        assert recipe.tuning.preset is not None
        assert recipe.tuning.rank is not None
        assert recipe.tuning.alpha is not None
        if recipe.tuning.preset != adapter.lora_target_preset:
            raise ValueError("recipe LoRA preset differs from the SANA adapter")
        application = apply_peft_lora_to_adapter(
            adapter,
            preset=recipe.tuning.preset,
            rank=recipe.tuning.rank,
            alpha=recipe.tuning.alpha,
            dropout=recipe.tuning.dropout,
            modules_to_save=recipe.tuning.modules_to_save,
        )
    elif recipe.tuning.mode == "partial":
        raise ValueError("partial SANA tuning needs an explicit parameter-selection policy")

    objective = _flow_objective(recipe, adapter)
    parameters = trainable_parameters(adapter.trainable_module)
    optimizer = build_adamw(
        parameters,
        learning_rate=recipe.optimizer.learning_rate,
        weight_decay=recipe.optimizer.weight_decay,
        betas=recipe.optimizer.betas,
        epsilon=recipe.optimizer.epsilon,
        fused=fused_adamw,
    )
    autocast_dtype = None if expected_dtype is torch.float32 else expected_dtype
    engine = SingleDeviceTrainEngine(
        adapter,
        objective,
        optimizer,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        autocast_dtype=autocast_dtype,
    )

    sampler = DeterministicDistributedSampler(
        dataset,
        dataset_digest=dataset.dataset_digest,
        seed=recipe.data.shuffle_seed,
        shuffle=recipe.data.shuffle,
        rank=0,
        world_size=1,
        tail_policy=recipe.data.tail_policy,
    )
    workers = int(loader_options.pop("num_workers", 0))
    pin_memory = _strict_bool(
        loader_options.pop("pin_memory", engine.device.type == "cuda"),
        field_name="data.options.pin_memory",
    )
    persistent_workers = _strict_bool(
        loader_options.pop("persistent_workers", False),
        field_name="data.options.persistent_workers",
    )
    prefetch_factor = loader_options.pop("prefetch_factor", None)
    snapshot_every = _positive_int(
        loader_options.pop("snapshot_every_n_steps", 1),
        field_name="data.options.snapshot_every_n_steps",
    )
    dataloader = build_stateful_dataloader(
        dataset,
        sampler,
        batch_size=configured_batch_size,
        collate_fn=collate_sana_cached_samples,
        num_workers=workers,
        worker_seed=recipe.data.shuffle_seed,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=None if prefetch_factor is None else int(prefetch_factor),
        snapshot_every_n_steps=snapshot_every,
    )
    return SingleDeviceTrainingSession(
        recipe=recipe,
        engine=engine,
        dataloader=dataloader,
        output_dir=output_dir,
        peft_application=application,
        data_identity={
            "cache_schema": dataset.index.schema,
            "cache_index_sha256": dataset.index_sha256,
            "cache_contract_sha256": expected_contract,
            "dataset_digest": dataset.dataset_digest,
            "sample_count": len(dataset),
        },
    )


def build_sana_fsdp2_session(
    *,
    recipe: TrainingRecipe,
    adapter: SanaTrainAdapter,
    dataset: SanaCachedDataset,
    distributed_context: DistributedTrainingContext,
    output_dir: str | Path | None = None,
    microbatch_size: int | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> FSDP2TrainingSession:
    """Construct an FSDP2 SANA session using the active torchrun topology."""

    if not isinstance(recipe, TrainingRecipe):
        raise TypeError("recipe must be TrainingRecipe")
    if not isinstance(adapter, SanaTrainAdapter):
        raise TypeError("adapter must be SanaTrainAdapter")
    if not isinstance(dataset, SanaCachedDataset):
        raise TypeError("dataset must be SanaCachedDataset")
    if not isinstance(distributed_context, DistributedTrainingContext):
        raise TypeError("distributed_context must be DistributedTrainingContext")
    _validate_recipe_for_sana(recipe, adapter, backend="fsdp2")
    if distributed_context.device.type != "cuda":
        raise ValueError("SANA FSDP2 training currently requires CUDA")
    if distributed_context.world_size > 1 and recipe.data.tail_policy == "uneven":
        raise ValueError(
            "multi-rank FSDP2 training requires data.tail_policy='drop' or 'pad' "
            "so every rank executes the same number of collectives"
        )

    loader_options = dict(recipe.data.options)
    unknown_loader_options = sorted(set(loader_options) - _CACHE_LOADER_OPTIONS)
    if unknown_loader_options:
        raise ValueError(f"unsupported cached SANA data options: {unknown_loader_options}")
    configured_batch_size = _positive_int(
        loader_options.pop("microbatch_size", 1),
        field_name="data.options.microbatch_size",
    )
    if microbatch_size is not None:
        configured_batch_size = _positive_int(microbatch_size, field_name="microbatch_size")
    expected_contract = validate_sana_cache_contract(
        recipe,
        adapter,
        dataset,
        microbatch_size=configured_batch_size,
    )

    expected_dtype = _torch_dtype(recipe.runtime.param_dtype)
    base_dtypes = {
        parameter.dtype for parameter in adapter.trainable_module.parameters() if parameter.is_floating_point()
    }
    if base_dtypes != {expected_dtype}:
        raise ValueError(
            "loaded SANA parameter dtype differs from runtime.param_dtype: "
            f"loaded={sorted(map(str, base_dtypes))}, expected={expected_dtype}"
        )

    # Every rank must inject identical adapter parameters before sharding.
    _seed_adapter_initialization(initialization_seed)
    peft_application: PeftLoraApplication | None = None
    if recipe.tuning.mode == "lora":
        assert recipe.tuning.preset is not None
        assert recipe.tuning.rank is not None
        assert recipe.tuning.alpha is not None
        if recipe.tuning.preset != adapter.lora_target_preset:
            raise ValueError("recipe LoRA preset differs from the SANA adapter")
        peft_application = apply_peft_lora_to_adapter(
            adapter,
            preset=recipe.tuning.preset,
            rank=recipe.tuning.rank,
            alpha=recipe.tuning.alpha,
            dropout=recipe.tuning.dropout,
            modules_to_save=recipe.tuning.modules_to_save,
        )
    elif recipe.tuning.mode == "partial":
        raise ValueError("partial SANA tuning needs an explicit parameter-selection policy")

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
        reduce_dtype=_torch_dtype(recipe.runtime.reduce_dtype),
    )
    # PyTorch requires optimizer construction after fully_shard so it owns the
    # active DTensor parameters instead of the pre-sharding Parameter objects.
    parameters = trainable_parameters(adapter.trainable_module)
    optimizer = build_adamw(
        parameters,
        learning_rate=recipe.optimizer.learning_rate,
        weight_decay=recipe.optimizer.weight_decay,
        betas=recipe.optimizer.betas,
        epsilon=recipe.optimizer.epsilon,
        fused=fused_adamw,
    )
    objective = _flow_objective(recipe, adapter)
    autocast_dtype = None if expected_dtype is torch.float32 else expected_dtype
    engine = FSDP2TrainEngine(
        adapter,
        objective,
        optimizer,
        application=fsdp_application,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        autocast_dtype=autocast_dtype,
    )

    sampler = DeterministicDistributedSampler(
        dataset,
        dataset_digest=dataset.dataset_digest,
        seed=recipe.data.shuffle_seed,
        shuffle=recipe.data.shuffle,
        rank=distributed_context.rank,
        world_size=distributed_context.world_size,
        tail_policy=recipe.data.tail_policy,
    )
    workers = int(loader_options.pop("num_workers", 0))
    pin_memory = _strict_bool(
        loader_options.pop("pin_memory", True),
        field_name="data.options.pin_memory",
    )
    persistent_workers = _strict_bool(
        loader_options.pop("persistent_workers", False),
        field_name="data.options.persistent_workers",
    )
    prefetch_factor = loader_options.pop("prefetch_factor", None)
    snapshot_every = _positive_int(
        loader_options.pop("snapshot_every_n_steps", 1),
        field_name="data.options.snapshot_every_n_steps",
    )
    dataloader = build_stateful_dataloader(
        dataset,
        sampler,
        batch_size=configured_batch_size,
        collate_fn=collate_sana_cached_samples,
        num_workers=workers,
        worker_seed=recipe.data.shuffle_seed + distributed_context.rank,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=None if prefetch_factor is None else int(prefetch_factor),
        snapshot_every_n_steps=snapshot_every,
    )
    return FSDP2TrainingSession(
        recipe=recipe,
        engine=engine,
        dataloader=dataloader,
        distributed_context=distributed_context,
        output_dir=output_dir,
        peft_application=peft_application,
        data_identity={
            "cache_schema": dataset.index.schema,
            "cache_index_sha256": dataset.index_sha256,
            "cache_contract_sha256": expected_contract,
            "dataset_digest": dataset.dataset_digest,
            "sample_count": len(dataset),
            "parallel_plan": plan.to_dict(),
            "fsdp2_application_digest": fsdp_application.digest,
        },
    )


def materialize_sana_cached_training_session(
    recipe: TrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    checkpoint_overrides: Mapping[str, object] | None = None,
    verify_media_hashes: bool = True,
    audit_cache_on_open: bool = True,
    verify_cache_on_read: bool = True,
    disable_xformers: bool = True,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> SingleDeviceTrainingSession | FSDP2TrainingSession:
    """Load only the SANA denoiser and construct a cache-backed session."""

    if not isinstance(recipe, TrainingRecipe):
        raise TypeError("recipe must be TrainingRecipe")
    root = Path(base_dir).expanduser().resolve()
    if recipe.data.cache is None:
        raise ValueError("cached SANA training requires data.cache")
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
        verify_files=True,
        verify_hashes=verify_media_hashes,
    )
    cache = SanaCachedDataset(
        cache_path,
        expected_dataset_digest=manifest.dataset_digest,
        audit_on_open=audit_cache_on_open,
        verify_on_read=verify_cache_on_read,
    )
    audit_sana_cache_against_manifest(cache, manifest)

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    distributed_context: DistributedTrainingContext | None = None
    if recipe.distributed.backend == "fsdp2":
        if resolved_device.type != "cuda":
            raise ValueError("native FSDP2 materialization currently requires device='cuda'")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device
    elif recipe.distributed.backend != "single":
        raise NotImplementedError(
            f"cached SANA materialization does not implement backend {recipe.distributed.backend!r}"
        )
    if disable_xformers:
        configured = os.environ.get("DISABLE_XFORMERS")
        if configured not in {None, "1"}:
            raise ValueError("correctness-first SANA training requires DISABLE_XFORMERS=1")
        os.environ["DISABLE_XFORMERS"] = "1"

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentKey,
        ComponentKind,
    )
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.registry import (
        default_native_diffusion_registry,
    )
    from worldfoundry.training.models.sana import build_cached_sana_train_adapter

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
        denoiser_key = ComponentKey(ComponentKind.DENOISER)
        components = NativeDiffusionAssembler().build_components(
            native_recipe,
            purpose=BuildPurpose.TRAINING,
            policy=RuntimePolicy(
                device=resolved_device,
                dtype=_torch_dtype(recipe.runtime.param_dtype),
                attention=AttentionBackend.TORCH,
            ),
            checkpoint_overrides=overrides,
            component_keys=(denoiser_key,),
        )
        adapter = build_cached_sana_train_adapter(
            components,
            expected_latent_channels=int(native_recipe.options.get("latent_channels", 32)),
            spatial_compression=int(native_recipe.options.get("spatial_compression", 32)),
            num_train_timesteps=int(recipe.objective.options.get("num_train_timesteps", 1000)),
        )
        if distributed_context is not None:
            return build_sana_fsdp2_session(
                recipe=recipe,
                adapter=adapter,
                dataset=cache,
                distributed_context=distributed_context,
                output_dir=destination,
                fused_adamw=fused_adamw,
                initialization_seed=initialization_seed,
            )
        return build_sana_single_device_session(
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
    "build_sana_fsdp2_session",
    "build_sana_single_device_session",
    "materialize_sana_cached_training_session",
]
