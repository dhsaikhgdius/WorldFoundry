"""Shared cache-backed execution for native video flow training."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.core.utils.torch_utils import set_seed_everywhere
from worldfoundry.training.api.contracts import TrainingObjective, TrainModelAdapter
from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.latent_token_sampler import LatentTokenBatchSampler
from worldfoundry.training.data.loader import build_stateful_dataloader
from worldfoundry.training.data.video_cache import (
    VideoCachedDataset,
    collate_video_cached_samples,
)
from worldfoundry.training.distributed.fsdp import apply_fsdp2
from worldfoundry.training.distributed.parallel import (
    DistributedTrainingContext,
    ParallelPlan,
)
from worldfoundry.training.optimization import build_lr_scheduler
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.spec import TrainingRecipe
from worldfoundry.training.tuning.application import AdapterApplication

from .fsdp import FSDP2TrainEngine
from .sessions.fsdp2 import FSDP2TrainingSession
from .sessions.single_device import SingleDeviceTrainingSession
from .single_device import SingleDeviceTrainEngine, build_adamw, trainable_parameters

VideoTrainingRecipe = TrainingRecipe | PostTrainingRecipe
VideoTuningFactory = Callable[[VideoTrainingRecipe, TrainModelAdapter], AdapterApplication | None]
VideoEmaFactory = Callable[[torch.nn.Module], object]
VideoEmaUpdate = Literal["microbatch", "optimizer-step"]

_LOADER_OPTIONS = frozenset(
    {
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "snapshot_every_n_steps",
    }
)
_COMMON_PREPROCESSING_OPTIONS = frozenset({"video_buckets", "bucket_policy", "decode"})


def torch_dtype(value: str) -> torch.dtype:
    """Resolve the closed dtype vocabulary used by training recipes."""

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def audit_video_cache_against_manifest(
    cache: VideoCachedDataset,
    manifest: TrainingManifestDataset,
) -> None:
    """Check that cached tensors still describe the selected source samples."""

    if cache.sample_ids != manifest.sample_ids:
        raise ValueError("video cache sample order differs from the selected manifest")
    for entry, sample in zip(cache.index.entries, manifest):
        source = entry.provenance
        if source.media_uri != sample.media.uri or source.prompt != sample.prompt:
            raise ValueError(f"video cache source differs for sample {sample.sample_id!r}")
        if (
            source.source_num_frames,
            source.source_height,
            source.source_width,
        ) != (sample.num_frames, sample.height, sample.width):
            raise ValueError(f"video cache geometry differs for sample {sample.sample_id!r}")
        if not math.isclose(source.source_fps, sample.fps, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"video cache fps differs for sample {sample.sample_id!r}")


def validate_video_cache_model(
    recipe: VideoTrainingRecipe,
    dataset: VideoCachedDataset,
) -> None:
    """Reject a cache created for another native model recipe."""

    for entry in dataset.index.entries:
        if entry.provenance.model_recipe != recipe.model.recipe:
            raise ValueError(
                f"cache entry {entry.sample_id!r} was created for "
                f"{entry.provenance.model_recipe!r}, not {recipe.model.recipe!r}"
            )


def build_cached_video_loader(
    *,
    recipe: VideoTrainingRecipe,
    dataset: VideoCachedDataset,
    rank: int,
    world_size: int,
    default_pin_memory: bool,
    consumed_data_options: frozenset[str] = _COMMON_PREPROCESSING_OPTIONS,
) -> tuple[object, LatentTokenBatchSampler]:
    """Build the stateful, token-budgeted loader shared by video families."""

    token_budget = recipe.data.max_latent_tokens_per_microbatch
    if token_budget is None:
        raise ValueError("cached video training requires data.max_latent_tokens_per_microbatch")
    options = dict(recipe.data.options)
    unknown = sorted(set(options) - _LOADER_OPTIONS - consumed_data_options)
    if unknown:
        raise ValueError(f"unsupported cached video data options: {unknown}")
    for name in consumed_data_options:
        options.pop(name, None)

    sampler = LatentTokenBatchSampler(
        dataset,
        max_latent_tokens=token_budget,
        seed=recipe.data.shuffle_seed,
        shuffle=recipe.data.shuffle,
        rank=rank,
        world_size=world_size,
        tail_policy=recipe.data.tail_policy,
    )
    workers = int(options.pop("num_workers", 0))
    pin_memory = options.pop("pin_memory", default_pin_memory)
    persistent_workers = options.pop("persistent_workers", False)
    if not isinstance(pin_memory, bool) or not isinstance(persistent_workers, bool):
        raise TypeError("pin_memory and persistent_workers must be bool values")
    prefetch_factor = options.pop("prefetch_factor", None)
    snapshot_every = int(options.pop("snapshot_every_n_steps", 1))
    if snapshot_every <= 0:
        raise ValueError("snapshot_every_n_steps must be positive")
    loader = build_stateful_dataloader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_video_cached_samples,
        num_workers=workers,
        worker_seed=recipe.data.shuffle_seed + rank,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=None if prefetch_factor is None else int(prefetch_factor),
        snapshot_every_n_steps=snapshot_every,
    )
    return loader, sampler


def _validate_execution(
    recipe: VideoTrainingRecipe,
    adapter: TrainModelAdapter,
    objective: TrainingObjective,
    *,
    backend: str,
) -> None:
    if recipe.execution_owner != "worldfoundry-native":
        raise ValueError("video training requires WorldFoundry execution ownership")
    if recipe.distributed.backend != backend:
        raise ValueError(f"video training requires distributed.backend={backend!r}")
    if recipe.optimizer.type != "adamw":
        raise ValueError("native video training requires AdamW")
    if recipe.runtime.reduce_dtype != "float32":
        raise ValueError("native video objective reduction must use float32")
    if recipe.runtime.compile:
        raise ValueError("torch.compile is not enabled for native video training")
    if adapter.prediction_type != objective.prediction_type:
        raise ValueError("video adapter and objective prediction types differ")


def _apply_tuning(
    recipe: VideoTrainingRecipe,
    adapter: TrainModelAdapter,
    *,
    tuning_factory: VideoTuningFactory | None,
    initialization_seed: int | None,
) -> AdapterApplication | None:
    if initialization_seed is not None:
        set_seed_everywhere(int(initialization_seed))
    if tuning_factory is not None:
        return tuning_factory(recipe, adapter)
    if recipe.tuning.mode != "full":
        raise ValueError("LoRA or partial tuning requires a model-family tuning factory")
    module = adapter.trainable_module
    if not isinstance(module, torch.nn.Module):
        raise TypeError("video adapter trainable_module must be an nn.Module")
    module.requires_grad_(True)
    return None


def _training_callbacks(
    module: torch.nn.Module,
    *,
    ema: object | None,
    ema_update: VideoEmaUpdate,
    lr_scheduler: object | None,
) -> tuple[Callable[[], None] | None, Callable[[], None] | None]:
    if ema_update not in {"microbatch", "optimizer-step"}:
        raise ValueError("ema_update must be 'microbatch' or 'optimizer-step'")

    train_batch_end: Callable[[], None] | None = None
    if ema is not None and ema_update == "microbatch":

        def update_ema_after_microbatch() -> None:
            ema(module)  # type: ignore[operator]

        train_batch_end = update_ema_after_microbatch

    optimizer_callbacks: list[Callable[[], None]] = []
    if lr_scheduler is not None:
        optimizer_callbacks.append(lr_scheduler.step)  # type: ignore[arg-type]
    if ema is not None and ema_update == "optimizer-step":

        def update_ema_after_optimizer_step() -> None:
            ema(module)  # type: ignore[operator]

        optimizer_callbacks.append(update_ema_after_optimizer_step)

    def optimizer_step_end() -> None:
        for callback in optimizer_callbacks:
            callback()

    return train_batch_end, optimizer_step_end if optimizer_callbacks else None


def _data_identity(
    dataset: VideoCachedDataset,
    sampler: LatentTokenBatchSampler,
    cache_contract: Mapping[str, object],
    *,
    parallel_plan: ParallelPlan | None = None,
    fsdp_application: object | None = None,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "cache_schema": dataset.index.schema,
        "cache_index": dataset.index.to_dict(),
        "cache_contract": dict(cache_contract),
        "sample_ids": list(dataset.sample_ids),
        "sample_count": len(dataset),
        "latent_token_budget": sampler.max_latent_tokens,
        "token_sampler": {
            "seed": sampler.seed,
            "shuffle": sampler.shuffle,
            "tail_policy": sampler.tail_policy,
            "rank": sampler.rank,
            "world_size": sampler.world_size,
        },
    }
    if parallel_plan is not None:
        identity["parallel_plan"] = parallel_plan.to_dict()
    if fsdp_application is not None:
        identity["fsdp2_application"] = fsdp_application.to_dict()
    return identity


def build_cached_video_flow_single_device_session(
    *,
    recipe: VideoTrainingRecipe,
    adapter: TrainModelAdapter,
    dataset: VideoCachedDataset,
    objective: TrainingObjective,
    cache_contract: Mapping[str, object],
    output_dir: str | Path | None = None,
    tuning_factory: VideoTuningFactory | None = None,
    ema_factory: VideoEmaFactory | None = None,
    export_ema: bool = False,
    ema_update: VideoEmaUpdate = "optimizer-step",
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
    consumed_data_options: frozenset[str] = _COMMON_PREPROCESSING_OPTIONS,
) -> SingleDeviceTrainingSession:
    """Construct one cache-backed native video training session.

    ``ema_update`` defaults to ``"optimizer-step"``: shadows absorb
    parameters exactly once per applied optimizer step, which is what
    counter-based decay schedules (PowerEMA, LitEma) assume.  Families that
    need Lightning ``on_train_batch_end`` author parity must opt in to
    ``"microbatch"`` explicitly; under gradient accumulation that mode
    updates the EMA after every microbatch backward (absorbing unchanged
    parameters N-1 times per step) and inflates ``num_updates``.
    """

    _validate_execution(recipe, adapter, objective, backend="single")
    validate_video_cache_model(recipe, dataset)
    peft_application = _apply_tuning(
        recipe,
        adapter,
        tuning_factory=tuning_factory,
        initialization_seed=initialization_seed,
    )
    parameters = trainable_parameters(adapter.trainable_module)
    ema = None if ema_factory is None else ema_factory(adapter.trainable_module)
    if ema is not None and not callable(ema):
        raise TypeError("video EMA must be callable")
    optimizer = build_adamw(
        parameters,
        learning_rate=recipe.optimizer.learning_rate,
        weight_decay=recipe.optimizer.weight_decay,
        betas=recipe.optimizer.betas,
        epsilon=recipe.optimizer.epsilon,
        fused=fused_adamw,
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        recipe.scheduler if isinstance(recipe, TrainingRecipe) else None,
    )
    train_batch_end, optimizer_step_end = _training_callbacks(
        adapter.trainable_module,
        ema=ema,
        ema_update=ema_update,
        lr_scheduler=lr_scheduler,
    )
    parameter_dtype = torch_dtype(recipe.runtime.param_dtype)
    engine = SingleDeviceTrainEngine(
        adapter,
        objective,
        optimizer,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        autocast_dtype=None if parameter_dtype is torch.float32 else parameter_dtype,
        train_batch_end=train_batch_end,
        optimizer_step_end=optimizer_step_end,
    )
    loader, sampler = build_cached_video_loader(
        recipe=recipe,
        dataset=dataset,
        rank=0,
        world_size=1,
        default_pin_memory=engine.device.type == "cuda",
        consumed_data_options=consumed_data_options,
    )
    return SingleDeviceTrainingSession(
        recipe=recipe,
        engine=engine,
        dataloader=loader,
        output_dir=output_dir,
        adapter_application=peft_application,
        data_identity=_data_identity(dataset, sampler, cache_contract),
        lr_scheduler=lr_scheduler,
        ema=ema,
        export_ema=export_ema,
    )


def build_cached_video_flow_fsdp2_session(
    *,
    recipe: VideoTrainingRecipe,
    adapter: TrainModelAdapter,
    dataset: VideoCachedDataset,
    objective: TrainingObjective,
    cache_contract: Mapping[str, object],
    distributed_context: DistributedTrainingContext,
    output_dir: str | Path | None = None,
    tuning_factory: VideoTuningFactory | None = None,
    ema_factory: VideoEmaFactory | None = None,
    export_ema: bool = False,
    ema_update: VideoEmaUpdate = "optimizer-step",
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
    master_parameter_dtype: torch.dtype | None = None,
    consumed_data_options: frozenset[str] = _COMMON_PREPROCESSING_OPTIONS,
) -> FSDP2TrainingSession:
    """Construct the same video training state machine on an FSDP2 topology.

    See ``build_cached_video_flow_single_device_session`` for the
    ``ema_update`` default semantics.
    """

    _validate_execution(recipe, adapter, objective, backend="fsdp2")
    validate_video_cache_model(recipe, dataset)
    if distributed_context.device.type != "cuda":
        raise ValueError("native FSDP2 video training requires CUDA")
    peft_application = _apply_tuning(
        recipe,
        adapter,
        tuning_factory=tuning_factory,
        initialization_seed=initialization_seed,
    )
    plan = ParallelPlan.resolve(recipe.distributed, world_size=distributed_context.world_size)
    mesh = plan.build_device_mesh(distributed_context.device.type)
    parameter_dtype = torch_dtype(recipe.runtime.param_dtype)
    fsdp_application = apply_fsdp2(
        adapter,
        plan=plan,
        mesh=mesh,
        param_dtype=parameter_dtype,
        reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
        master_parameter_dtype=master_parameter_dtype,
    )
    ema = None if ema_factory is None else ema_factory(adapter.trainable_module)
    if ema is not None and not callable(ema):
        raise TypeError("video EMA must be callable")
    optimizer = build_adamw(
        trainable_parameters(adapter.trainable_module),
        learning_rate=recipe.optimizer.learning_rate,
        weight_decay=recipe.optimizer.weight_decay,
        betas=recipe.optimizer.betas,
        epsilon=recipe.optimizer.epsilon,
        fused=fused_adamw,
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        recipe.scheduler if isinstance(recipe, TrainingRecipe) else None,
    )
    train_batch_end, optimizer_step_end = _training_callbacks(
        adapter.trainable_module,
        ema=ema,
        ema_update=ema_update,
        lr_scheduler=lr_scheduler,
    )
    engine = FSDP2TrainEngine(
        adapter,
        objective,
        optimizer,
        application=fsdp_application,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        autocast_dtype=None if parameter_dtype is torch.float32 else parameter_dtype,
        train_batch_end=train_batch_end,
        optimizer_step_end=optimizer_step_end,
    )
    loader, sampler = build_cached_video_loader(
        recipe=recipe,
        dataset=dataset,
        rank=distributed_context.rank,
        world_size=distributed_context.world_size,
        default_pin_memory=True,
        consumed_data_options=consumed_data_options,
    )
    return FSDP2TrainingSession(
        recipe=recipe,
        engine=engine,
        dataloader=loader,
        distributed_context=distributed_context,
        output_dir=output_dir,
        adapter_application=peft_application,
        data_identity=_data_identity(
            dataset,
            sampler,
            cache_contract,
            parallel_plan=plan,
            fsdp_application=fsdp_application,
        ),
        lr_scheduler=lr_scheduler,
        ema=ema,
        export_ema=export_ema,
    )


__all__ = [
    "VideoTuningFactory",
    "VideoEmaFactory",
    "VideoEmaUpdate",
    "VideoTrainingRecipe",
    "audit_video_cache_against_manifest",
    "build_cached_video_flow_fsdp2_session",
    "build_cached_video_flow_single_device_session",
    "build_cached_video_loader",
    "torch_dtype",
    "validate_video_cache_model",
]
