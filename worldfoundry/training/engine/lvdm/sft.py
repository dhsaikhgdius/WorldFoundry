"""Executable cached-latent training for the released LVDM short UNet."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from torch.distributed.tensor import DTensor, distribute_tensor

from worldfoundry.core.nn.ema import LitEma
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
from worldfoundry.training.models.lvdm import LVDMUnconditionalTrainAdapter
from worldfoundry.training.objectives.classic_diffusion import (
    ClassicDiffusionConfig,
    ClassicDiffusionObjective,
)
from worldfoundry.training.recipes.spec import TrainingRecipe

from .cache import LVDM_SHORT_MODEL_RECIPE, validate_lvdm_short_cache_contract

_OBJECTIVE_OPTIONS = frozenset({"num_train_timesteps", "beta_start", "beta_end", "loss_type"})
_DATA_OPTIONS = frozenset({"video_buckets", "bucket_policy", "decode", "precomputed_tensors"})


def _checkpoint_payload(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    nested = state_dict.get("state_dict")
    return nested if isinstance(nested, Mapping) else state_dict


def _lvdm_ema_checkpoint_state(state_dict: Mapping[str, object]) -> dict[str, object]:
    payload = _checkpoint_payload(state_dict)
    prefix = "model_ema."
    return {key[len(prefix) :]: value for key, value in payload.items() if key.startswith(prefix)}


def _ema_value_like(value: object, target: torch.Tensor) -> object:
    if not isinstance(value, torch.Tensor):
        return value
    if isinstance(target, DTensor) and not isinstance(value, DTensor):
        full = value.to(device=target.device, dtype=target.dtype)
        return distribute_tensor(full, target.device_mesh, target.placements)
    return value.to(device=target.device, dtype=target.dtype)


def _build_lvdm_ema(
    module: torch.nn.Module,
    checkpoint_state: Mapping[str, object] | None,
) -> LitEma:
    ema = LitEma(module, decay=0.9999, use_num_upates=True)
    if not checkpoint_state:
        return ema
    target = ema.state_dict()
    converted: dict[str, object] = {}
    for name, target_value in target.items():
        candidates = (name,) if name in {"decay", "num_updates"} else (f"diffusion_model{name}", name)
        source_name = next((candidate for candidate in candidates if candidate in checkpoint_state), None)
        if source_name is None:
            raise KeyError(f"LVDM checkpoint EMA is missing {name!r}")
        converted[name] = _ema_value_like(checkpoint_state[source_name], target_value)
    ema.load_state_dict(converted, strict=True)
    return ema


def build_lvdm_short_objective(recipe: TrainingRecipe) -> ClassicDiffusionObjective:
    options = dict(recipe.objective.options)
    unknown = sorted(set(options) - _OBJECTIVE_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported LVDM objective options: {unknown}")
    config = ClassicDiffusionConfig(
        num_train_timesteps=int(options.pop("num_train_timesteps", 1000)),
        beta_start=float(options.pop("beta_start", 0.0015)),
        beta_end=float(options.pop("beta_end", 0.0155)),
        prediction_type="epsilon",
        loss_type=str(options.pop("loss_type", "l1")),
    )
    return ClassicDiffusionObjective(config)


def validate_lvdm_short_recipe(recipe: TrainingRecipe, *, backend: str) -> None:
    if recipe.model.recipe != LVDM_SHORT_MODEL_RECIPE:
        raise ValueError(f"LVDM short training cannot train {recipe.model.recipe!r}")
    if recipe.distributed.backend != backend:
        raise ValueError(f"LVDM short training requires distributed.backend={backend!r}")
    if recipe.tuning.mode != "full":
        raise ValueError("the released LVDM short trainer performs full-parameter tuning")
    if recipe.objective.type != "classic-diffusion" or recipe.objective.prediction_type != "epsilon":
        raise ValueError("LVDM short training requires epsilon classic diffusion")
    if recipe.objective.timestep_sampler != "uniform":
        raise ValueError("LVDM short training samples timesteps uniformly")
    if recipe.objective.conditioning_dropout != 0.0:
        raise ValueError("unconditional LVDM short training has no conditioning dropout")
    if recipe.runtime.activation_checkpoint not in {"none", "full"}:
        raise ValueError("LVDM activation_checkpoint must be 'none' or 'full'")
    if recipe.data.max_latent_tokens_per_microbatch is None:
        raise ValueError("cached LVDM training requires a latent token budget")
    build_lvdm_short_objective(recipe)


def build_lvdm_short_single_device_session(
    *,
    recipe: TrainingRecipe,
    adapter: LVDMUnconditionalTrainAdapter,
    dataset: VideoCachedDataset,
    output_dir: str | Path | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
    ema_checkpoint_state: Mapping[str, object] | None = None,
) -> SingleDeviceTrainingSession:
    validate_lvdm_short_recipe(recipe, backend="single")
    contract = validate_lvdm_short_cache_contract(recipe, adapter, dataset)
    return build_cached_video_flow_single_device_session(
        recipe=recipe,
        adapter=adapter,
        dataset=dataset,
        objective=build_lvdm_short_objective(recipe),
        cache_contract=contract,
        output_dir=output_dir,
        fused_adamw=fused_adamw,
        initialization_seed=initialization_seed,
        consumed_data_options=_DATA_OPTIONS,
        ema_factory=lambda module: _build_lvdm_ema(module, ema_checkpoint_state),
        export_ema=True,
        # Author parity with the released Lightning trainer: LitEma runs in
        # on_train_batch_end, i.e. once per microbatch, and its num_updates
        # counter reflects microbatches rather than optimizer steps.
        ema_update="microbatch",
    )


def build_lvdm_short_fsdp2_session(
    *,
    recipe: TrainingRecipe,
    adapter: LVDMUnconditionalTrainAdapter,
    dataset: VideoCachedDataset,
    distributed_context: DistributedTrainingContext,
    output_dir: str | Path | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
    ema_checkpoint_state: Mapping[str, object] | None = None,
) -> FSDP2TrainingSession:
    validate_lvdm_short_recipe(recipe, backend="fsdp2")
    contract = validate_lvdm_short_cache_contract(recipe, adapter, dataset)
    return build_cached_video_flow_fsdp2_session(
        recipe=recipe,
        adapter=adapter,
        dataset=dataset,
        objective=build_lvdm_short_objective(recipe),
        cache_contract=contract,
        distributed_context=distributed_context,
        output_dir=output_dir,
        fused_adamw=fused_adamw,
        initialization_seed=initialization_seed,
        consumed_data_options=_DATA_OPTIONS,
        ema_factory=lambda module: _build_lvdm_ema(module, ema_checkpoint_state),
        export_ema=True,
        # Author parity: see build_lvdm_short_single_device_session.
        ema_update="microbatch",
    )


def _resolved_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _short_unet_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    state_dict = _checkpoint_payload(state_dict)
    prefix = "model.diffusion_model."
    converted = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    return converted or state_dict


def materialize_lvdm_short_training_session(
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
    """Load an official short-UNet checkpoint and build a native session."""

    validate_lvdm_short_recipe(recipe, backend=recipe.distributed.backend)
    if recipe.data.cache is None:
        raise ValueError("cached LVDM short training requires data.cache")
    root = Path(base_dir).expanduser().resolve()
    manifest = TrainingManifestDataset.from_file(
        _resolved_path(root, recipe.data.manifest), split=recipe.data.split, verify_files=verify_media_files
    )
    cache = VideoCachedDataset(
        _resolved_path(root, recipe.data.cache),
        expected_sample_ids=manifest.sample_ids,
        audit_on_open=audit_cache_on_open,
        verify_on_read=verify_cache_on_read,
    )
    audit_video_cache_against_manifest(cache, manifest)

    overrides = dict(checkpoint_overrides or {})
    checkpoint_value = overrides.pop("denoiser", None)
    if overrides:
        raise ValueError(f"unknown LVDM checkpoint overrides: {sorted(overrides)}")
    if checkpoint_value is not None and recipe.model.checkpoint != "default":
        raise ValueError("model.checkpoint and denoiser override cannot both be set")
    checkpoint_value = checkpoint_value or recipe.model.checkpoint
    if checkpoint_value == "default":
        raise ValueError("LVDM short training requires model.checkpoint or a denoiser override")

    resolved_device = torch.device(device)
    distributed_context: DistributedTrainingContext | None = None
    if recipe.distributed.backend == "fsdp2":
        if resolved_device.type != "cuda":
            raise ValueError("FSDP2 materialization requires device='cuda'")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device
    elif recipe.distributed.backend != "single":
        raise NotImplementedError(f"LVDM materialization does not implement {recipe.distributed.backend!r}")

    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader
    from worldfoundry.base_models.diffusion_model.models.networks.lvdm.short_unet import LVDMShortUNet
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy

    config = {
        "image_size": 32,
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 256,
        "attention_resolutions": (8, 4, 2),
        "num_res_blocks": 3,
        "channel_mult": (1, 2, 3, 4),
        "num_heads": 4,
        "use_scale_shift_norm": True,
        "use_checkpoint": recipe.runtime.activation_checkpoint == "full",
        "legacy": False,
        "kernel_size_t": 1,
        "padding_t": 0,
        "temporal_length": 4,
        "use_relative_position": True,
        "use_temporal_transformer": False,
    }
    ema_checkpoint_state: dict[str, object] = {}

    def convert_checkpoint(state_dict: Mapping[str, object]) -> Mapping[str, object]:
        ema_checkpoint_state.update(_lvdm_ema_checkpoint_state(state_dict))
        return _short_unet_state_dict(state_dict)

    try:
        denoiser = NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=LVDMShortUNet,
                config=config,
                state_dict_converter=convert_checkpoint,
            ),
            CheckpointSpec(source=str(_resolved_path(root, str(checkpoint_value)))),
            RuntimePolicy(
                device=resolved_device,
                dtype=torch_dtype(recipe.runtime.param_dtype),
                attention=AttentionBackend.TORCH,
            ),
        )
        adapter = LVDMUnconditionalTrainAdapter(denoiser, codec=None)
        destination = _resolved_path(root, output_dir or recipe.run.output_dir)
        if distributed_context is not None:
            return build_lvdm_short_fsdp2_session(
                recipe=recipe,
                adapter=adapter,
                dataset=cache,
                distributed_context=distributed_context,
                output_dir=destination,
                fused_adamw=fused_adamw,
                initialization_seed=initialization_seed,
                ema_checkpoint_state=ema_checkpoint_state,
            )
        return build_lvdm_short_single_device_session(
            recipe=recipe,
            adapter=adapter,
            dataset=cache,
            output_dir=destination,
            fused_adamw=fused_adamw,
            initialization_seed=initialization_seed,
            ema_checkpoint_state=ema_checkpoint_state,
        )
    except Exception:
        if distributed_context is not None:
            distributed_context.close()
        raise


__all__ = [
    "build_lvdm_short_fsdp2_session",
    "build_lvdm_short_objective",
    "build_lvdm_short_single_device_session",
    "materialize_lvdm_short_training_session",
    "validate_lvdm_short_recipe",
]
