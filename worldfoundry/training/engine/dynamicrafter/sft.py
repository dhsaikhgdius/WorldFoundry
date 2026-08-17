"""Executable cached training for official DynamiCrafter I2V profiles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.engine.sessions.fsdp2 import FSDP2TrainingSession
from worldfoundry.training.engine.sessions.single_device import SingleDeviceTrainingSession
from worldfoundry.training.engine.video_flow import (
    audit_video_cache_against_manifest,
    build_cached_video_flow_fsdp2_session,
    build_cached_video_flow_single_device_session,
)
from worldfoundry.training.models.dynamicrafter import (
    DynamiCrafterTrainAdapter,
    dynamicrafter_objective,
)
from worldfoundry.training.objectives.classic_diffusion import ClassicDiffusionObjective
from worldfoundry.training.recipes.spec import TrainingRecipe

from .cache import DYNAMICRAFTER_MODEL_RECIPES, validate_dynamicrafter_cache_contract

_MODEL_OPTIONS = frozenset({"training_mode", "default_fps"})
_OBJECTIVE_OPTIONS = frozenset(
    {
        "num_train_timesteps",
        "beta_start",
        "beta_end",
        "loss_type",
        "zero_terminal_snr",
        "dynamic_rescale_final",
        "dynamic_rescale_transition_steps",
    }
)
_DATA_OPTIONS = frozenset({"video_buckets", "bucket_policy", "decode", "precomputed_tensors"})


def _official_dynamic_rescale(model_recipe: str) -> float:
    return 0.3 if model_recipe == "dynamicrafter-1024-i2v" else 0.7


def _training_mode(recipe: TrainingRecipe) -> str:
    options = dict(recipe.model.options)
    unknown = sorted(set(options) - _MODEL_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported DynamiCrafter model options: {unknown}")
    mode = str(options.get("training_mode", "image-to-video")).lower().replace("_", "-")
    if mode not in {"image-to-video", "interpolation"}:
        raise ValueError("DynamiCrafter training_mode must be image-to-video or interpolation")
    if mode == "interpolation" and recipe.model.recipe != "dynamicrafter-512-i2v":
        raise ValueError("the released interpolation trainer is a DynamiCrafter 512 profile")
    return mode


def build_dynamicrafter_objective(
    recipe: TrainingRecipe,
    adapter: DynamiCrafterTrainAdapter,
) -> ClassicDiffusionObjective:
    options = dict(recipe.objective.options)
    unknown = sorted(set(options) - _OBJECTIVE_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported DynamiCrafter objective options: {unknown}")
    expected: dict[str, object] = {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "loss_type": "l2",
        "zero_terminal_snr": True,
        "dynamic_rescale_final": _official_dynamic_rescale(recipe.model.recipe),
        "dynamic_rescale_transition_steps": 400,
    }
    for name, default in expected.items():
        if options.get(name, default) != default:
            raise ValueError(f"DynamiCrafter {name} must match the released trainer value {default!r}")
    return dynamicrafter_objective(
        adapter,
        dynamic_rescale_final=float(expected["dynamic_rescale_final"]),
    )


def validate_dynamicrafter_recipe(
    recipe: TrainingRecipe,
    adapter: DynamiCrafterTrainAdapter,
    *,
    backend: str,
) -> None:
    if recipe.model.recipe not in DYNAMICRAFTER_MODEL_RECIPES:
        raise ValueError(f"DynamiCrafter training cannot train {recipe.model.recipe!r}")
    if recipe.distributed.backend != backend:
        raise ValueError(f"DynamiCrafter training requires distributed.backend={backend!r}")
    if recipe.tuning.mode != "full":
        raise ValueError("the released DynamiCrafter trainer performs full denoiser/projector tuning")
    if recipe.objective.type != "classic-diffusion" or recipe.objective.prediction_type != "v_prediction":
        raise ValueError("DynamiCrafter training requires v-prediction classic diffusion")
    if recipe.objective.timestep_sampler != "uniform":
        raise ValueError("DynamiCrafter training samples timesteps uniformly")
    if recipe.objective.conditioning_dropout != 0.05:
        raise ValueError("DynamiCrafter conditioning_dropout must be 0.05")
    if adapter.interpolation != (_training_mode(recipe) == "interpolation"):
        raise ValueError("DynamiCrafter adapter training mode differs from the recipe")
    default_fps = int(recipe.model.options.get("default_fps", 10))
    if default_fps != 10:
        raise ValueError("the released DynamiCrafter trainer uses default_fps=10")
    if adapter.default_fps != default_fps:
        raise ValueError("DynamiCrafter adapter default FPS differs from the recipe")
    if recipe.runtime.activation_checkpoint not in {"none", "full"}:
        raise ValueError("DynamiCrafter activation_checkpoint must be 'none' or 'full'")
    if recipe.data.max_latent_tokens_per_microbatch is None:
        raise ValueError("cached DynamiCrafter training requires a latent token budget")
    build_dynamicrafter_objective(recipe, adapter)


def build_dynamicrafter_single_device_session(
    *,
    recipe: TrainingRecipe,
    adapter: DynamiCrafterTrainAdapter,
    dataset: VideoCachedDataset,
    output_dir: str | Path | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> SingleDeviceTrainingSession:
    validate_dynamicrafter_recipe(recipe, adapter, backend="single")
    contract = validate_dynamicrafter_cache_contract(recipe, adapter, dataset)
    return build_cached_video_flow_single_device_session(
        recipe=recipe,
        adapter=adapter,
        dataset=dataset,
        objective=build_dynamicrafter_objective(recipe, adapter),
        cache_contract=contract,
        output_dir=output_dir,
        fused_adamw=fused_adamw,
        initialization_seed=initialization_seed,
        consumed_data_options=_DATA_OPTIONS,
    )


def build_dynamicrafter_fsdp2_session(
    *,
    recipe: TrainingRecipe,
    adapter: DynamiCrafterTrainAdapter,
    dataset: VideoCachedDataset,
    distributed_context: DistributedTrainingContext,
    output_dir: str | Path | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> FSDP2TrainingSession:
    validate_dynamicrafter_recipe(recipe, adapter, backend="fsdp2")
    contract = validate_dynamicrafter_cache_contract(recipe, adapter, dataset)
    return build_cached_video_flow_fsdp2_session(
        recipe=recipe,
        adapter=adapter,
        dataset=dataset,
        objective=build_dynamicrafter_objective(recipe, adapter),
        cache_contract=contract,
        distributed_context=distributed_context,
        output_dir=output_dir,
        fused_adamw=fused_adamw,
        initialization_seed=initialization_seed,
        consumed_data_options=_DATA_OPTIONS,
    )


def _resolved_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _dynamicrafter_state_dict(state_dict: Mapping[str, object]) -> dict[str, object]:
    nested = state_dict.get("state_dict")
    if isinstance(nested, Mapping):
        state_dict = nested
    else:
        module_state = state_dict.get("module")
        if isinstance(module_state, Mapping):
            state_dict = {key[16:]: value for key, value in module_state.items()}
    return {key.replace("framestride_embed", "fps_embedding"): value for key, value in state_dict.items()}


def _dynamicrafter_component_state_dicts(
    state_dict: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    state = _dynamicrafter_state_dict(state_dict)
    denoiser_prefix = "model."
    denoiser = {key[len(denoiser_prefix) :]: value for key, value in state.items() if key.startswith(denoiser_prefix)}
    projector_prefix = "image_proj_model."
    projector = {
        key[len(projector_prefix) :]: value for key, value in state.items() if key.startswith(projector_prefix)
    }
    return denoiser, projector


def _load_native_dynamicrafter(
    recipe: TrainingRecipe,
    *,
    root: Path,
    checkpoint: object | None,
    device: torch.device,
) -> DynamiCrafterTrainAdapter:
    import yaml

    from worldfoundry.base_models.diffusion_model.models.networks.lvdm.latent_diffusion import (
        DiffusionWrapper,
    )
    from worldfoundry.core.io.paths import resolve_data_path
    from worldfoundry.core.model_loading.factory import instantiate_from_config
    from worldfoundry.core.model_loading.file import load_state_dict
    from worldfoundry.runtime.assets import expand_worldfoundry_path

    defaults_path = resolve_data_path("models", "runtime", "configs", "dynamicrafter", "runtime_defaults.yaml")
    defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))["defaults"]
    profile = defaults[recipe.model.recipe.replace("-", "_")]
    config_path = resolve_data_path("models", *Path(profile["config"]).parts)
    checkpoint_value = checkpoint or recipe.model.checkpoint
    if checkpoint_value == "default":
        checkpoint_value = profile["ckpt_path"]
        checkpoint_path = expand_worldfoundry_path(str(checkpoint_value))
    else:
        checkpoint_path = _resolved_path(root, str(checkpoint_value))

    model_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    params = model_config["params"]
    unet_config = params["unet_config"]
    unet_config["params"]["use_checkpoint"] = recipe.runtime.activation_checkpoint == "full"
    unet_config["params"]["default_fs"] = 10
    denoiser = DiffusionWrapper(unet_config, params["conditioning_key"])
    image_projector = instantiate_from_config(params["image_proj_stage_config"])

    denoiser_state, projector_state = _dynamicrafter_component_state_dicts(
        load_state_dict(checkpoint_path, device="cpu")
    )
    denoiser.load_state_dict(denoiser_state, strict=True)
    image_projector.load_state_dict(projector_state, strict=True)
    # Lightning precision=16 keeps FP32 optimizer/master parameters and uses
    # FP16 only for autocast compute.  The shared engine owns that autocast and
    # its checkpointed GradScaler; materialization must not downcast weights.
    denoiser.to(device=device, dtype=torch.float32)
    image_projector.to(device=device, dtype=torch.float32)
    adapter = DynamiCrafterTrainAdapter(
        denoiser=denoiser,
        image_projector=image_projector,
        interpolation=_training_mode(recipe) == "interpolation",
        conditioning_dropout_probability=recipe.objective.conditioning_dropout,
        default_fps=int(recipe.model.options.get("default_fps", 10)),
    )
    return adapter


def materialize_dynamicrafter_training_session(
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
    """Build a native session from the packaged architecture and official checkpoint."""

    if recipe.model.recipe not in DYNAMICRAFTER_MODEL_RECIPES:
        raise ValueError(f"DynamiCrafter materialization cannot train {recipe.model.recipe!r}")
    if recipe.data.cache is None:
        raise ValueError("cached DynamiCrafter training requires data.cache")
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
    checkpoint = overrides.pop("model", None)
    if overrides:
        raise ValueError(f"unknown DynamiCrafter checkpoint overrides: {sorted(overrides)}")
    if checkpoint is not None and recipe.model.checkpoint != "default":
        raise ValueError("model.checkpoint and model override cannot both be set")

    resolved_device = torch.device(device)
    distributed_context: DistributedTrainingContext | None = None
    if recipe.distributed.backend == "fsdp2":
        if resolved_device.type != "cuda":
            raise ValueError("FSDP2 materialization requires device='cuda'")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device
    elif recipe.distributed.backend != "single":
        raise NotImplementedError(f"DynamiCrafter materialization does not implement {recipe.distributed.backend!r}")
    try:
        adapter = _load_native_dynamicrafter(recipe, root=root, checkpoint=checkpoint, device=resolved_device)
        destination = _resolved_path(root, output_dir or recipe.run.output_dir)
        if distributed_context is not None:
            return build_dynamicrafter_fsdp2_session(
                recipe=recipe,
                adapter=adapter,
                dataset=cache,
                distributed_context=distributed_context,
                output_dir=destination,
                fused_adamw=fused_adamw,
                initialization_seed=initialization_seed,
            )
        return build_dynamicrafter_single_device_session(
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
    "build_dynamicrafter_fsdp2_session",
    "build_dynamicrafter_objective",
    "build_dynamicrafter_single_device_session",
    "materialize_dynamicrafter_training_session",
    "validate_dynamicrafter_recipe",
]
