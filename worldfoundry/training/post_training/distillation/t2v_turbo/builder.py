"""Native materialization and session lifecycle for T2V-Turbo distillation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from worldfoundry.core.utils.torch_utils import set_seed_everywhere
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
from worldfoundry.training.recipes.post_training.algorithms.t2v_turbo import (
    T2VTurboAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from .cache import validate_t2v_turbo_cache_contract
from .lora import T2VTurboLoraApplication, apply_t2v_turbo_lora
from .objective import (
    DifferentiableImageReward,
    DifferentiableVideoReward,
    LVDMEpsilonPredictor,
    T2VTurboConfig,
    T2VTurboObjective,
    T2VTurboTrainAdapter,
)

_DATA_OPTIONS = frozenset({"video_buckets", "bucket_policy", "decode", "precomputed_tensors"})


@dataclass(frozen=True, slots=True)
class T2VTurboRoles:
    student: nn.Module
    teacher: nn.Module
    codec: nn.Module | None = None
    text_encoder: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.student, nn.Module) or not isinstance(self.teacher, nn.Module):
            raise TypeError("T2V-Turbo student and teacher must be nn.Module values")
        if self.student is self.teacher:
            raise ValueError("T2V-Turbo student and teacher must be distinct modules")


def _algorithm(recipe: PostTrainingRecipe) -> T2VTurboAlgorithmSpec:
    if not isinstance(recipe.algorithm, T2VTurboAlgorithmSpec):
        raise TypeError("T2V-Turbo builder requires T2VTurboAlgorithmSpec")
    return recipe.algorithm


def build_t2v_turbo_objective(
    recipe: PostTrainingRecipe,
    adapter: T2VTurboTrainAdapter,
    *,
    image_reward: DifferentiableImageReward | None = None,
    video_reward: DifferentiableVideoReward | None = None,
) -> T2VTurboObjective:
    algorithm = _algorithm(recipe)
    return T2VTurboObjective(
        adapter=adapter,
        config=T2VTurboConfig(
            num_train_timesteps=algorithm.num_train_timesteps,
            num_ddim_timesteps=algorithm.num_ddim_timesteps,
            topk=algorithm.topk,
            guidance_min=algorithm.guidance_min,
            guidance_max=algorithm.guidance_max,
            guidance_embedding_dim=algorithm.guidance_embedding_dim,
            sigma_data=algorithm.sigma_data,
            timestep_scaling=algorithm.timestep_scaling,
            loss_type=algorithm.loss_type,
            pseudo_huber_c=algorithm.pseudo_huber_c,
            distillation_weight=algorithm.distillation_weight,
            image_reward_weight=algorithm.image_reward_weight,
            video_reward_weight=algorithm.video_reward_weight,
            image_reward_frames=algorithm.image_reward_frames,
            image_reward_batch_size=algorithm.image_reward_batch_size,
            video_reward_frames=algorithm.video_reward_frames,
        ),
        image_reward=image_reward,
        video_reward=video_reward,
    )


def validate_t2v_turbo_recipe(recipe: PostTrainingRecipe, *, backend: str) -> None:
    algorithm = _algorithm(recipe)
    if recipe.model.recipe != "t2v-turbo":
        raise ValueError(f"T2V-Turbo cannot train model recipe {recipe.model.recipe!r}")
    if recipe.distributed.backend != backend:
        raise ValueError(f"T2V-Turbo requires distributed.backend={backend!r}")
    if recipe.tuning.mode != "lora" or recipe.tuning.preset != "t2v-turbo-unet":
        raise ValueError("native T2V-Turbo requires its UNet Linear/Conv2d/Conv3d LoRA seam")
    if recipe.tuning.rank != 64 or recipe.tuning.alpha != 1 or recipe.tuning.dropout != 0.1:
        raise ValueError("the released T2V-Turbo profile uses rank 64, scale 1, and branch dropout 0.1")
    if recipe.tuning.modules_to_save:
        raise ValueError("T2V-Turbo does not configure modules_to_save")
    if recipe.model.options:
        raise ValueError("T2V-Turbo model.options must be empty")
    if recipe.optimizer.type != "adamw" or recipe.optimizer.weight_decay != 0.0:
        raise ValueError("T2V-Turbo requires AdamW with zero weight decay")
    if not math.isclose(recipe.optimizer.learning_rate, 1.0e-5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("the released T2V-Turbo profile uses a constant 1e-5 learning rate")
    if recipe.runtime.activation_checkpoint not in {"none", "full"}:
        raise ValueError("T2V-Turbo activation_checkpoint must be 'none' or 'full'")
    if recipe.data.max_latent_tokens_per_microbatch is None:
        raise ValueError("cached T2V-Turbo training requires a latent token budget")
    if algorithm.default_fps != 16:
        raise ValueError("the released T2V-Turbo profile requires 16 FPS conditioning")


def _apply_tuning(
    recipe: PostTrainingRecipe,
    adapter: object,
) -> T2VTurboLoraApplication:
    if not isinstance(adapter, T2VTurboTrainAdapter):
        raise TypeError("T2V-Turbo tuning requires T2VTurboTrainAdapter")
    assert recipe.tuning.rank is not None
    application = apply_t2v_turbo_lora(
        adapter.trainable_module,
        rank=recipe.tuning.rank,
        dropout=recipe.tuning.dropout,
    )
    adapter.replace_student_module(application.model)
    return application


def build_t2v_turbo_adapter(
    recipe: PostTrainingRecipe,
    roles: T2VTurboRoles,
) -> T2VTurboTrainAdapter:
    algorithm = _algorithm(recipe)
    roles.teacher.to(dtype=torch.float32)
    compute_dtype = torch_dtype(recipe.runtime.param_dtype)
    return T2VTurboTrainAdapter(
        student=LVDMEpsilonPredictor(roles.student),
        teacher=LVDMEpsilonPredictor(roles.teacher),
        codec=roles.codec,
        text_encoder=roles.text_encoder,
        default_fps=algorithm.default_fps,
        student_autocast_dtype=None if compute_dtype is torch.float32 else compute_dtype,
    )


def build_t2v_turbo_single_device_session(
    *,
    recipe: PostTrainingRecipe,
    roles: T2VTurboRoles,
    dataset: VideoCachedDataset,
    output_dir: str | Path | None = None,
    image_reward: DifferentiableImageReward | None = None,
    video_reward: DifferentiableVideoReward | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> SingleDeviceTrainingSession:
    validate_t2v_turbo_recipe(recipe, backend="single")
    adapter = build_t2v_turbo_adapter(recipe, roles)
    contract = validate_t2v_turbo_cache_contract(recipe, adapter, dataset)
    objective = build_t2v_turbo_objective(recipe, adapter, image_reward=image_reward, video_reward=video_reward)
    return build_cached_video_flow_single_device_session(
        recipe=recipe,
        adapter=adapter,
        dataset=dataset,
        objective=objective,
        cache_contract=contract,
        output_dir=output_dir,
        tuning_factory=_apply_tuning,
        fused_adamw=fused_adamw,
        initialization_seed=initialization_seed,
        consumed_data_options=_DATA_OPTIONS,
    )


def build_t2v_turbo_fsdp2_session(
    *,
    recipe: PostTrainingRecipe,
    roles: T2VTurboRoles,
    dataset: VideoCachedDataset,
    distributed_context: DistributedTrainingContext,
    output_dir: str | Path | None = None,
    image_reward: DifferentiableImageReward | None = None,
    video_reward: DifferentiableVideoReward | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> FSDP2TrainingSession:
    validate_t2v_turbo_recipe(recipe, backend="fsdp2")
    adapter = build_t2v_turbo_adapter(recipe, roles)
    contract = validate_t2v_turbo_cache_contract(recipe, adapter, dataset)
    objective = build_t2v_turbo_objective(recipe, adapter, image_reward=image_reward, video_reward=video_reward)
    return build_cached_video_flow_fsdp2_session(
        recipe=recipe,
        adapter=adapter,
        dataset=dataset,
        objective=objective,
        cache_contract=contract,
        distributed_context=distributed_context,
        output_dir=output_dir,
        tuning_factory=_apply_tuning,
        fused_adamw=fused_adamw,
        initialization_seed=initialization_seed,
        consumed_data_options=_DATA_OPTIONS,
    )


class T2VTurboTrainingRun:
    """CLI-facing lifecycle over the shared native training session."""

    def __init__(
        self,
        session: SingleDeviceTrainingSession | FSDP2TrainingSession,
        *,
        seed: int,
        resume_checkpoint: str | Path | None,
    ) -> None:
        self.session = session
        self.seed = int(seed)
        self.resume_checkpoint = resume_checkpoint

    @property
    def output_dir(self) -> Path:
        return self.session.output_dir

    @property
    def world_size(self) -> int:
        return self.session.world_size

    @property
    def is_coordinator(self) -> bool:
        return self.session.is_coordinator

    def run(self, *, max_steps: int):
        return self.session.run(
            max_steps=max_steps,
            seed=self.seed,
            resume_checkpoint=self.resume_checkpoint,
        )

    def export_student(self):
        return self.session.export_adapter()

    def close(self) -> None:
        self.session.close()


def _resolved_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _base_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    nested = state_dict.get("state_dict")
    if isinstance(nested, Mapping):
        state_dict = nested
    prefix = "model.diffusion_model."
    converted = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    if not converted:
        raise KeyError("VideoCrafter checkpoint contains no model.diffusion_model parameters")
    return converted


def _checkpoint_spec(value: object, *, root: Path):
    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec

    if value == "default":
        return CheckpointSpec(repo_id="VideoCrafter/VideoCrafter2", files=("model.ckpt",))
    return CheckpointSpec(source=str(_resolved_path(root, str(value))))


def _load_roles(
    recipe: PostTrainingRecipe,
    *,
    root: Path,
    device: torch.device,
    overrides: Mapping[str, object],
    initialization_seed: int,
) -> T2VTurboRoles:
    from worldfoundry.base_models.diffusion_model.loaders import NativeCheckpointResolver
    from worldfoundry.base_models.diffusion_model.models.denoisers.t2v_turbo import (
        T2V_TURBO_UNET_CONFIG,
        t2v_turbo_guidance_projection,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.lvdm.unet3d import UNetModel
    from worldfoundry.core.model_loading import load_torch_checkpoint

    unknown = sorted(set(overrides) - {"student", "teacher"})
    if unknown:
        raise ValueError(f"unknown T2V-Turbo checkpoint overrides: {unknown}")
    student_value = overrides.get("student", recipe.model.checkpoint)
    teacher_value = overrides.get("teacher", _algorithm(recipe).teacher_checkpoint)
    if "student" in overrides and recipe.model.checkpoint != "default":
        raise ValueError("model.checkpoint and student override cannot both be set")
    if "teacher" in overrides and _algorithm(recipe).teacher_checkpoint != "default":
        raise ValueError("algorithm.teacher_checkpoint and teacher override cannot both be set")
    student_config = dict(T2V_TURBO_UNET_CONFIG)
    student_config["use_checkpoint"] = recipe.runtime.activation_checkpoint == "full"
    teacher_config = dict(student_config)
    teacher_config["time_cond_proj_dim"] = None
    resolver = NativeCheckpointResolver()

    def checkpoint_state(value: object) -> Mapping[str, object]:
        materialized = resolver.materialize(_checkpoint_spec(value, root=root))
        source: str | list[str]
        source = (
            str(materialized.paths[0]) if len(materialized.paths) == 1 else [str(path) for path in materialized.paths]
        )
        if isinstance(source, list):
            from worldfoundry.core.model_loading import load_state_dict

            loaded = load_state_dict(source, device="cpu")
        else:
            loaded = load_torch_checkpoint(source, map_location="cpu")
        if not isinstance(loaded, Mapping):
            raise TypeError("VideoCrafter checkpoint must contain a state mapping")
        return _base_state_dict(loaded)

    teacher_state = checkpoint_state(teacher_value)
    teacher = UNetModel(**teacher_config)
    teacher.load_state_dict(teacher_state, strict=True)
    teacher.requires_grad_(False)

    # The released LoRA artifact omits this frozen projection.  Use the same
    # deterministic value here and in fresh-base inference so an exported
    # adapter reproduces the effective model that was trained.
    set_seed_everywhere(int(initialization_seed))
    student = UNetModel(**student_config)
    student_state = teacher_state if student_value == teacher_value else checkpoint_state(student_value)
    incompatible = student.load_state_dict(student_state, strict=False)
    if tuple(incompatible.missing_keys) not in ((), ("time_cond_proj.weight",)) or incompatible.unexpected_keys:
        raise ValueError(
            "T2V-Turbo student checkpoint differs from the VideoCrafter UNet outside time_cond_proj.weight"
        )
    if "time_cond_proj.weight" in incompatible.missing_keys:
        with torch.no_grad():
            student.time_cond_proj.weight.copy_(t2v_turbo_guidance_projection(student.time_cond_proj.weight))
    student.requires_grad_(False)
    student.train()

    teacher.to(device=device, dtype=torch.float32)
    student.to(device=device, dtype=torch.float32)
    return T2VTurboRoles(student=student, teacher=teacher)


def materialize_t2v_turbo_training_run(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    checkpoint_overrides: Mapping[str, object] | None = None,
    verify_media_files: bool = True,
    audit_cache_on_open: bool = True,
    verify_cache_on_read: bool = True,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int = 42,
) -> T2VTurboTrainingRun:
    """Materialize the official base roles for consistency-only training."""

    validate_t2v_turbo_recipe(recipe, backend=recipe.distributed.backend)
    algorithm = _algorithm(recipe)
    if algorithm.image_reward_weight or algorithm.video_reward_weight:
        raise NotImplementedError(
            "CLI materialization of the official HPSv2/ViCLIP reward models is not implemented; "
            "use zero reward weights or inject differentiable reward callables programmatically"
        )
    if recipe.data.cache is None:
        raise ValueError("cached T2V-Turbo training requires data.cache")
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
    resolved_device = torch.device(device)
    distributed_context: DistributedTrainingContext | None = None
    if recipe.distributed.backend == "fsdp2":
        if resolved_device.type != "cuda":
            raise ValueError("FSDP2 materialization requires device='cuda'")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device
    elif recipe.distributed.backend != "single":
        raise NotImplementedError(f"T2V-Turbo materialization does not implement {recipe.distributed.backend!r}")
    try:
        roles = _load_roles(
            recipe,
            root=root,
            device=resolved_device,
            overrides=dict(checkpoint_overrides or {}),
            initialization_seed=initialization_seed,
        )
        destination = _resolved_path(root, output_dir or recipe.run.output_dir)
        if distributed_context is not None:
            session = build_t2v_turbo_fsdp2_session(
                recipe=recipe,
                roles=roles,
                dataset=cache,
                distributed_context=distributed_context,
                output_dir=destination,
                fused_adamw=fused_adamw,
                initialization_seed=initialization_seed,
            )
        else:
            session = build_t2v_turbo_single_device_session(
                recipe=recipe,
                roles=roles,
                dataset=cache,
                output_dir=destination,
                fused_adamw=fused_adamw,
                initialization_seed=initialization_seed,
            )
        return T2VTurboTrainingRun(
            session,
            seed=initialization_seed,
            resume_checkpoint=resume_checkpoint,
        )
    except Exception:
        if distributed_context is not None:
            distributed_context.close()
        raise


__all__ = [
    "T2VTurboRoles",
    "T2VTurboTrainingRun",
    "build_t2v_turbo_adapter",
    "build_t2v_turbo_fsdp2_session",
    "build_t2v_turbo_objective",
    "build_t2v_turbo_single_device_session",
    "materialize_t2v_turbo_training_run",
    "validate_t2v_turbo_recipe",
]
