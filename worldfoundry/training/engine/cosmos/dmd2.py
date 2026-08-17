"""Native Cosmos Predict2.5 DMD2 materialization and run lifecycle."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.core.io.integrity import append_jsonl_durable
from worldfoundry.core.time import utc_now_iso
from worldfoundry.core.utils.torch_utils import set_seed_everywhere
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.distributed.fsdp import (
    FSDP2Application,
    apply_fsdp2,
    apply_fsdp2_frozen_reference,
)
from worldfoundry.training.distributed.parallel import DistributedTrainingContext, ParallelPlan
from worldfoundry.training.engine.video_flow import (
    audit_video_cache_against_manifest,
    build_cached_video_loader,
    torch_dtype,
)
from worldfoundry.training.models.cosmos import (
    CosmosPredict25TrainAdapter,
    build_cached_cosmos_predict25_train_adapter,
)
from worldfoundry.training.post_training.distillation.dmd2.builder import (
    build_native_dmd2_training_stack,
)
from worldfoundry.training.post_training.distillation.dmd2.session import (
    NativeDMD2TrainingSession,
)
from worldfoundry.training.post_training.shared.distributed import PostTrainingParallelContext
from worldfoundry.training.post_training.shared.role_checkpoints import (
    ResolvedRoleCheckpoint,
    resolve_role_checkpoint,
)
from worldfoundry.training.recipes.post_training.algorithms.dmd2 import DMD2AlgorithmSpec
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.peft import PeftLoraApplication

from ..artifacts import create_run_directory
from ..student_distillation import StudentDistillationTrainingRun
from .dmd2_data import (
    COSMOS_PREDICT25_DMD2_CONDITIONAL_FRAME_PROBABILITIES,
    CosmosPredict25DMD2DataLoader,
)
from .dmd2_roles import (
    COSMOS_DMD2_GENERATOR_UPDATE_INTERVAL,
    COSMOS_PREDICT25_DMD2_FLOW_SIGMAS,
    COSMOS_PREDICT25_DMD2_TRIGFLOW_TIMES,
    CosmosDMD2DiscriminatorHead,
    CosmosDMD2GuidanceAdapter,
    CosmosFlowDMD2PredictionAdapter,
)
from .precision import promote_trainable_parameters_to_fp32
from .sft import apply_cosmos_tuning

COSMOS_PREDICT25_DMD2_RUN_SCHEMA = "worldfoundry-cosmos-predict25-dmd2-run"
COSMOS_PREDICT25_DMD2_FEATURE_IDS = (11, 19, 27)
COSMOS_PREDICT25_DMD2_REPO_ID = "nvidia/Cosmos-Predict2.5-2B"
COSMOS_PREDICT25_DMD2_PRETRAINED_REVISION = "15a82a2ec231bc318692aa0456a36537c806e7d4"
COSMOS_PREDICT25_DMD2_PRETRAINED_WEIGHT_FILE = "base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
_DMD2_DATA_OPTIONS = frozenset({"video_buckets", "bucket_policy", "decode", "conditional_frame_probabilities"})


def cosmos_predict25_dmd2_lr_multiplier(step: int) -> float:
    """Released LambdaLinear warmup/decay multiplier for each active role."""

    current = min(int(step), 400_000)
    if current < 100:
        return (0.99 - 1.0e-6) / 100.0 * current + 1.0e-6
    return 0.4 + (0.99 - 0.4) * (400_000 - current) / (400_000 - 100)


def _cosmos_predict25_dmd2_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=cosmos_predict25_dmd2_lr_multiplier,
    )


def cosmos_predict25_dmd2_pretrained_checkpoint() -> CheckpointSpec:
    """Return the public pretrained 2B checkpoint used by released DMD2."""

    return CheckpointSpec(
        repo_id=COSMOS_PREDICT25_DMD2_REPO_ID,
        revision=COSMOS_PREDICT25_DMD2_PRETRAINED_REVISION,
        files=(COSMOS_PREDICT25_DMD2_PRETRAINED_WEIGHT_FILE,),
        allow_patterns=(COSMOS_PREDICT25_DMD2_PRETRAINED_WEIGHT_FILE,),
    )


def _same_floats(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=1.0e-12, abs_tol=1.0e-12) for a, b in zip(left, right, strict=True)
    )


def _conditional_frame_probabilities(recipe: PostTrainingRecipe) -> tuple[float, ...]:
    raw = recipe.data.options.get(
        "conditional_frame_probabilities",
        COSMOS_PREDICT25_DMD2_CONDITIONAL_FRAME_PROBABILITIES,
    )
    if not isinstance(raw, (list, tuple)):
        raise TypeError("conditional_frame_probabilities must be a sequence")
    values = tuple(float(value) for value in raw)
    if not _same_floats(values, COSMOS_PREDICT25_DMD2_CONDITIONAL_FRAME_PROBABILITIES):
        raise ValueError("Predict2.5 DMD2 uses the released T2V probabilities [0.6, 0.2, 0.2]")
    return values


def _validate_official_recipe(recipe: PostTrainingRecipe) -> DMD2AlgorithmSpec:
    if not isinstance(recipe, PostTrainingRecipe):
        raise TypeError("recipe must be PostTrainingRecipe")
    if not isinstance(recipe.algorithm, DMD2AlgorithmSpec):
        raise TypeError("Cosmos Predict2.5 materialization requires algorithm.type='dmd2'")
    if recipe.model.recipe != "cosmos-predict2.5-2b":
        raise ValueError("the native author-parity DMD2 path supports Cosmos Predict2.5 2B")
    if recipe.data.cache is None or recipe.data.max_latent_tokens_per_microbatch is None:
        raise ValueError("Cosmos Predict2.5 DMD2 requires a token-budgeted video cache")
    if recipe.data.tail_policy not in {"drop", "pad"}:
        raise ValueError("Cosmos Predict2.5 DMD2 requires data.tail_policy='drop' or 'pad'")
    if recipe.distributed.backend not in {"single", "fsdp2"}:
        raise ValueError("Cosmos Predict2.5 DMD2 supports single or FSDP2 execution")
    if recipe.distributed.cp != 1 or recipe.distributed.tp != 1:
        raise ValueError("the local Cosmos DMD2 graph currently scales over data parallelism")
    if recipe.runtime.activation_checkpoint != "none":
        raise ValueError("Cosmos Predict2.5 activation checkpointing is not wired into the local graph")
    if recipe.tuning.mode != "full":
        raise ValueError("the released Cosmos Predict2.5 DMD2 trainer updates full model weights")

    algorithm = recipe.algorithm
    released_schedule = _same_floats(
        algorithm.student_timesteps,
        COSMOS_PREDICT25_DMD2_TRIGFLOW_TIMES,
    ) and _same_floats(algorithm.student_sigmas, COSMOS_PREDICT25_DMD2_FLOW_SIGMAS)
    released_dynamics = (
        algorithm.generator_update_interval == COSMOS_DMD2_GENERATOR_UPDATE_INTERVAL
        and algorithm.student_scheduler_cadence == "generator-update"
        and algorithm.update_mode == "alternating"
        and algorithm.rollout_noise_mode == "shared-initial"
        and algorithm.student_step_sampling == "rank-shared"
        and algorithm.score_sampling == "continuous"
        and algorithm.score_timestep_mode == "per-sample"
        and algorithm.normalization_reference == "generated-clean"
        and algorithm.shared_adversarial_score_input
        and algorithm.distribution_matching_dtype == "float64"
    )
    released_loss = (
        algorithm.normalization_axes == (1, 2, 3, 4)
        and algorithm.num_train_timesteps == 1000
        and math.isclose(algorithm.score_min_sigma, 0.0)
        and math.isclose(algorithm.score_max_sigma, 1.0)
        and math.isclose(algorithm.score_flow_shift, 5.0)
        and math.isclose(algorithm.teacher_guidance_scale, 4.0)
        and math.isclose(algorithm.normalization_epsilon, 1.0e-5)
        and math.isclose(algorithm.distribution_matching_weight, 2.0)
        and math.isclose(algorithm.generator_adversarial_weight, 1.0)
        and math.isclose(algorithm.guidance_denoising_weight, 1.0)
        and math.isclose(algorithm.guidance_adversarial_weight, 1.0)
        and algorithm.diffusion_gan_max_sigma is None
    )
    if not released_schedule or not released_dynamics or not released_loss:
        raise ValueError("Cosmos Predict2.5 DMD2 recipe differs from the released T2V discriminator loop")
    if recipe.optimizer.type != "adamw" or recipe.guidance_optimizer is None:
        raise ValueError("Cosmos Predict2.5 DMD2 requires student and guidance AdamW optimizers")
    if not (
        math.isclose(recipe.optimizer.learning_rate, 1.0e-6)
        and math.isclose(recipe.guidance_optimizer.learning_rate, 2.0e-7)
        and math.isclose(recipe.optimizer.weight_decay, 0.01)
        and math.isclose(recipe.guidance_optimizer.weight_decay, 0.01)
        and recipe.optimizer.betas == (0.9, 0.999)
        and recipe.guidance_optimizer.betas == (0.9, 0.999)
    ):
        raise ValueError("Cosmos Predict2.5 DMD2 optimizer values differ from the released 2B trainer")
    _conditional_frame_probabilities(recipe)
    return algorithm


def validate_cosmos_predict25_dmd2_cache(
    recipe: PostTrainingRecipe,
    adapter: CosmosPredict25TrainAdapter,
    dataset: VideoCachedDataset,
) -> dict[str, object]:
    """Validate the latent and positive/negative text tensors consumed by DMD2."""

    for entry in dataset.index.entries:
        if entry.provenance.model_recipe != recipe.model.recipe:
            raise ValueError(f"cache entry {entry.sample_id!r} belongs to another model recipe")
        latents = entry.tensors["clean_latents"]
        if latents.shape[0] != adapter.expected_latent_channels:
            raise ValueError(f"cache entry {entry.sample_id!r} has incompatible latent channels")
        if any(size % patch for size, patch in zip(latents.shape[-3:], adapter.patch_size, strict=True)):
            raise ValueError(f"cache entry {entry.sample_id!r} is incompatible with the DiT patch size")
        positive = entry.tensors.get("condition.context")
        negative = entry.tensors.get("condition.negative_context")
        if positive is None or negative is None:
            raise ValueError(f"cache entry {entry.sample_id!r} requires positive and negative text context")
        if positive.shape != negative.shape or positive.dtype != negative.dtype or positive.layout != negative.layout:
            raise ValueError(f"cache entry {entry.sample_id!r} has mismatched CFG context tensors")
    return {
        "model_recipe": recipe.model.recipe,
        "latent_channels": adapter.expected_latent_channels,
        "conditioning": ["context", "negative_context"],
        "conditional_frame_probabilities": list(_conditional_frame_probabilities(recipe)),
    }


def _load_role_adapter(
    *,
    assembler: object,
    native_recipe: object,
    checkpoint: ResolvedRoleCheckpoint,
    device: torch.device,
    dtype: torch.dtype,
    num_train_timesteps: int,
) -> CosmosPredict25TrainAdapter:
    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentKey,
        ComponentKind,
    )
    from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.spec import NativeDiffusionRecipe

    if not isinstance(assembler, NativeDiffusionAssembler) or not isinstance(native_recipe, NativeDiffusionRecipe):
        raise TypeError("Cosmos role loading requires native assembler and recipe values")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    components = assembler.build_components(
        native_recipe,
        purpose=BuildPurpose.TRAINING,
        policy=RuntimePolicy(device=device, dtype=dtype, attention=AttentionBackend.TORCH),
        checkpoint_overrides={"transformer": checkpoint.checkpoint},
        component_keys=(denoiser_key,),
    )
    return build_cached_cosmos_predict25_train_adapter(
        components,
        expected_latent_channels=int(native_recipe.options["latent_channels"]),
        temporal_compression=int(native_recipe.options["temporal_compression"]),
        spatial_compression=int(native_recipe.options["spatial_compression"]),
        num_train_timesteps=num_train_timesteps,
    )


def _freeze_text_projection(adapter: CosmosPredict25TrainAdapter) -> None:
    projection = getattr(adapter.trainable_module, "text_embed", None)
    if not isinstance(projection, nn.Module):
        raise TypeError("Cosmos Predict2.5 DiT must expose its text projection")
    projection.requires_grad_(False)


def _prepare_trainable_backbone(adapter: CosmosPredict25TrainAdapter) -> None:
    _freeze_text_projection(adapter)
    promote_trainable_parameters_to_fp32(adapter.trainable_module)


class CosmosDMD2TrainableRoles(nn.Module):
    """DCP model tree containing the two mutable DMD2 roles."""

    def __init__(self, student: nn.Module, guidance: nn.Module) -> None:
        super().__init__()
        self.student = student
        self.guidance = guidance


@dataclass(frozen=True, slots=True)
class CosmosPredict25DMD2RoleBundle:
    student: CosmosPredict25TrainAdapter
    real_score: CosmosPredict25TrainAdapter
    fake_score: CosmosPredict25TrainAdapter
    guidance: CosmosDMD2GuidanceAdapter
    student_checkpoint: ResolvedRoleCheckpoint
    real_score_checkpoint: ResolvedRoleCheckpoint
    guidance_checkpoint: ResolvedRoleCheckpoint
    student_peft: PeftLoraApplication | None
    guidance_peft: PeftLoraApplication | None
    student_fsdp: FSDP2Application | None
    real_score_fsdp: FSDP2Application | None
    guidance_fsdp: FSDP2Application | None

    def runtime_identity(self) -> dict[str, object]:
        def fsdp(value: FSDP2Application | None) -> object:
            return None if value is None else value.to_dict()

        return {
            "checkpoints": {
                "student": self.student_checkpoint.to_dict(),
                "real_score": self.real_score_checkpoint.to_dict(),
                "guidance": self.guidance_checkpoint.to_dict(),
            },
            "fsdp2": {
                "student": fsdp(self.student_fsdp),
                "real_score": fsdp(self.real_score_fsdp),
                "guidance": fsdp(self.guidance_fsdp),
            },
        }


class CosmosPredict25DMD2TrainingRun(StudentDistillationTrainingRun):
    """Cosmos Predict2.5 specialization of the shared distillation run."""

    run_schema = COSMOS_PREDICT25_DMD2_RUN_SCHEMA
    algorithm_label = "Cosmos Predict2.5 DMD2"
    export_role_label = "Cosmos Predict2.5 DMD2 student"


def materialize_cosmos_predict25_dmd2_training_run(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    role_checkpoint_overrides: Mapping[str, object] | None = None,
    verify_media_files: bool = True,
    audit_cache_on_open: bool = True,
    verify_cache_on_read: bool = True,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> CosmosPredict25DMD2TrainingRun:
    """Materialize the released 2B T2V DMD2 roles on the active world size."""

    algorithm = _validate_official_recipe(recipe)
    root = Path(base_dir).expanduser().resolve()
    cache_path = Path(recipe.data.cache or "")
    manifest_path = Path(recipe.data.manifest)
    destination = Path(output_dir or recipe.run.output_dir)
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    distributed_context: DistributedTrainingContext | None = None
    if recipe.distributed.backend == "fsdp2":
        if resolved_device.type != "cuda":
            raise ValueError("Cosmos Predict2.5 DMD2 FSDP2 requires CUDA")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device

    try:
        world_size = 1 if distributed_context is None else distributed_context.world_size
        rank = 0 if distributed_context is None else distributed_context.rank
        plan = ParallelPlan.resolve(recipe.distributed, world_size=world_size)
        create_run_directory(destination, distributed_context)
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

        from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
        from worldfoundry.base_models.diffusion_model.recipes.registry import (
            default_native_diffusion_registry,
        )

        native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
        released_transformer = cosmos_predict25_dmd2_pretrained_checkpoint()
        overrides = dict(role_checkpoint_overrides or {})
        unknown = sorted(set(overrides) - {"student", "real-score", "guidance"})
        if unknown:
            raise ValueError(f"unknown Cosmos DMD2 role checkpoint overrides: {unknown}")
        if any(not isinstance(value, CheckpointSpec) for value in overrides.values()):
            raise TypeError("Cosmos DMD2 role checkpoint overrides must be CheckpointSpec values")
        student_checkpoint = resolve_role_checkpoint(
            role="student",
            reference=recipe.model.checkpoint,
            native_default=released_transformer,
            local_override=overrides.get("student"),
        )
        real_score_checkpoint = resolve_role_checkpoint(
            role="real-score",
            reference=algorithm.real_score_checkpoint,
            native_default=released_transformer,
            local_override=overrides.get("real-score"),
        )
        guidance_checkpoint = resolve_role_checkpoint(
            role="guidance",
            reference=algorithm.guidance_checkpoint,
            native_default=released_transformer,
            local_override=overrides.get("guidance"),
        )

        seed = int(recipe.data.shuffle_seed if initialization_seed is None else initialization_seed)
        set_seed_everywhere(seed)
        assembler = NativeDiffusionAssembler()
        dtype = torch_dtype(recipe.runtime.param_dtype)
        role_options = {
            "assembler": assembler,
            "native_recipe": native_recipe,
            "device": resolved_device,
            "dtype": dtype,
            "num_train_timesteps": algorithm.num_train_timesteps,
        }
        student = _load_role_adapter(checkpoint=student_checkpoint, **role_options)
        student_peft = apply_cosmos_tuning(recipe, student)
        _prepare_trainable_backbone(student)

        real_score = _load_role_adapter(checkpoint=real_score_checkpoint, **role_options)
        real_score.trainable_module.requires_grad_(False)
        real_score.trainable_module.eval()

        fake_score = _load_role_adapter(checkpoint=guidance_checkpoint, **role_options)
        guidance_peft = apply_cosmos_tuning(recipe, fake_score)
        _prepare_trainable_backbone(fake_score)
        blocks = getattr(fake_score.trainable_module, "transformer_blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) <= max(COSMOS_PREDICT25_DMD2_FEATURE_IDS):
            raise ValueError("Cosmos Predict2.5 2B DiT does not expose the released discriminator features")
        output_projection = getattr(fake_score.trainable_module, "proj_out", None)
        if not isinstance(output_projection, nn.Linear):
            raise TypeError("Cosmos Predict2.5 DiT must expose proj_out")
        discriminator = CosmosDMD2DiscriminatorHead(
            model_channels=output_projection.in_features,
            num_branches=len(COSMOS_PREDICT25_DMD2_FEATURE_IDS),
        ).to(device=resolved_device, dtype=torch.float32)

        autocast_dtype = None if dtype is torch.float32 else dtype
        student_prediction = CosmosFlowDMD2PredictionAdapter(
            student,
            checkpoint_identity=recipe.model.checkpoint,
            autocast_dtype=autocast_dtype,
        )
        real_prediction = CosmosFlowDMD2PredictionAdapter(
            real_score,
            checkpoint_identity=algorithm.real_score_checkpoint,
            autocast_dtype=autocast_dtype,
        )
        fake_prediction = CosmosFlowDMD2PredictionAdapter(
            fake_score,
            checkpoint_identity=algorithm.guidance_checkpoint,
            autocast_dtype=autocast_dtype,
        )
        guidance = CosmosDMD2GuidanceAdapter(
            fake_prediction,
            checkpoint_identity=algorithm.guidance_checkpoint,
            discriminator=discriminator,
            intermediate_feature_ids=COSMOS_PREDICT25_DMD2_FEATURE_IDS,
            trigflow_denoising_weight=True,
        )

        student_fsdp: FSDP2Application | None = None
        real_score_fsdp: FSDP2Application | None = None
        guidance_fsdp: FSDP2Application | None = None
        if distributed_context is not None:
            mesh = plan.build_device_mesh(resolved_device.type)
            reduce_dtype = torch_dtype(recipe.runtime.reduce_dtype)
            student_fsdp = apply_fsdp2(
                student,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=reduce_dtype,
                master_parameter_dtype=torch.float32,
            )
            real_score_fsdp = apply_fsdp2_frozen_reference(
                real_score,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=reduce_dtype,
            )
            guidance_fsdp = apply_fsdp2(
                guidance,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=reduce_dtype,
                master_parameter_dtype=torch.float32,
            )

        cache_contract = validate_cosmos_predict25_dmd2_cache(recipe, student, cache)
        source_loader, token_sampler = build_cached_video_loader(
            recipe=recipe,  # PostTrainingRecipe exposes the same data contract.
            dataset=cache,
            rank=rank,
            world_size=world_size,
            default_pin_memory=resolved_device.type == "cuda",
            consumed_data_options=_DMD2_DATA_OPTIONS,
        )
        dmd2_loader = CosmosPredict25DMD2DataLoader(
            source_loader,
            student,
            conditional_frame_probabilities=_conditional_frame_probabilities(recipe),
            seed=recipe.data.shuffle_seed + rank,
        )
        stack = build_native_dmd2_training_stack(
            recipe,
            student=student_prediction,
            real_score=real_prediction,
            guidance=guidance,
            student_scheduler_factory=_cosmos_predict25_dmd2_scheduler,
            guidance_scheduler_factory=_cosmos_predict25_dmd2_scheduler,
            parallel_context=PostTrainingParallelContext.current(),
            fused_adamw=fused_adamw,
        )
        roles = CosmosPredict25DMD2RoleBundle(
            student=student,
            real_score=real_score,
            fake_score=fake_score,
            guidance=guidance,
            student_checkpoint=student_checkpoint,
            real_score_checkpoint=real_score_checkpoint,
            guidance_checkpoint=guidance_checkpoint,
            student_peft=student_peft,
            guidance_peft=guidance_peft,
            student_fsdp=student_fsdp,
            real_score_fsdp=real_score_fsdp,
            guidance_fsdp=guidance_fsdp,
        )
        data_identity = {
            "cache_index": cache.index.to_dict(),
            "cache_contract": cache_contract,
            "latent_token_budget": token_sampler.max_latent_tokens,
            "token_sampler": {
                "seed": token_sampler.seed,
                "shuffle": token_sampler.shuffle,
                "tail_policy": token_sampler.tail_policy,
                "rank": token_sampler.rank,
                "world_size": token_sampler.world_size,
            },
            "parallel_plan": plan.to_dict(),
        }
        progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
        generator = torch.Generator(device=resolved_device).manual_seed((seed + rank) % (2**63 - 1))
        checkpoint_model = CosmosDMD2TrainableRoles(
            student.trainable_module,
            guidance.module,
        )
        checkpoint_state = TrainingState(
            model=checkpoint_model,
            optimizer=(stack.student_optimizer, stack.guidance_optimizer),
            engine=stack.engine,
            dataloader=dmd2_loader,
            objective_generator=generator,
            progress=progress,
            identity={
                "recipe": recipe.to_dict(),
                "roles": roles.runtime_identity(),
                "data": data_identity,
                "initialization_seed": seed,
            },
            ignore_frozen_parameters=False,
            **stack.checkpoint_state_kwargs(),
        )
        checkpointer = TrainingCheckpointer(destination / "checkpoints")
        resume_artifact = None
        if resume_checkpoint is not None:
            resume_artifact = checkpointer.load(checkpoint_state, resume_checkpoint)

        def event_sink(event: Mapping[str, object]) -> None:
            if rank == 0:
                append_jsonl_durable(
                    destination / "metrics.jsonl",
                    {**dict(event), "run_id": recipe.run.id, "recorded_at": utc_now_iso()},
                    root=destination,
                )

        session = NativeDMD2TrainingSession(
            stack.engine,
            dmd2_loader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=recipe.checkpoint.save_every_steps,
            asynchronous_checkpoints=recipe.checkpoint.async_save,
            event_sink=event_sink,
        )
        return CosmosPredict25DMD2TrainingRun(
            recipe=recipe,
            session=session,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            roles=roles,
            output_dir=destination,
            data_identity=data_identity,
            resume_artifact=resume_artifact,
            distributed_context=distributed_context,
        )
    except Exception:
        if distributed_context is not None:
            distributed_context.close()
        raise


__all__ = [
    "COSMOS_PREDICT25_DMD2_FEATURE_IDS",
    "COSMOS_PREDICT25_DMD2_PRETRAINED_REVISION",
    "COSMOS_PREDICT25_DMD2_PRETRAINED_WEIGHT_FILE",
    "COSMOS_PREDICT25_DMD2_REPO_ID",
    "COSMOS_PREDICT25_DMD2_RUN_SCHEMA",
    "CosmosDMD2TrainableRoles",
    "CosmosPredict25DMD2RoleBundle",
    "CosmosPredict25DMD2TrainingRun",
    "cosmos_predict25_dmd2_lr_multiplier",
    "cosmos_predict25_dmd2_pretrained_checkpoint",
    "materialize_cosmos_predict25_dmd2_training_run",
    "validate_cosmos_predict25_dmd2_cache",
]
