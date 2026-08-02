"""Materialize a real SANA-Sprint SCM-LADD run on WorldFoundry infrastructure."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.loader import build_stateful_dataloader
from worldfoundry.training.data.sampler import DeterministicDistributedSampler
from worldfoundry.training.data.sana_cache import (
    SanaCachedDataset,
    collate_sana_cached_samples,
)
from worldfoundry.training.data.shared_conditioning import SharedConditioningStore
from worldfoundry.training.distributed.fsdp import (
    FSDP2Application,
    apply_fsdp2,
    apply_fsdp2_frozen_reference,
)
from worldfoundry.training.distributed.parallel import DistributedTrainingContext, ParallelPlan
from worldfoundry.training.models.sana import SanaTrainAdapter, build_cached_sana_train_adapter
from worldfoundry.training.models.sana_scm_ladd import (
    SanaLADDDiscriminatorAdapter,
    SanaSCMVelocityAdapter,
)
from worldfoundry.training.post_training.distillation.scm_ladd.builder import (
    build_native_scm_ladd_training_stack,
)
from worldfoundry.training.post_training.distillation.scm_ladd.session import (
    NativeSCMLADDTrainingSession,
)
from worldfoundry.training.post_training.shared.distributed import PostTrainingParallelContext
from worldfoundry.training.post_training.shared.role_checkpoints import resolve_role_checkpoint
from worldfoundry.training.recipes.post_training.algorithms.scm_ladd import (
    SCMLADDAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.peft import PeftLoraApplication, apply_peft_lora_to_adapter

from ..artifacts import create_run_directory
from .cache import audit_sana_cache_against_manifest, validate_sana_cache_contract
from .scm_ladd_data import SanaSCMLADDDataLoader, audit_sana_scm_ladd_unconditional
from .scm_ladd_roles import SanaSCMLADDRoleBundle, SanaSCMLADDTrainableRoles
from .scm_ladd_run import SANA_SCM_LADD_RUN_SCHEMA, SanaSCMLADDTrainingRun

_CACHE_OPTIONS = frozenset(
    {
        "microbatch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "snapshot_every_n_steps",
    }
)


def _dtype(value: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _strict_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")
    return value


def _seed(seed: int | None) -> None:
    if seed is None:
        return
    if isinstance(seed, bool):
        raise TypeError("initialization_seed must be an integer")
    resolved = int(seed) % (2**63 - 1)
    random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)


def _validate_recipe(recipe: PostTrainingRecipe) -> SCMLADDAlgorithmSpec:
    if not isinstance(recipe, PostTrainingRecipe):
        raise TypeError("recipe must be PostTrainingRecipe")
    if not isinstance(recipe.algorithm, SCMLADDAlgorithmSpec):
        raise TypeError("SANA SCM-LADD requires algorithm.type='scm-ladd'")
    if recipe.model.recipe not in {
        "sana-sprint-600m-1024px",
        "sana-sprint-1600m-1024px",
    }:
        raise ValueError("official SANA SCM-LADD requires a supported 0.6B or 1.6B Sprint recipe")
    if recipe.data.cache is None:
        raise ValueError("SANA SCM-LADD requires an immutable data.cache")
    if recipe.distributed.backend not in {"single", "fsdp2"}:
        raise ValueError("SANA SCM-LADD supports single or FSDP2 execution")
    if recipe.distributed.backend == "fsdp2" and recipe.data.tail_policy not in {"drop", "pad"}:
        raise ValueError("multi-rank SANA SCM-LADD requires data.tail_policy='drop' or 'pad'")
    if recipe.runtime.activation_checkpoint != "none":
        raise ValueError("SANA SCM-LADD activation checkpointing is not implemented")
    if recipe.runtime.compile:
        raise ValueError("SANA SCM-LADD does not support torch.compile")
    if recipe.runtime.reduce_dtype != "float32":
        raise ValueError("SANA SCM-LADD objective reduction must use float32")
    if recipe.optimizer.type != "came" or (
        recipe.discriminator_optimizer is None or recipe.discriminator_optimizer.type != "came"
    ):
        raise ValueError("official SANA SCM-LADD requires CAME for both mutable roles")
    if recipe.tuning.mode not in {"full", "lora"}:
        raise ValueError("SANA SCM-LADD supports full or LoRA student tuning")
    return recipe.algorithm


def _validate_dtype(adapter: SanaTrainAdapter | SanaSCMVelocityAdapter, expected: torch.dtype) -> None:
    module = adapter.trainable_module
    values = {parameter.dtype for parameter in module.parameters() if parameter.is_floating_point()}
    if values != {expected}:
        raise ValueError(
            "loaded SANA role dtype differs from runtime.param_dtype: "
            f"loaded={sorted(map(str, values))}, expected={expected}"
        )


def _apply_student_tuning(
    recipe: PostTrainingRecipe,
    adapter: SanaTrainAdapter,
) -> PeftLoraApplication | None:
    if recipe.tuning.mode == "full":
        adapter.trainable_module.requires_grad_(True)
        return None
    assert recipe.tuning.preset is not None
    assert recipe.tuning.rank is not None
    assert recipe.tuning.alpha is not None
    if recipe.tuning.preset != adapter.lora_target_preset:
        raise ValueError("recipe LoRA preset differs from the SANA adapter")
    return apply_peft_lora_to_adapter(
        adapter,
        preset=recipe.tuning.preset,
        rank=recipe.tuning.rank,
        alpha=recipe.tuning.alpha,
        dropout=recipe.tuning.dropout,
        modules_to_save=recipe.tuning.modules_to_save,
    )


def materialize_sana_scm_ladd_training_run(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    audited_role_overrides: Mapping[str, object] | None = None,
    verify_media_hashes: bool = True,
    audit_cache_on_open: bool = True,
    verify_cache_on_read: bool = True,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> SanaSCMLADDTrainingRun:
    """Build independent student/teacher/head roles and compound exact resume."""

    algorithm = _validate_recipe(recipe)
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
            raise ValueError("SANA SCM-LADD FSDP2 materialization requires CUDA")
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
        unconditional = SharedConditioningStore(cache_path).read("unconditional")
        audit_sana_scm_ladd_unconditional(unconditional, cache)

        from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
        from worldfoundry.base_models.diffusion_model.components import (
            BuildPurpose,
            ComponentBuildContext,
            ComponentKey,
            ComponentKind,
        )
        from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
        from worldfoundry.base_models.diffusion_model.models.denoisers.sana import (
            build_sana_sprint_teacher_denoiser,
        )
        from worldfoundry.base_models.diffusion_model.optimizations import (
            AttentionBackend,
            RuntimePolicy,
        )
        from worldfoundry.base_models.diffusion_model.recipes.registry import (
            default_native_diffusion_registry,
        )

        native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
        if "teacher" not in native_recipe.checkpoints:
            raise ValueError("native SANA recipe does not declare the official SCM teacher checkpoint")
        raw_overrides = dict(audited_role_overrides or {})
        unknown = sorted(set(raw_overrides) - {"student", "teacher"})
        if unknown:
            raise ValueError(f"unknown SANA SCM-LADD role overrides: {unknown}")
        for name, value in raw_overrides.items():
            if not isinstance(value, CheckpointSpec):
                raise TypeError(f"audited role override {name!r} must be CheckpointSpec")
        student_checkpoint = resolve_role_checkpoint(
            role="student",
            reference=recipe.model.checkpoint,
            native_default=native_recipe.checkpoints["dit"],
            audited_local_override=raw_overrides.get("student"),
        )
        teacher_checkpoint = resolve_role_checkpoint(
            role="teacher",
            reference=algorithm.teacher_checkpoint,
            native_default=native_recipe.checkpoints["teacher"],
            audited_local_override=raw_overrides.get("teacher"),
        )

        dtype = _dtype(recipe.runtime.param_dtype)
        autocast_dtype = None if dtype is torch.float32 else dtype
        policy = RuntimePolicy(
            device=resolved_device,
            dtype=dtype,
            attention=AttentionBackend.TORCH,
        )
        denoiser_key = ComponentKey(ComponentKind.DENOISER)
        denoiser_spec = next(spec for spec in native_recipe.components if spec.key == denoiser_key)
        assembler = NativeDiffusionAssembler()
        _seed(initialization_seed)
        student_components = assembler.build_components(
            native_recipe,
            purpose=BuildPurpose.TRAINING,
            policy=policy,
            checkpoint_overrides={"dit": student_checkpoint.checkpoint},
            component_keys=(denoiser_key,),
        )
        student_preparation = build_cached_sana_train_adapter(
            student_components,
            expected_latent_channels=int(native_recipe.options.get("latent_channels", 32)),
            spatial_compression=int(native_recipe.options.get("spatial_compression", 32)),
        )
        _validate_dtype(student_preparation, dtype)
        student_peft = _apply_student_tuning(recipe, student_preparation)
        student = SanaSCMVelocityAdapter(
            student_preparation.denoiser,
            role="student",
            checkpoint_identity=recipe.model.checkpoint,
            expected_latent_channels=student_preparation.expected_latent_channels,
            autocast_dtype=autocast_dtype,
        )

        teacher_denoiser = build_sana_sprint_teacher_denoiser(
            ComponentBuildContext(
                model_id=native_recipe.model_id,
                key=denoiser_key,
                policy=policy,
                purpose=BuildPurpose.TRAINING,
                checkpoints={"weights": teacher_checkpoint.checkpoint},
                recipe_options=native_recipe.options,
                component_options=denoiser_spec.options,
            )
        )
        teacher = SanaSCMVelocityAdapter(
            teacher_denoiser,
            role="teacher",
            checkpoint_identity=algorithm.teacher_checkpoint,
            expected_latent_channels=student_preparation.expected_latent_channels,
            autocast_dtype=autocast_dtype,
        )
        _validate_dtype(teacher, dtype)

        from worldfoundry.base_models.diffusion_model.models.networks.sana.ladd import (
            SANAFeatureDiscriminatorHeads,
        )

        heads = SANAFeatureDiscriminatorHeads(
            hidden_size=int(getattr(teacher.module, "hidden_size")),
            block_ids=algorithm.discriminator_head_block_ids,
        ).to(device=resolved_device)
        discriminator = SanaLADDDiscriminatorAdapter(
            teacher,
            heads,
            autocast_dtype=autocast_dtype,
        )

        student_fsdp: FSDP2Application | None = None
        teacher_fsdp: FSDP2Application | None = None
        discriminator_fsdp: FSDP2Application | None = None
        if distributed_context is not None:
            mesh = plan.build_device_mesh(resolved_device.type)
            reduce_dtype = _dtype(recipe.runtime.reduce_dtype)
            student_fsdp = apply_fsdp2(
                student_preparation,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=reduce_dtype,
            )
            teacher_fsdp = apply_fsdp2_frozen_reference(
                teacher,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=reduce_dtype,
            )
            discriminator_fsdp = apply_fsdp2(
                discriminator,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=reduce_dtype,
            )

        loader_options = dict(recipe.data.options)
        unknown_loader = sorted(set(loader_options) - _CACHE_OPTIONS)
        if unknown_loader:
            raise ValueError(f"unsupported SANA SCM-LADD data options: {unknown_loader}")
        microbatch_size = _positive_int(
            loader_options.pop("microbatch_size", 1),
            field_name="data.options.microbatch_size",
        )
        expected_contract = validate_sana_cache_contract(
            recipe,
            student_preparation,
            cache,
            microbatch_size=microbatch_size,
        )
        sampler = DeterministicDistributedSampler(
            cache,
            dataset_digest=cache.dataset_digest,
            seed=recipe.data.shuffle_seed,
            shuffle=recipe.data.shuffle,
            rank=rank,
            world_size=world_size,
            tail_policy=recipe.data.tail_policy,
        )
        if len(sampler) < microbatch_size:
            raise ValueError("SANA SCM-LADD microbatch_size would leave this rank without a batch")
        workers = int(loader_options.pop("num_workers", 0))
        pin_memory = _strict_bool(
            loader_options.pop("pin_memory", resolved_device.type == "cuda"),
            field_name="data.options.pin_memory",
        )
        persistent_workers = _strict_bool(
            loader_options.pop("persistent_workers", False),
            field_name="data.options.persistent_workers",
        )
        prefetch = loader_options.pop("prefetch_factor", None)
        snapshot = _positive_int(
            loader_options.pop("snapshot_every_n_steps", 1),
            field_name="data.options.snapshot_every_n_steps",
        )
        source_loader = build_stateful_dataloader(
            cache,
            sampler,
            batch_size=microbatch_size,
            collate_fn=collate_sana_cached_samples,
            num_workers=workers,
            worker_seed=recipe.data.shuffle_seed,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=None if prefetch is None else int(prefetch),
            snapshot_every_n_steps=snapshot,
        )
        scm_loader = SanaSCMLADDDataLoader(
            source_loader,
            adapter=student_preparation,
            unconditional=unconditional,
        )
        parallel_context = PostTrainingParallelContext.current()
        stack = build_native_scm_ladd_training_stack(
            recipe,
            student=student,
            teacher=teacher,
            discriminator=discriminator,
            parallel_context=parallel_context,
            fused_adamw=fused_adamw,
        )
        roles = SanaSCMLADDRoleBundle(
            student_preparation=student_preparation,
            student=student,
            teacher=teacher,
            discriminator=discriminator,
            student_checkpoint=student_checkpoint,
            teacher_checkpoint=teacher_checkpoint,
            student_peft=student_peft,
            student_fsdp=student_fsdp,
            teacher_fsdp=teacher_fsdp,
            discriminator_fsdp=discriminator_fsdp,
        )
        progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed((int(initialization_seed or recipe.data.shuffle_seed) + rank) % (2**63 - 1))
        data_identity = {
            "cache_index_sha256": cache.index_sha256,
            "cache_contract_sha256": expected_contract,
            "dataset_digest": cache.dataset_digest,
            "unconditional_identity_sha256": unconditional.artifact.identity_sha256,
            "parallel_plan_digest": plan.digest,
        }
        identity = {
            "schema": "worldfoundry-sana-scm-ladd-resume-identity",
            "recipe_digest": recipe.digest,
            "roles": roles.runtime_identity(),
            "data": data_identity,
            "initialization_seed": initialization_seed,
        }
        checkpoint_model = SanaSCMLADDTrainableRoles(student.module, discriminator.module)
        checkpoint_state = TrainingState(
            model=checkpoint_model,
            optimizer=(stack.student_optimizer, stack.discriminator_optimizer),
            engine=stack.engine,
            dataloader=scm_loader,
            objective_generator=generator,
            progress=progress,
            identity=identity,
            ignore_frozen_parameters=recipe.tuning.mode == "lora",
            **stack.checkpoint_state_kwargs(),
        )
        checkpointer = TrainingCheckpointer(destination / "checkpoints")
        resume_artifact = (
            None
            if resume_checkpoint is None
            else checkpointer.load(checkpoint_state, resume_checkpoint)
        )
        session = NativeSCMLADDTrainingSession(
            stack.engine,
            scm_loader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=recipe.checkpoint.save_every_steps,
            asynchronous_checkpoints=recipe.checkpoint.async_save,
        )
        return SanaSCMLADDTrainingRun(
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
    "SANA_SCM_LADD_RUN_SCHEMA",
    "SanaSCMLADDTrainingRun",
    "materialize_sana_scm_ladd_training_run",
]
