"""Native Wan role materialization and DMD run lifecycle."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.core.io.integrity import (
    append_jsonl_durable,
)
from worldfoundry.core.time import utc_now_iso
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.shared_conditioning import (
    SharedConditioningSample,
    SharedConditioningStore,
)
from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.distributed.fsdp import (
    FSDP2Application,
    apply_fsdp2,
    apply_fsdp2_frozen_reference,
)
from worldfoundry.training.distributed.parallel import (
    DistributedTrainingContext,
    ParallelPlan,
)
from worldfoundry.training.models.wan import WanTrainAdapter
from worldfoundry.training.post_training.distillation.dmd.batching import NativeDMDDataLoader
from worldfoundry.training.post_training.distillation.dmd.builder import (
    build_native_dmd_training_stack,
)
from worldfoundry.training.post_training.distillation.dmd.session import (
    NativeDMDTrainingSession,
)
from worldfoundry.training.post_training.shared.distributed import (
    PostTrainingParallelContext,
)
from worldfoundry.training.post_training.shared.prediction import NativeFlowPredictionAdapter
from worldfoundry.training.post_training.shared.role_checkpoints import resolve_role_checkpoint
from worldfoundry.training.recipes.post_training.algorithms.dmd import DMDAlgorithmSpec
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ..artifacts import create_run_directory
from .cache import (
    audit_wan_cache_against_manifest,
    build_wan_cache_loader,
    validate_wan_cache_contract,
)
from .dmd_run import WAN_DMD_RUN_SCHEMA, WanDMDTrainingRun
from .roles import (
    DMDTrainableRoles,
    WanDMDRoleBundle,
    apply_wan_tuning,
    load_wan_role_adapter,
    seed_initialization,
    torch_dtype,
    validate_model_dtype,
)


def _validate_recipe(recipe: PostTrainingRecipe) -> DMDAlgorithmSpec:
    if not isinstance(recipe, PostTrainingRecipe):
        raise TypeError("recipe must be PostTrainingRecipe")
    if not isinstance(recipe.algorithm, DMDAlgorithmSpec):
        raise TypeError("Wan DMD materialization requires algorithm.type='dmd'")
    if recipe.model.recipe != "wan2.1-t2v-1.3b":
        raise ValueError("native Wan DMD currently requires wan2.1-t2v-1.3b")
    if recipe.data.cache is None:
        raise ValueError("Wan DMD requires an immutable data.cache")
    if recipe.data.max_latent_tokens_per_microbatch is None:
        raise ValueError("Wan DMD requires data.max_latent_tokens_per_microbatch")
    if recipe.data.tail_policy not in {"drop", "pad"}:
        raise ValueError("Wan DMD token batching requires data.tail_policy='drop' or 'pad'")
    if recipe.distributed.backend not in {"single", "fsdp2"}:
        raise ValueError("Wan DMD currently supports single or FSDP2 execution")
    if recipe.runtime.activation_checkpoint not in {"none", "full"}:
        raise ValueError("Wan DMD activation_checkpoint must be 'none' or 'full'")
    if recipe.tuning.mode == "partial":
        raise ValueError("partial Wan DMD tuning needs an explicit parameter policy")
    return recipe.algorithm


def _validate_unconditional_cache(
    sample: SharedConditioningSample,
    dataset: VideoCachedDataset,
    adapter: WanTrainAdapter,
    *,
    expected_contract: Mapping[str, object],
) -> None:
    identity = sample.artifact.identity
    if identity.branch != "unconditional":
        raise ValueError("Wan DMD shared conditioning must use the unconditional branch")
    if identity.model_recipe != expected_contract["model_recipe"]:
        raise ValueError("Wan DMD unconditional context belongs to another model contract")
    if any(
        entry.provenance.conditioner != identity.conditioner
        or entry.provenance.tokenizer != identity.tokenizer
        for entry in dataset.index.entries
    ):
        raise ValueError("Wan DMD unconditional context encoder identity differs from sample contexts")
    if set(sample.tensors) != {"context"}:
        raise ValueError("Wan DMD unconditional conditioning must contain only context")
    descriptor = identity.tensors["context"]
    expected_shape = (
        adapter.expected_text_length,
        adapter.expected_context_features,
    )
    if descriptor.shape != expected_shape or descriptor.layout != "sequence-features":
        raise ValueError("Wan DMD unconditional context tensor contract is incompatible")


def materialize_wan_dmd_training_run(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    audited_role_overrides: Mapping[str, object] | None = None,
    verify_media_files: bool = True,
    audit_cache_on_open: bool = True,
    verify_cache_on_read: bool = True,
    force_torch_attention: bool = True,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> WanDMDTrainingRun:
    """Materialize an independently owned native Wan three-role DMD run."""

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
            raise ValueError("native Wan DMD FSDP2 materialization requires CUDA")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device
    try:
        world_size = 1 if distributed_context is None else distributed_context.world_size
        rank = 0 if distributed_context is None else distributed_context.rank
        plan = ParallelPlan.resolve(recipe.distributed, world_size=world_size)
        create_run_directory(destination, distributed_context)
        if force_torch_attention:
            for name in (
                "WORLDFOUNDRY_ATTENTION_IMPLEMENTATION",
                "WORLDFOUNDRY_ATTENTION_BACKEND",
            ):
                configured = os.environ.get(name)
                if configured not in {None, "torch"}:
                    raise ValueError(f"correctness-first Wan DMD requires {name}=torch; got {configured!r}")
                os.environ[name] = "torch"

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
        unconditional = SharedConditioningStore(cache_path).read("unconditional")

        from worldfoundry.base_models.diffusion_model.assembly import (
            NativeDiffusionAssembler,
        )
        from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
        from worldfoundry.base_models.diffusion_model.recipes.registry import (
            default_native_diffusion_registry,
        )

        native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
        default_dit = native_recipe.checkpoints["dit"]
        raw_overrides = dict(audited_role_overrides or {})
        unknown_overrides = sorted(set(raw_overrides) - {"student", "real-score", "fake-score"})
        if unknown_overrides:
            raise ValueError(f"unknown Wan DMD role overrides: {unknown_overrides}")
        for name, value in raw_overrides.items():
            if not isinstance(value, CheckpointSpec):
                raise TypeError(f"audited role override {name!r} must be CheckpointSpec")
        student_checkpoint = resolve_role_checkpoint(
            role="student",
            reference=recipe.model.checkpoint,
            native_default=default_dit,
            local_override=raw_overrides.get("student"),
        )
        real_checkpoint = resolve_role_checkpoint(
            role="real-score",
            reference=algorithm.real_score_checkpoint,
            native_default=default_dit,
            local_override=raw_overrides.get("real-score"),
        )
        fake_checkpoint = resolve_role_checkpoint(
            role="fake-score",
            reference=algorithm.fake_score_checkpoint,
            native_default=default_dit,
            local_override=raw_overrides.get("fake-score"),
        )
        assembler = NativeDiffusionAssembler()
        dtype = torch_dtype(recipe.runtime.param_dtype)
        adapter_options = {
            "assembler": assembler,
            "native_recipe": native_recipe,
            "device": resolved_device,
            "dtype": dtype,
            "num_train_timesteps": algorithm.num_train_timesteps,
            "gradient_checkpointing": recipe.runtime.activation_checkpoint == "full",
            "force_torch_attention": force_torch_attention,
        }
        seed_initialization(initialization_seed)
        student = load_wan_role_adapter(
            checkpoint=student_checkpoint,
            **adapter_options,
        )
        validate_model_dtype(student, dtype)
        student_peft = apply_wan_tuning(recipe, student)
        student_fsdp: FSDP2Application | None = None
        mesh = None
        if distributed_context is not None:
            mesh = plan.build_device_mesh(resolved_device.type)
            student_fsdp = apply_fsdp2(
                student,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
            )

        real_score = load_wan_role_adapter(
            checkpoint=real_checkpoint,
            **adapter_options,
        )
        validate_model_dtype(real_score, dtype)
        real_score.trainable_module.requires_grad_(False)
        real_score.trainable_module.eval()
        real_score_fsdp: FSDP2Application | None = None
        if distributed_context is not None:
            assert mesh is not None
            real_score_fsdp = apply_fsdp2_frozen_reference(
                real_score,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
            )

        fake_score = load_wan_role_adapter(
            checkpoint=fake_checkpoint,
            **adapter_options,
        )
        validate_model_dtype(fake_score, dtype)
        fake_score_peft = apply_wan_tuning(recipe, fake_score)
        fake_score_fsdp: FSDP2Application | None = None
        if distributed_context is not None:
            assert mesh is not None
            fake_score_fsdp = apply_fsdp2(
                fake_score,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
            )

        expected_contract = validate_wan_cache_contract(recipe, student, cache)
        _validate_unconditional_cache(
            unconditional,
            cache,
            student,
            expected_contract=expected_contract,
        )
        source_loader, token_sampler = build_wan_cache_loader(
            recipe=recipe,
            dataset=cache,
            rank=rank,
            world_size=world_size,
            default_pin_memory=resolved_device.type == "cuda",
        )
        dmd_loader = NativeDMDDataLoader(
            source_loader,
            student,
            shared_unconditional_conditioning=unconditional.tensors,
        )
        autocast_dtype = None if dtype is torch.float32 else dtype
        parallel_context = PostTrainingParallelContext.current()
        stack = build_native_dmd_training_stack(
            recipe,
            student=NativeFlowPredictionAdapter(
                student,
                autocast_dtype=autocast_dtype,
                checkpoint_identity=recipe.model.checkpoint,
            ),
            real_score=NativeFlowPredictionAdapter(
                real_score,
                autocast_dtype=autocast_dtype,
                checkpoint_identity=algorithm.real_score_checkpoint,
            ),
            fake_score=NativeFlowPredictionAdapter(
                fake_score,
                autocast_dtype=autocast_dtype,
                checkpoint_identity=algorithm.fake_score_checkpoint,
            ),
            parallel_context=parallel_context,
            fused_adamw=fused_adamw,
        )
        roles = WanDMDRoleBundle(
            student=student,
            real_score=real_score,
            fake_score=fake_score,
            student_checkpoint=student_checkpoint,
            real_score_checkpoint=real_checkpoint,
            fake_score_checkpoint=fake_checkpoint,
            student_peft=student_peft,
            fake_score_peft=fake_score_peft,
            student_fsdp=student_fsdp,
            real_score_fsdp=real_score_fsdp,
            fake_score_fsdp=fake_score_fsdp,
        )
        progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed((int(initialization_seed or recipe.data.shuffle_seed) + rank) % (2**63 - 1))
        data_identity = {
            "cache_schema": cache.index.schema,
            "cache_index": cache.index.to_dict(),
            "cache_contract": dict(expected_contract),
            "sample_ids": list(cache.sample_ids),
            "sample_count": len(cache),
            "unconditional_conditioning": unconditional.artifact.to_dict(),
            "latent_token_budget": token_sampler.max_latent_tokens,
            "token_sampler": {
                "sample_ids": list(token_sampler.sample_ids),
                "bucket_keys": [key.to_dict() for key in token_sampler.bucket_keys],
                "batch_contracts": [dict(value) for value in token_sampler.batch_contracts],
                "seed": token_sampler.seed,
                "shuffle": token_sampler.shuffle,
                "tail_policy": token_sampler.tail_policy,
                "rank": token_sampler.rank,
                "world_size": token_sampler.world_size,
            },
            "parallel_plan": plan.to_dict(),
        }
        identity = {
            "schema": "worldfoundry-wan-dmd-resume-identity",
            "recipe": recipe.to_dict(),
            "roles": roles.runtime_identity(),
            "data": data_identity,
            "runtime": recipe.to_dict()["runtime"],
            "distributed": recipe.to_dict()["distributed"],
            "tuning": recipe.to_dict()["tuning"],
            "initialization_seed": initialization_seed,
            "rank_seed_derivation": "base-seed-plus-rank",
        }
        checkpoint_model = DMDTrainableRoles(
            student.trainable_module,
            fake_score.trainable_module,
        )
        checkpoint_state = TrainingState(
            model=checkpoint_model,
            optimizer=(stack.student_optimizer, stack.fake_score_optimizer),
            engine=stack.engine,
            dataloader=dmd_loader,
            objective_generator=generator,
            progress=progress,
            identity=identity,
            ignore_frozen_parameters=recipe.tuning.mode == "lora",
            **stack.checkpoint_state_kwargs(),
        )
        checkpointer = TrainingCheckpointer(destination / "checkpoints")
        resume_artifact = None
        if resume_checkpoint is not None:
            resume_artifact = checkpointer.load(checkpoint_state, resume_checkpoint)

        def event_sink(event: Mapping[str, object]) -> None:
            if rank != 0:
                return
            append_jsonl_durable(
                destination / "metrics.jsonl",
                {
                    **dict(event),
                    "run_id": recipe.run.id,
                    "recorded_at": utc_now_iso(),
                },
                root=destination,
            )

        session = NativeDMDTrainingSession(
            stack.engine,
            dmd_loader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=recipe.checkpoint.save_every_steps,
            asynchronous_checkpoints=recipe.checkpoint.async_save,
            event_sink=event_sink,
        )
        return WanDMDTrainingRun(
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
    "DMDTrainableRoles",
    "WAN_DMD_RUN_SCHEMA",
    "WanDMDRoleBundle",
    "WanDMDTrainingRun",
    "materialize_wan_dmd_training_run",
]
