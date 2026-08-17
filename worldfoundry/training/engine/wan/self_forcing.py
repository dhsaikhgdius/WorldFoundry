"""Native Wan Self-Forcing role materialization and training lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.core.io.integrity import append_jsonl_durable
from worldfoundry.core.time import utc_now_iso
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.distributed.fsdp import (
    FSDP2Application,
    apply_fsdp2,
    apply_fsdp2_frozen_reference,
)
from worldfoundry.training.models.causal_wan import (
    SELF_FORCING_ODE_CHECKPOINT,
    CausalWanTrainRole,
    load_causal_wan_1p3b,
    validate_causal_wan_dtype,
)
from worldfoundry.training.post_training.distillation.self_forcing.batching import (
    NativeSelfForcingDataLoader,
)
from worldfoundry.training.post_training.distillation.self_forcing.builder import (
    build_native_self_forcing_training_stack,
)
from worldfoundry.training.post_training.distillation.self_forcing.session import (
    NativeSelfForcingTrainingSession,
)
from worldfoundry.training.post_training.distillation.self_forcing.wan import (
    WanSelfForcingChunkAdapter,
)
from worldfoundry.training.post_training.shared.distributed import (
    PostTrainingParallelContext,
)
from worldfoundry.training.post_training.shared.prediction import (
    NativeFlowPredictionAdapter,
)
from worldfoundry.training.post_training.shared.role_checkpoints import (
    resolve_role_checkpoint,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from .roles import (
    DMDTrainableRoles,
    apply_wan_tuning,
    load_wan_role_adapter,
    torch_dtype,
    validate_model_dtype,
)
from .rollout_materialization import (
    build_wan_rollout_source,
    prepare_wan_rollout_assets,
    wan_rollout_latent_shape,
)
from .self_forcing_recipe import validate_wan_self_forcing_recipe
from .self_forcing_roles import (
    WanSelfForcingRoleBundle,
    apply_causal_wan_tuning,
)
from .self_forcing_run import (
    WAN_SELF_FORCING_RUN_SCHEMA,
    WanSelfForcingTrainingRun,
)


def _role_overrides(values: Mapping[str, object] | None) -> dict[str, object]:
    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec

    overrides = {str(name): value for name, value in dict(values or {}).items()}
    allowed = {"student", "real-score", "fake-score"}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown Wan Self-Forcing role overrides: {unknown}")
    for name, value in overrides.items():
        if not isinstance(value, CheckpointSpec):
            raise TypeError(f"audited role override {name!r} must be CheckpointSpec")
    return overrides


def materialize_wan_self_forcing_training_run(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    audited_role_overrides: Mapping[str, object] | None = None,
    audited_component_overrides: Mapping[str, object] | None = None,
    force_torch_attention: bool = True,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> WanSelfForcingTrainingRun:
    """Materialize the causal student and independent DMD score roles."""

    algorithm, data_plan = validate_wan_self_forcing_recipe(recipe)
    assets = prepare_wan_rollout_assets(
        recipe,
        data_plan,
        frame_factor=1,
        base_dir=base_dir,
        device=device,
        reward_device=None,
        output_dir=output_dir,
        audited_component_overrides=audited_component_overrides,
        force_torch_attention=force_torch_attention,
        initialization_seed=initialization_seed,
    )
    try:
        from worldfoundry.base_models.diffusion_model.recipes.registry import (
            default_native_diffusion_registry,
        )

        registry = default_native_diffusion_registry()
        real_native_recipe = registry.resolve(algorithm.real_score_model_recipe)
        fake_native_recipe = registry.resolve(algorithm.fake_score_model_recipe)
        for role_name, native_recipe in (
            ("real-score", real_native_recipe),
            ("fake-score", fake_native_recipe),
        ):
            if native_recipe.options.get("latent_channels", 16) != 16:
                raise ValueError(f"Self-Forcing {role_name} must use 16-channel Wan latents")

        overrides = _role_overrides(audited_role_overrides)
        student_checkpoint = resolve_role_checkpoint(
            role="student",
            reference=recipe.model.checkpoint,
            native_default=SELF_FORCING_ODE_CHECKPOINT,
            local_override=overrides.get("student"),  # type: ignore[arg-type]
        )
        real_checkpoint = resolve_role_checkpoint(
            role="real-score",
            reference=algorithm.real_score_checkpoint,
            native_default=real_native_recipe.checkpoints["dit"],
            local_override=overrides.get("real-score"),  # type: ignore[arg-type]
        )
        fake_checkpoint = resolve_role_checkpoint(
            role="fake-score",
            reference=algorithm.fake_score_checkpoint,
            native_default=fake_native_recipe.checkpoints["dit"],
            local_override=overrides.get("fake-score"),  # type: ignore[arg-type]
        )

        dtype = assets.dtype
        student_graph = load_causal_wan_1p3b(
            student_checkpoint.checkpoint,
            device=assets.device,
            dtype=dtype,
            gradient_checkpointing=recipe.runtime.activation_checkpoint == "full",
        )
        student = CausalWanTrainRole(student_graph)
        validate_causal_wan_dtype(student, dtype)
        student_peft = apply_causal_wan_tuning(recipe, student)
        student_fsdp: FSDP2Application | None = None
        mesh = None
        if assets.distributed_context is not None:
            mesh = assets.parallel_plan.build_device_mesh(assets.device.type)
            student_fsdp = apply_fsdp2(
                student,
                plan=assets.parallel_plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
            )

        score_options = {
            "assembler": assets.assembler,
            "device": assets.device,
            "dtype": dtype,
            "num_train_timesteps": algorithm.num_train_timesteps,
            "gradient_checkpointing": recipe.runtime.activation_checkpoint == "full",
            "force_torch_attention": force_torch_attention,
        }
        real_score = load_wan_role_adapter(
            native_recipe=real_native_recipe,
            checkpoint=real_checkpoint,
            **score_options,
        )
        validate_model_dtype(real_score, dtype)
        real_score.trainable_module.requires_grad_(False)
        real_score.trainable_module.eval()
        real_score_fsdp: FSDP2Application | None = None
        if assets.distributed_context is not None:
            assert mesh is not None
            real_score_fsdp = apply_fsdp2_frozen_reference(
                real_score,
                plan=assets.parallel_plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
            )

        fake_score = load_wan_role_adapter(
            native_recipe=fake_native_recipe,
            checkpoint=fake_checkpoint,
            **score_options,
        )
        validate_model_dtype(fake_score, dtype)
        fake_score_peft = apply_wan_tuning(recipe, fake_score)
        fake_score_fsdp: FSDP2Application | None = None
        if assets.distributed_context is not None:
            assert mesh is not None
            fake_score_fsdp = apply_fsdp2(
                fake_score,
                plan=assets.parallel_plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
            )

        source = build_wan_rollout_source(
            assets,
            recipe,
            data_plan,
            fake_score,
            guidance_scale=algorithm.teacher_guidance_scale,
            requires_unconditional=True,
        )
        if source.unconditional is None:
            raise RuntimeError("Self-Forcing DMD requires unconditional score conditioning")
        latent_shape = wan_rollout_latent_shape(fake_score, assets.generation_geometry)
        if latent_shape[1] % algorithm.frames_per_block:
            raise RuntimeError("materialized latent geometry differs from Self-Forcing block geometry")
        loader = NativeSelfForcingDataLoader(
            source.loader,
            latent_shape=latent_shape,
            device=assets.device,
            dtype=dtype,
            shared_unconditional_conditioning=source.unconditional.tensors,
        )
        autocast_dtype = None if dtype is torch.float32 else dtype
        stack = build_native_self_forcing_training_stack(
            recipe,
            student=WanSelfForcingChunkAdapter(
                student.trainable_module,
                frames_per_block=algorithm.frames_per_block,
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
            parallel_context=PostTrainingParallelContext.current(),
            fused_adamw=fused_adamw,
        )
        roles = WanSelfForcingRoleBundle(
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
        height, width, frames = assets.generation_geometry
        data_identity = {
            "prompt_records": [record.to_dict() for record in assets.prompts],
            "conditioning_index": assets.conditioning.index.to_dict(),
            "model_contract": dict(assets.model_contract),
            "conditioner": dict(assets.conditioner),
            "tokenizer": dict(assets.tokenizer),
            "sample_count": len(assets.conditioning),
            "generation": {
                "height": height,
                "width": width,
                "num_frames": frames,
            },
            "latent_shape": list(latent_shape),
            "prompt_batch_size": data_plan.prompt_batch_size,
            "tail_policy": recipe.data.tail_policy,
            "unconditional_conditioning": source.unconditional.artifact.to_dict(),
            "parallel_plan": assets.parallel_plan.to_dict(),
        }
        progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
        identity = {
            "schema": "worldfoundry-wan-self-forcing-resume-identity",
            "recipe": recipe.to_dict(),
            "roles": roles.runtime_identity(),
            "data": data_identity,
            "runtime": recipe.to_dict()["runtime"],
            "distributed": recipe.to_dict()["distributed"],
            "tuning": recipe.to_dict()["tuning"],
            "initialization_seed": assets.base_seed,
            "rank_seed_derivation": "base-seed-plus-global-rank",
        }
        checkpoint_state = TrainingState(
            model=DMDTrainableRoles(
                student.trainable_module,
                fake_score.trainable_module,
            ),
            optimizer=(stack.student_optimizer, stack.fake_score_optimizer),
            engine=stack.engine,
            dataloader=loader,
            objective_generator=source.generator,
            progress=progress,
            identity=identity,
            ignore_frozen_parameters=recipe.tuning.mode == "lora",
            **stack.checkpoint_state_kwargs(),
        )
        checkpointer = TrainingCheckpointer(assets.output_dir / "checkpoints")
        resume_artifact = None
        if resume_checkpoint is not None:
            resume_artifact = checkpointer.load(checkpoint_state, resume_checkpoint)

        def event_sink(event: Mapping[str, object]) -> None:
            if assets.rank != 0:
                return
            append_jsonl_durable(
                assets.output_dir / "metrics.jsonl",
                {
                    **dict(event),
                    "run_id": recipe.run.id,
                    "recorded_at": utc_now_iso(),
                },
                root=assets.output_dir,
            )

        session = NativeSelfForcingTrainingSession(
            stack.engine,
            loader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=recipe.checkpoint.save_every_steps,
            asynchronous_checkpoints=recipe.checkpoint.async_save,
            event_sink=event_sink,
        )
        return WanSelfForcingTrainingRun(
            recipe=recipe,
            session=session,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            roles=roles,  # type: ignore[arg-type]
            output_dir=assets.output_dir,
            data_identity=data_identity,
            resume_artifact=resume_artifact,
            distributed_context=assets.distributed_context,
        )
    except Exception:
        if assets.distributed_context is not None:
            assets.distributed_context.close()
        raise


__all__ = [
    "WAN_SELF_FORCING_RUN_SCHEMA",
    "WanSelfForcingRoleBundle",
    "WanSelfForcingTrainingRun",
    "materialize_wan_self_forcing_training_run",
]
