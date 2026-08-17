"""Native Wan flow-policy role materialization and run lifecycle."""

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
from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (
    build_native_flow_policy_training_stack,
)
from worldfoundry.training.post_training.rl.batching import NativeFlowPolicyDataLoader
from worldfoundry.training.post_training.shared.building import (
    require_checkpoint_identity,
    require_independent_modules,
)
from worldfoundry.training.post_training.shared.distributed import (
    PostTrainingParallelContext,
)
from worldfoundry.training.post_training.shared.prediction import NativeFlowPredictionAdapter
from worldfoundry.training.post_training.shared.role_checkpoints import (
    resolve_role_checkpoint,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from .flow_policy_recipe import (
    WanFlowPolicyDataPlan,
    validate_wan_flow_policy_recipe,
)
from .flow_policy_run import (
    WAN_FLOW_POLICY_RUN_SCHEMA,
    WanFlowPolicyRoleBundle,
    WanFlowPolicyRunSummary,
    WanFlowPolicyTrainingRun,
)
from .roles import (
    apply_wan_tuning,
    load_wan_role_adapter,
    torch_dtype,
    validate_model_dtype,
)
from .rollout_materialization import (
    build_wan_rollout_source,
    build_wan_terminal_reward_adapter,
    prepare_wan_rollout_assets,
    wan_rollout_latent_shape,
)


def materialize_wan_flow_policy_training_run(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    reward_device: str | torch.device | None = None,
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    audited_role_overrides: Mapping[str, object] | None = None,
    audited_component_overrides: Mapping[str, object] | None = None,
    force_torch_attention: bool = True,
    videoalign_attention_implementation: str = "sdpa",
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> WanFlowPolicyTrainingRun:
    """Materialize a WorldFoundry-owned Wan flow-policy training run."""

    algorithm, data_plan = validate_wan_flow_policy_recipe(recipe)
    assets = prepare_wan_rollout_assets(
        recipe,
        data_plan,
        frame_factor=algorithm.reward_model.frame_factor,
        base_dir=base_dir,
        device=device,
        reward_device=reward_device,
        output_dir=output_dir,
        audited_component_overrides=audited_component_overrides,
        force_torch_attention=force_torch_attention,
        initialization_seed=initialization_seed,
    )
    distributed_context = assets.distributed_context

    try:
        from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec

        raw_role_overrides = dict(audited_role_overrides or {})
        unknown_role_overrides = sorted(set(raw_role_overrides) - {"policy", "reference"})
        if unknown_role_overrides:
            raise ValueError(f"unknown Wan flow-policy role overrides: {unknown_role_overrides}")
        if not algorithm.requires_reference_policy and "reference" in raw_role_overrides:
            raise ValueError("reference role override is unused by this flow-policy objective")
        for name, value in raw_role_overrides.items():
            if not isinstance(value, CheckpointSpec):
                raise TypeError(f"audited role override {name!r} must be CheckpointSpec")
        default_dit = assets.native_recipe.checkpoints["dit"]
        policy_checkpoint = resolve_role_checkpoint(
            role="policy",
            reference=recipe.model.checkpoint,
            native_default=default_dit,
            local_override=raw_role_overrides.get("policy"),
        )
        reference_checkpoint = None
        if algorithm.requires_reference_policy:
            assert algorithm.reference_checkpoint is not None
            reference_checkpoint = resolve_role_checkpoint(
                role="reference",
                reference=algorithm.reference_checkpoint,
                native_default=default_dit,
                local_override=raw_role_overrides.get("reference"),
            )

        dtype = assets.dtype
        adapter_options = {
            "assembler": assets.assembler,
            "native_recipe": assets.native_recipe,
            "device": assets.device,
            "dtype": dtype,
            "num_train_timesteps": algorithm.num_train_timesteps,
            "gradient_checkpointing": recipe.runtime.activation_checkpoint == "full",
            "force_torch_attention": force_torch_attention,
        }
        policy = load_wan_role_adapter(
            checkpoint=policy_checkpoint,
            **adapter_options,
        )
        validate_model_dtype(policy, dtype)
        policy_peft = apply_wan_tuning(recipe, policy)
        policy_fsdp: FSDP2Application | None = None
        mesh = None
        if distributed_context is not None:
            mesh = assets.parallel_plan.build_device_mesh(assets.device.type)
            policy_fsdp = apply_fsdp2(
                policy,
                plan=assets.parallel_plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
            )

        reference = None
        reference_fsdp: FSDP2Application | None = None
        if reference_checkpoint is not None:
            reference = load_wan_role_adapter(
                checkpoint=reference_checkpoint,
                **adapter_options,
            )
            validate_model_dtype(reference, dtype)
            reference.trainable_module.requires_grad_(False)
            reference.trainable_module.eval()
            if distributed_context is not None:
                assert mesh is not None
                reference_fsdp = apply_fsdp2_frozen_reference(
                    reference,
                    plan=assets.parallel_plan,
                    mesh=mesh,
                    param_dtype=dtype,
                    reduce_dtype=torch_dtype(recipe.runtime.reduce_dtype),
                )

        source = build_wan_rollout_source(
            assets,
            recipe,
            data_plan,
            policy,
            guidance_scale=algorithm.guidance_scale,
            replay_microbatch_size=data_plan.replay_microbatch_size,
        )
        generator = source.generator
        autocast_dtype = None if dtype is torch.float32 else dtype
        policy_prediction = NativeFlowPredictionAdapter(
            policy,
            autocast_dtype=autocast_dtype,
            checkpoint_identity=recipe.model.checkpoint,
        )
        require_checkpoint_identity(
            policy_prediction,
            recipe.model.checkpoint,
            role="Wan flow-policy policy",
        )
        reference_prediction = (
            None
            if reference is None
            else NativeFlowPredictionAdapter(
                reference,
                autocast_dtype=autocast_dtype,
                checkpoint_identity=algorithm.reference_checkpoint,
            )
        )
        if reference_prediction is not None:
            assert algorithm.reference_checkpoint is not None
            require_checkpoint_identity(
                reference_prediction,
                algorithm.reference_checkpoint,
                role="Wan flow-policy reference policy",
            )
            require_independent_modules(
                {
                    "policy": policy_prediction.module,
                    "reference-policy": reference_prediction.module,
                }
            )
        initial_policy_revision = (
            f"{policy_checkpoint.requested_reference}:seed-{assets.base_seed}"
        )
        parallel_context = PostTrainingParallelContext.current()
        stack = build_native_flow_policy_training_stack(
            recipe,
            policy=policy_prediction,
            initial_policy_revision=initial_policy_revision,
            reference_policy=reference_prediction,
            parallel_context=parallel_context,
            fused_adamw=fused_adamw,
            rollout_forward_batch_size=data_plan.rollout_forward_batch_size,
            replay_microbatch_size=data_plan.replay_microbatch_size,
        )
        height, width, frames = assets.generation_geometry
        latent_shape = wan_rollout_latent_shape(
            policy,
            assets.generation_geometry,
        )
        rollout_loader = NativeFlowPolicyDataLoader(
            source.loader,
            group_size=stack.group_size,
            policy_revision=lambda: stack.engine.current_policy_revision,
            latent_shape=latent_shape,
            sigmas=stack.sigmas,
            device=assets.device,
            dtype=torch_dtype(algorithm.trajectory_dtype),
            generator=generator,
            generation_defaults=data_plan.generation,
            group_namespace=f"rank-{assets.rank:08d}",
            shared_negative_conditioning=(None if source.unconditional is None else source.unconditional.tensors),
            init_same_noise=stack.init_same_noise,
        )

        reward_adapter = build_wan_terminal_reward_adapter(
            assets,
            data_plan,
            algorithm.reward_model,
            attention_implementation=videoalign_attention_implementation,
        )
        roles = WanFlowPolicyRoleBundle(
            policy=policy,
            reference=reference,
            policy_checkpoint=policy_checkpoint,
            reference_checkpoint=reference_checkpoint,
            policy_peft=policy_peft,
            policy_fsdp=policy_fsdp,
            reference_fsdp=reference_fsdp,
        )
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
            "rollout_forward_batch_size": data_plan.rollout_forward_batch_size,
            "replay_microbatch_size": data_plan.replay_microbatch_size,
            "group_size": algorithm.group_size,
            "guidance_scale": algorithm.guidance_scale,
            "init_same_noise": algorithm.init_same_noise,
            "unconditional_conditioning": (
                None if source.unconditional is None else source.unconditional.artifact.to_dict()
            ),
            "sde_index_schedule": dict(stack.sde_index_schedule.identity),
            "transition_strategy": dict(stack.transition_strategy.identity),
            "tail_policy": recipe.data.tail_policy,
            "group_namespace_derivation": "rank-{global-rank:08d}",
            "parallel_plan": assets.parallel_plan.to_dict(),
        }
        reward_identity = {
            "adapter": reward_adapter.identity,
            "codec": {
                "checkpoint": assets.resolved_component_checkpoints["vae"].to_dict(),
                "options": dict(data_plan.codec_options),
                "device_type": assets.reward_device.type,
            },
        }
        progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
        identity = {
            "schema": "worldfoundry-wan-flow-policy-resume-identity",
            "recipe": recipe.to_dict(),
            "roles": roles.runtime_identity(),
            "data": data_identity,
            "reward": reward_identity,
            "runtime": recipe.to_dict()["runtime"],
            "distributed": recipe.to_dict()["distributed"],
            "tuning": recipe.to_dict()["tuning"],
            "initialization_seed": assets.base_seed,
            "rank_seed_derivation": "base-seed-plus-global-rank",
        }
        checkpoint_state = TrainingState(
            model=policy.trainable_module,
            optimizer=stack.optimizer,
            engine=stack.engine,
            dataloader=rollout_loader,
            objective_generator=generator,
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
                    "metric_scope": "rank-zero-local",
                    "run_id": recipe.run.id,
                    "recorded_at": utc_now_iso(),
                },
                root=assets.output_dir,
            )

        session = stack.session_type(
            sampler=stack.sampler,
            reward_adapter=reward_adapter,
            scalarizer=stack.scalarizer,
            engine=stack.engine,
            progress=progress,
            sde_index_schedule=stack.sde_index_schedule,
            old_log_prob_source=stack.old_log_prob_source,
            advantage_epsilon=stack.advantage_epsilon,
            advantage_normalization=stack.advantage_normalization,
            advantage_clip_max=stack.advantage_clip_max,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=recipe.checkpoint.save_every_steps,
            asynchronous_checkpoints=recipe.checkpoint.async_save,
            event_sink=event_sink,
            **stack.session_kwargs,
        )
        return WanFlowPolicyTrainingRun(
            recipe=recipe,
            session=session,
            dataloader=rollout_loader,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            roles=roles,
            reward_adapter=reward_adapter,
            output_dir=assets.output_dir,
            data_identity=data_identity,
            reward_identity=reward_identity,
            resume_artifact=resume_artifact,
            distributed_context=distributed_context,
        )
    except Exception:
        if distributed_context is not None:
            distributed_context.close()
        raise


__all__ = [
    "WAN_FLOW_POLICY_RUN_SCHEMA",
    "WanFlowPolicyDataPlan",
    "WanFlowPolicyRoleBundle",
    "WanFlowPolicyRunSummary",
    "WanFlowPolicyTrainingRun",
    "materialize_wan_flow_policy_training_run",
    "validate_wan_flow_policy_recipe",
]
