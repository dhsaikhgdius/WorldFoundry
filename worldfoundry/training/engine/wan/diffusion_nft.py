"""Native Wan DiffusionNFT role materialization and run lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.core.io.integrity import append_jsonl_durable, canonical_sha256
from worldfoundry.core.time import utc_now_iso
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.data.wan.contracts import wan_checkpoint_asset_digest
from worldfoundry.training.post_training.rl.algorithms.diffusion_nft.builder import (
    build_native_diffusion_nft_training_stack,
)
from worldfoundry.training.post_training.rl.batching import (
    NativeFlowPolicyDataLoader,
)
from worldfoundry.training.post_training.shared.prediction import (
    NativeFlowPredictionAdapter,
)
from worldfoundry.training.post_training.shared.role_checkpoints import (
    resolve_role_checkpoint,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from .diffusion_nft_recipe import validate_wan_diffusion_nft_recipe
from .diffusion_nft_run import (
    WAN_DIFFUSION_NFT_RUN_SCHEMA,
    WanDiffusionNFTRoleBundle,
    WanDiffusionNFTRunSummary,
    WanDiffusionNFTTrainingRun,
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


def materialize_wan_diffusion_nft_training_run(
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
) -> WanDiffusionNFTTrainingRun:
    """Materialize a WorldFoundry-owned Wan DiffusionNFT training run."""

    algorithm, data_plan = validate_wan_diffusion_nft_recipe(recipe)
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
    try:
        from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec

        role_overrides = dict(audited_role_overrides or {})
        unknown_role_overrides = sorted(set(role_overrides) - {"policy", "reference"})
        if unknown_role_overrides:
            raise ValueError(f"unknown Wan DiffusionNFT role overrides: {unknown_role_overrides}")
        if algorithm.reference_mse_weight == 0 and "reference" in role_overrides:
            raise ValueError("reference role override is unused when reference MSE is disabled")
        for name, value in role_overrides.items():
            if not isinstance(value, CheckpointSpec):
                raise TypeError(f"audited role override {name!r} must be CheckpointSpec")
        default_dit = assets.native_recipe.checkpoints["dit"]
        policy_checkpoint = resolve_role_checkpoint(
            role="policy",
            reference=recipe.model.checkpoint,
            native_default=default_dit,
            audited_local_override=role_overrides.get("policy"),
        )
        old_policy_checkpoint = policy_checkpoint
        reference_checkpoint = None
        if algorithm.reference_mse_weight > 0:
            assert algorithm.reference_checkpoint is not None
            reference_checkpoint = resolve_role_checkpoint(
                role="reference",
                reference=algorithm.reference_checkpoint,
                native_default=default_dit,
                audited_local_override=role_overrides.get("reference"),
            )

        adapter_options = {
            "assembler": assets.assembler,
            "native_recipe": assets.native_recipe,
            "device": assets.device,
            "dtype": assets.dtype,
            "num_train_timesteps": algorithm.num_train_timesteps,
            "gradient_checkpointing": recipe.runtime.activation_checkpoint == "full",
            "force_torch_attention": force_torch_attention,
        }
        policy = load_wan_role_adapter(
            checkpoint=policy_checkpoint,
            **adapter_options,
        )
        validate_model_dtype(policy, assets.dtype)
        policy_peft = apply_wan_tuning(recipe, policy)

        old_policy = load_wan_role_adapter(
            checkpoint=old_policy_checkpoint,
            **adapter_options,
        )
        validate_model_dtype(old_policy, assets.dtype)
        old_policy_peft = apply_wan_tuning(recipe, old_policy)
        old_policy.trainable_module.requires_grad_(False)
        old_policy.trainable_module.eval()

        reference = None
        if reference_checkpoint is not None:
            reference = load_wan_role_adapter(
                checkpoint=reference_checkpoint,
                **adapter_options,
            )
            validate_model_dtype(reference, assets.dtype)
            reference.trainable_module.requires_grad_(False)
            reference.trainable_module.eval()

        source = build_wan_rollout_source(
            assets,
            recipe,
            data_plan,
            policy,
            guidance_scale=algorithm.collection.guidance_scale,
        )
        reward_adapter = build_wan_terminal_reward_adapter(
            assets,
            data_plan,
            algorithm.reward_model,
            attention_implementation=videoalign_attention_implementation,
        )
        autocast_dtype = None if assets.dtype is torch.float32 else assets.dtype
        policy_prediction = NativeFlowPredictionAdapter(
            policy,
            autocast_dtype=autocast_dtype,
        )
        old_policy_prediction = NativeFlowPredictionAdapter(
            old_policy,
            autocast_dtype=autocast_dtype,
        )
        reference_prediction = (
            None
            if reference is None
            else NativeFlowPredictionAdapter(
                reference,
                autocast_dtype=autocast_dtype,
            )
        )
        initial_old_policy_revision = canonical_sha256(
            {
                "schema": "worldfoundry-initial-diffusion-nft-old-policy-revision",
                "checkpoint_digest": policy_checkpoint.digest,
                "tuning": recipe.to_dict()["tuning"],
                "initialization_seed": assets.base_seed,
            }
        )
        stack = build_native_diffusion_nft_training_stack(
            recipe,
            policy=policy_prediction,
            old_policy=old_policy_prediction,
            initial_old_policy_revision=initial_old_policy_revision,
            reward_adapter=reward_adapter,
            reference_policy=reference_prediction,
            fused_adamw=fused_adamw,
        )
        latent_shape = wan_rollout_latent_shape(
            policy,
            assets.generation_geometry,
        )
        rollout_loader = NativeFlowPolicyDataLoader(
            source.loader,
            group_size=stack.group_size,
            policy_revision=lambda: stack.engine.current_collection_policy_revision,
            latent_shape=latent_shape,
            sigmas=stack.sigmas,
            device=assets.device,
            dtype=torch_dtype(algorithm.collection.latent_dtype),
            generator=source.generator,
            generation_defaults=data_plan.generation,
            group_namespace=f"rank-{assets.rank:08d}",
            shared_negative_conditioning=(None if source.unconditional is None else source.unconditional.tensors),
            init_same_noise=False,
        )
        roles = WanDiffusionNFTRoleBundle(
            policy=policy,
            old_policy=old_policy,
            reference=reference,
            policy_checkpoint=policy_checkpoint,
            old_policy_checkpoint=old_policy_checkpoint,
            reference_checkpoint=reference_checkpoint,
            policy_peft=policy_peft,
            old_policy_peft=old_policy_peft,
        )
        height, width, frames = assets.generation_geometry
        data_identity = {
            "prompt_manifest_sha256": assets.prompts.manifest_sha256,
            "prompt_dataset_digest": assets.prompts.dataset_digest,
            "conditioned_dataset_digest": assets.conditioning.dataset_digest,
            "conditioning_index_sha256": assets.conditioning.index.digest,
            "model_recipe_digest": assets.model_contract_digest,
            "conditioner_digest": assets.conditioner_digest,
            "tokenizer_digest": assets.tokenizer_digest,
            "sample_count": len(assets.conditioning),
            "generation": {
                "height": height,
                "width": width,
                "num_frames": frames,
            },
            "latent_shape": list(latent_shape),
            "prompt_batch_size": data_plan.prompt_batch_size,
            "group_size": algorithm.collection.group_size,
            "guidance_scale": algorithm.collection.guidance_scale,
            "sigmas": list(algorithm.collection.sigmas),
            "tail_policy": recipe.data.tail_policy,
        }
        reward_identity = {
            "adapter": reward_adapter.identity,
            "adapter_digest": reward_adapter.digest,
            "codec": {
                "checkpoint_digest": wan_checkpoint_asset_digest(assets.resolved_component_checkpoints["vae"]),
                "options": dict(data_plan.codec_options),
                "device_type": assets.reward_device.type,
            },
        }
        progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
        identity = {
            "schema": "worldfoundry-wan-diffusion-nft-resume-identity",
            "recipe_digest": recipe.digest,
            "roles": roles.runtime_identity(),
            "data": data_identity,
            "reward": reward_identity,
            "runtime": recipe.to_dict()["runtime"],
            "distributed": recipe.to_dict()["distributed"],
            "tuning": recipe.to_dict()["tuning"],
            "initialization_seed": assets.base_seed,
        }
        checkpoint_state = TrainingState(
            model=policy.trainable_module,
            optimizer=stack.optimizer,
            engine=stack.engine,
            dataloader=rollout_loader,
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
            append_jsonl_durable(
                assets.output_dir / "metrics.jsonl",
                {
                    **dict(event),
                    "run_id": recipe.run.id,
                    "recipe_digest": recipe.digest,
                    "recorded_at": utc_now_iso(),
                },
                root=assets.output_dir,
            )

        session = stack.build_session(
            rollout_loader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=recipe.checkpoint.save_every_steps,
            asynchronous_checkpoints=recipe.checkpoint.async_save,
            event_sink=event_sink,
        )
        return WanDiffusionNFTTrainingRun(
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
            distributed_context=None,
        )
    except Exception:
        if assets.distributed_context is not None:
            assets.distributed_context.close()
        raise


__all__ = [
    "WAN_DIFFUSION_NFT_RUN_SCHEMA",
    "WanDiffusionNFTRoleBundle",
    "WanDiffusionNFTRunSummary",
    "WanDiffusionNFTTrainingRun",
    "materialize_wan_diffusion_nft_training_run",
    "validate_wan_diffusion_nft_recipe",
]
