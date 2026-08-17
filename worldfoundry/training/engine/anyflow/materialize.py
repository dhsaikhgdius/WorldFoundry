"""Materialize executable single-GPU or DDP AnyFlow training runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.shared_conditioning import SharedConditioningStore
from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.distributed.parallel import (
    DistributedTrainingContext,
    ParallelPlan,
)
from worldfoundry.training.post_training.distillation.anyflow.builder import (
    build_native_anyflow_on_policy_training_stack,
    build_native_anyflow_pretraining_stack,
)
from worldfoundry.training.post_training.distillation.anyflow.session import (
    NativeAnyFlowOnPolicyTrainingSession,
    NativeAnyFlowPretrainingSession,
)
from worldfoundry.training.post_training.shared.building import resolve_tensor_dtype
from worldfoundry.training.post_training.shared.distributed import (
    PostTrainingParallelContext,
)
from worldfoundry.training.recipes.post_training.algorithms.anyflow import (
    AnyFlowBidirectionalOnPolicyAlgorithmSpec,
    AnyFlowBidirectionalPretrainAlgorithmSpec,
    AnyFlowFAROnPolicyAlgorithmSpec,
    AnyFlowFARPretrainAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from ..artifacts import create_run_directory
from ..wan.cache import build_wan_cache_loader
from .data import AnyFlowCachedDataLoader
from .roles import (
    AnyFlowAlgorithm,
    far_partition,
    materialize_anyflow_roles,
)
from .run import AnyFlowTrainingRun

_ANYFLOW_ALGORITHMS = (
    AnyFlowFARPretrainAlgorithmSpec,
    AnyFlowBidirectionalPretrainAlgorithmSpec,
    AnyFlowFAROnPolicyAlgorithmSpec,
    AnyFlowBidirectionalOnPolicyAlgorithmSpec,
)


def _validate_recipe(recipe: PostTrainingRecipe) -> AnyFlowAlgorithm:
    algorithm = recipe.algorithm
    if not isinstance(algorithm, _ANYFLOW_ALGORITHMS):
        raise TypeError("AnyFlow materialization requires an AnyFlow algorithm")
    if recipe.model.recipe != "wan2.1-t2v-1.3b":
        raise ValueError("native AnyFlow currently requires wan2.1-t2v-1.3b data")
    if recipe.tuning.mode != "full":
        raise ValueError("native AnyFlow production runs currently support full tuning")
    if recipe.data.cache is None or recipe.data.max_latent_tokens_per_microbatch is None:
        raise ValueError("AnyFlow requires a cached latent dataset and token budget")
    if recipe.data.tail_policy not in {"drop", "pad"}:
        raise ValueError("AnyFlow cached loading supports tail_policy drop or pad")
    if recipe.distributed.backend not in {"single", "ddp"}:
        raise ValueError("AnyFlow production runs currently support single or DDP")
    if recipe.distributed.cp != 1 or recipe.distributed.tp != 1:
        raise ValueError("AnyFlow DDP currently uses data parallelism only")
    if recipe.runtime.activation_checkpoint not in {"none", "full"}:
        raise ValueError("AnyFlow activation_checkpoint must be none or full")
    if recipe.runtime.compile:
        raise ValueError("AnyFlow compile is not enabled in the production materializer")
    return algorithm


def _validate_cache(
    recipe: PostTrainingRecipe,
    algorithm: AnyFlowAlgorithm,
    cache: VideoCachedDataset,
) -> None:
    partition = far_partition(algorithm)
    patch = (1, 2, 2) if partition is None else partition.patch_size
    for entry in cache.index.entries:
        latents = entry.tensors["clean_latents"]
        context = entry.tensors.get("condition.context")
        if entry.provenance.model_recipe != recipe.model.recipe:
            raise ValueError("AnyFlow cache was prepared for a different model recipe")
        if latents.shape[0] != 16 or any(
            size % divisor for size, divisor in zip(latents.shape[1:], patch)
        ):
            raise ValueError("AnyFlow cache latent geometry is incompatible")
        if partition is not None and latents.shape[1] != partition.frame_count:
            raise ValueError("AnyFlow FAR cache must contain one complete chunk partition")
        if context is None or context.shape[-1] != 4096:
            raise ValueError("AnyFlow cache requires Wan UMT5 context")


def materialize_anyflow_training_run(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    checkpoint_overrides: Mapping[str, CheckpointSpec] | None = None,
    audit_cache_on_open: bool = True,
    verify_cache_on_read: bool = True,
    force_torch_attention: bool = True,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> AnyFlowTrainingRun:
    """Build a complete native AnyFlow data/model/optimizer/session lifecycle."""

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
    distributed_context: DistributedTrainingContext | None = None
    if recipe.distributed.backend == "ddp":
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device
    elif resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    try:
        world_size = 1 if distributed_context is None else distributed_context.world_size
        rank = 0 if distributed_context is None else distributed_context.rank
        plan = ParallelPlan.resolve(recipe.distributed, world_size=world_size)
        create_run_directory(destination, distributed_context)

        manifest = TrainingManifestDataset.from_file(
            manifest_path,
            split=recipe.data.split,
            verify_files=False,
        )
        cache = VideoCachedDataset(
            cache_path,
            expected_sample_ids=manifest.sample_ids,
            audit_on_open=audit_cache_on_open,
            verify_on_read=verify_cache_on_read,
        )
        _validate_cache(recipe, algorithm, cache)
        unconditional = SharedConditioningStore(cache_path).read("unconditional")
        if unconditional.artifact.identity.model_recipe != recipe.model.recipe:
            raise ValueError("AnyFlow unconditional context belongs to another model recipe")

        dtype = resolve_tensor_dtype(recipe.runtime.param_dtype)
        resolved_seed = (
            recipe.data.shuffle_seed
            if initialization_seed is None
            else int(initialization_seed)
        )
        rank_seed = resolved_seed + rank
        torch.manual_seed(rank_seed)
        if resolved_device.type == "cuda":
            torch.cuda.manual_seed_all(rank_seed)
        roles = materialize_anyflow_roles(
            recipe,
            algorithm=algorithm,
            device=resolved_device,
            distributed_context=distributed_context,
            checkpoint_overrides=checkpoint_overrides,
            force_torch_attention=force_torch_attention,
        )
        source_loader, token_sampler = build_wan_cache_loader(
            recipe=recipe,
            dataset=cache,
            rank=rank,
            world_size=world_size,
            default_pin_memory=resolved_device.type == "cuda",
        )
        loader = AnyFlowCachedDataLoader(
            source_loader,
            unconditional_conditioning=unconditional.tensors,
            device=resolved_device,
            dtype=dtype,
        )
        parallel_context = PostTrainingParallelContext.current()
        if isinstance(
            algorithm,
            (AnyFlowFARPretrainAlgorithmSpec, AnyFlowBidirectionalPretrainAlgorithmSpec),
        ):
            stack = build_native_anyflow_pretraining_stack(
                recipe,
                student=roles.student,
                parallel_context=parallel_context,
                fused_adamw=fused_adamw,
            )
            optimizers = stack.optimizer
        else:
            if roles.real_score is None or roles.fake_score is None:
                raise RuntimeError("AnyFlow on-policy roles were not materialized")
            stack = build_native_anyflow_on_policy_training_stack(
                recipe,
                student=roles.student,
                real_score=roles.real_score,
                fake_score=roles.fake_score,
                parallel_context=parallel_context,
                fused_adamw=fused_adamw,
            )
            optimizers = (stack.student_optimizer, stack.fake_score_optimizer)

        progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
        generator = torch.Generator(device=resolved_device).manual_seed(rank_seed)
        checkpoint_state = TrainingState(
            model=roles.trainable_model(),
            optimizer=optimizers,
            engine=stack.engine,
            dataloader=loader,
            objective_generator=generator,
            progress=progress,
            identity={
                "recipe": recipe.to_dict(),
                "roles": roles.checkpoint_identity(),
                "data": {
                    "sample_ids": list(cache.sample_ids),
                    "unconditional": unconditional.artifact.identity.to_dict(),
                    "latent_token_budget": token_sampler.max_latent_tokens,
                },
                "parallel": plan.to_dict(),
                "initialization_seed": resolved_seed,
            },
            **stack.checkpoint_state_kwargs(),
        )
        checkpointer = TrainingCheckpointer(destination / "checkpoints")
        resume_artifact = None
        if resume_checkpoint is not None:
            resume_artifact = checkpointer.load(checkpoint_state, resume_checkpoint)

        session_options = {
            "checkpoint_state": checkpoint_state,
            "checkpointer": checkpointer,
            "save_every_steps": recipe.checkpoint.save_every_steps,
            "asynchronous_checkpoints": recipe.checkpoint.async_save,
        }
        if isinstance(
            algorithm,
            (AnyFlowFARPretrainAlgorithmSpec, AnyFlowBidirectionalPretrainAlgorithmSpec),
        ):
            session = NativeAnyFlowPretrainingSession(
                stack.engine,
                loader,
                progress,
                **session_options,
            )
        else:
            session = NativeAnyFlowOnPolicyTrainingSession(
                stack.engine,
                loader,
                progress,
                **session_options,
            )
        return AnyFlowTrainingRun(
            recipe=recipe,
            session=session,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            roles=roles,
            student_ema=stack.ema,
            output_dir=destination,
            resume_artifact=resume_artifact,
            distributed_context=distributed_context,
        )
    except Exception:
        if distributed_context is not None:
            distributed_context.close()
        raise


__all__ = ["materialize_anyflow_training_run"]
