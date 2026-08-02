"""Wan DiffusionNFT execution-envelope validation."""

from __future__ import annotations

from worldfoundry.training.recipes.post_training.algorithms.diffusion_nft import (
    DiffusionNFTAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from .flow_policy_recipe import (
    WanFlowPolicyDataPlan,
    build_wan_rollout_data_plan,
    validate_generation_geometry,
)


def validate_wan_diffusion_nft_recipe(
    recipe: PostTrainingRecipe,
) -> tuple[DiffusionNFTAlgorithmSpec, WanFlowPolicyDataPlan]:
    """Validate the complete currently executable Wan DiffusionNFT run."""

    if not isinstance(recipe, PostTrainingRecipe):
        raise TypeError("recipe must be PostTrainingRecipe")
    if not isinstance(recipe.algorithm, DiffusionNFTAlgorithmSpec):
        raise TypeError("Wan DiffusionNFT materialization requires diffusion-nft")
    if recipe.model.recipe != "wan2.1-t2v-1.3b":
        raise ValueError("native Wan DiffusionNFT currently requires wan2.1-t2v-1.3b")
    if recipe.data.cache is None:
        raise ValueError("Wan DiffusionNFT requires an immutable data.cache")
    if recipe.distributed.backend != "single":
        raise ValueError(
            "Wan DiffusionNFT currently requires distributed.backend='single'; "
            "old-policy refresh and old-policy checkpoint state do not yet use "
            "sharding-aware FSDP2 primitives"
        )
    if recipe.distributed.cp != 1 or recipe.distributed.tp != 1:
        raise ValueError("Wan DiffusionNFT context/tensor parallelism is not implemented")
    if recipe.runtime.activation_checkpoint not in {"none", "full"}:
        raise ValueError("Wan DiffusionNFT activation_checkpoint must be 'none' or 'full'")
    if recipe.tuning.mode == "partial":
        raise ValueError("partial Wan DiffusionNFT tuning needs an explicit parameter policy")
    if recipe.data.shuffle_seed < 0:
        raise ValueError("Wan DiffusionNFT data.shuffle_seed must be non-negative")
    plan = build_wan_rollout_data_plan(recipe)
    if plan.replay_microbatch_size is not None:
        raise ValueError("Wan DiffusionNFT does not replay trajectories or use replay_microbatch_size")
    if plan.rollout_forward_batch_size is not None:
        raise ValueError("Wan DiffusionNFT collection batching belongs in algorithm.collection.forward_batch_size")
    validate_generation_geometry(
        (
            plan.generation["height"],
            plan.generation["width"],
            plan.generation["num_frames"],
        ),
        frame_factor=recipe.algorithm.reward_model.frame_factor,
    )
    return recipe.algorithm, plan


__all__ = ["validate_wan_diffusion_nft_recipe"]
