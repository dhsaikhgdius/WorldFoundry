"""Wan execution validation for native Self-Forcing distillation."""

from __future__ import annotations

from collections.abc import Mapping

from worldfoundry.training.recipes.post_training.algorithms.self_forcing import (
    SelfForcingAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from .flow_policy_recipe import WanFlowPolicyDataPlan

_SELF_FORCING_DATA_OPTIONS = frozenset(
    {
        "generation",
        "multiprocessing_context",
        "num_workers",
        "persistent_workers",
        "pin_memory",
        "prefetch_factor",
        "prompt_batch_size",
        "snapshot_every_n_steps",
    }
)


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return resolved


def _optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def build_wan_self_forcing_data_plan(
    recipe: PostTrainingRecipe,
) -> WanFlowPolicyDataPlan:
    """Resolve only the prompt-loader options consumed by Self-Forcing."""

    options = dict(recipe.data.options)
    unknown = sorted(set(options) - _SELF_FORCING_DATA_OPTIONS)
    if unknown:
        raise ValueError(f"unknown Wan Self-Forcing data.options: {unknown}")
    generation = options.get("generation")
    if not isinstance(generation, Mapping):
        raise TypeError("Self-Forcing data.options.generation must be a mapping")
    if set(generation) != {"height", "width", "num_frames"}:
        raise ValueError("Self-Forcing generation must contain exactly height, width, and num_frames")
    resolved_generation = {
        name: _positive_int(
            generation[name],
            field_name=f"data.options.generation.{name}",
        )
        for name in ("height", "width", "num_frames")
    }
    height = resolved_generation["height"]
    width = resolved_generation["width"]
    frames = resolved_generation["num_frames"]
    if height % 16 or width % 16:
        raise ValueError("Wan Self-Forcing height and width must be divisible by 16")
    if frames < 5 or (frames - 1) % 4:
        raise ValueError("Wan Self-Forcing num_frames must be at least five and satisfy 1 + 4k")

    persistent_workers = options.get("persistent_workers", False)
    if not isinstance(persistent_workers, bool):
        raise TypeError("data.options.persistent_workers must be a bool")
    prefetch_factor = options.get("prefetch_factor")
    if prefetch_factor is not None:
        prefetch_factor = _positive_int(
            prefetch_factor,
            field_name="data.options.prefetch_factor",
        )
    multiprocessing_context = options.get("multiprocessing_context")
    if multiprocessing_context is not None:
        multiprocessing_context = str(multiprocessing_context).strip()
        if not multiprocessing_context:
            raise ValueError("data.options.multiprocessing_context cannot be empty")
    return WanFlowPolicyDataPlan(
        generation=resolved_generation,
        prompt_batch_size=_positive_int(
            options.get("prompt_batch_size", 1),
            field_name="data.options.prompt_batch_size",
        ),
        rollout_forward_batch_size=None,
        replay_microbatch_size=None,
        num_workers=_non_negative_int(
            options.get("num_workers", 0),
            field_name="data.options.num_workers",
        ),
        pin_memory=_optional_bool(
            options.get("pin_memory"),
            field_name="data.options.pin_memory",
        ),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
        snapshot_every_n_steps=_positive_int(
            options.get("snapshot_every_n_steps", 1),
            field_name="data.options.snapshot_every_n_steps",
        ),
        codec_options={},
    )


def validate_wan_self_forcing_recipe(
    recipe: PostTrainingRecipe,
) -> tuple[SelfForcingAlgorithmSpec, WanFlowPolicyDataPlan]:
    """Validate the complete scalable Wan Self-Forcing execution envelope."""

    if not isinstance(recipe, PostTrainingRecipe):
        raise TypeError("recipe must be PostTrainingRecipe")
    if not isinstance(recipe.algorithm, SelfForcingAlgorithmSpec):
        raise TypeError("Wan Self-Forcing materialization requires algorithm.type='self-forcing'")
    if recipe.model.recipe != "wan2.1-t2v-1.3b":
        raise ValueError("native Wan Self-Forcing requires a causal Wan2.1 T2V 1.3B student")
    if recipe.data.cache is None:
        raise ValueError("Wan Self-Forcing requires an immutable prompt-conditioning data.cache")
    if recipe.data.max_latent_tokens_per_microbatch is not None:
        raise ValueError(
            "Wan Self-Forcing uses prompt_batch_size and fixed rollout geometry; "
            "max_latent_tokens_per_microbatch is not consumed"
        )
    if recipe.distributed.backend not in {"single", "fsdp2"}:
        raise ValueError("Wan Self-Forcing supports single-device or FSDP2 execution")
    if recipe.distributed.cp != 1 or recipe.distributed.tp != 1:
        raise ValueError("Wan Self-Forcing context/tensor parallelism is not implemented")
    if recipe.runtime.activation_checkpoint != "full":
        raise ValueError(
            "Wan Self-Forcing requires activation_checkpoint='full': the official causal graph "
            "recomputes selected-step activations after the live KV cache is clean-committed"
        )
    if recipe.tuning.mode == "partial":
        raise ValueError("partial Wan Self-Forcing tuning needs an explicit parameter policy")
    if recipe.data.shuffle_seed < 0:
        raise ValueError("Wan Self-Forcing data.shuffle_seed must be non-negative")
    plan = build_wan_self_forcing_data_plan(recipe)
    latent_frames = 1 + (plan.generation["num_frames"] - 1) // 4
    if latent_frames % recipe.algorithm.frames_per_block:
        raise ValueError("Wan Self-Forcing latent frame count must be divisible by algorithm.frames_per_block")
    return recipe.algorithm, plan


__all__ = [
    "build_wan_self_forcing_data_plan",
    "validate_wan_self_forcing_recipe",
]
