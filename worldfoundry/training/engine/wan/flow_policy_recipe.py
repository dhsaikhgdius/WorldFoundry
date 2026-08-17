"""Wan flow-policy recipe, rollout-cache, and generation validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from worldfoundry.training.data.rollout_cache import RolloutConditioningDataset
from worldfoundry.training.data.shared_conditioning import SharedConditioningSample
from worldfoundry.training.models.wan import WanTrainAdapter
from worldfoundry.training.recipes.post_training.algorithms.flow_policy import (
    FlowPolicyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.post_training.rollout import LocalRolloutSpec

_FLOW_DATA_OPTIONS = frozenset(
    {
        "generation",
        "multiprocessing_context",
        "num_workers",
        "persistent_workers",
        "pin_memory",
        "prefetch_factor",
        "prompt_batch_size",
        "replay_microbatch_size",
        "rollout_forward_batch_size",
        "snapshot_every_n_steps",
        "vae_chunk_duration",
        "vae_tile_size",
        "vae_tile_stride",
        "vae_tiled",
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


def _strict_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _pair(value: object, *, field_name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must contain two positive integers")
    values = tuple(_positive_int(item, field_name=field_name) for item in value)
    if len(values) != 2:
        raise ValueError(f"{field_name} must contain two positive integers")
    return values


@dataclass(frozen=True, slots=True)
class WanFlowPolicyDataPlan:
    """Validated prompt-loader, generation, and reward-decoder settings."""

    generation: Mapping[str, int]
    prompt_batch_size: int
    rollout_forward_batch_size: int | None
    replay_microbatch_size: int | None
    num_workers: int
    pin_memory: bool | None
    persistent_workers: bool
    prefetch_factor: int | None
    multiprocessing_context: str | None
    snapshot_every_n_steps: int
    codec_options: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", MappingProxyType(dict(self.generation)))
        object.__setattr__(self, "codec_options", MappingProxyType(dict(self.codec_options)))


def build_wan_rollout_data_plan(recipe: PostTrainingRecipe) -> WanFlowPolicyDataPlan:
    """Resolve the shared prompt-loader, generation, and decoder data plane."""

    options = dict(recipe.data.options)
    unknown = sorted(set(options) - _FLOW_DATA_OPTIONS)
    if unknown:
        raise ValueError(f"unknown Wan flow-policy data.options: {unknown}")
    raw_generation = options.get("generation")
    if not isinstance(raw_generation, Mapping):
        raise TypeError("flow-policy data.options.generation must be a mapping")
    generation = {str(key): value for key, value in raw_generation.items()}
    expected_generation = {"height", "width", "num_frames"}
    if set(generation) != expected_generation:
        raise ValueError("flow-policy generation defaults must contain exactly height, width, and num_frames")
    resolved_generation = {
        name: _positive_int(
            generation[name],
            field_name=f"data.options.generation.{name}",
        )
        for name in ("height", "width", "num_frames")
    }
    pin_memory = options.get("pin_memory")
    if pin_memory is not None:
        pin_memory = _strict_bool(pin_memory, field_name="data.options.pin_memory")
    persistent_workers = _strict_bool(
        options.get("persistent_workers", False),
        field_name="data.options.persistent_workers",
    )
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
    codec_options: dict[str, object] = {}
    if "vae_tiled" in options:
        codec_options["tiled"] = _strict_bool(
            options["vae_tiled"],
            field_name="data.options.vae_tiled",
        )
    if "vae_tile_size" in options:
        codec_options["tile_size"] = _pair(
            options["vae_tile_size"],
            field_name="data.options.vae_tile_size",
        )
    if "vae_tile_stride" in options:
        codec_options["tile_stride"] = _pair(
            options["vae_tile_stride"],
            field_name="data.options.vae_tile_stride",
        )
    if "vae_chunk_duration" in options:
        codec_options["chunk_duration"] = _positive_int(
            options["vae_chunk_duration"],
            field_name="data.options.vae_chunk_duration",
        )
    return WanFlowPolicyDataPlan(
        generation=resolved_generation,
        prompt_batch_size=_positive_int(
            options.get("prompt_batch_size", 1),
            field_name="data.options.prompt_batch_size",
        ),
        rollout_forward_batch_size=(
            None
            if "rollout_forward_batch_size" not in options
            else _positive_int(
                options["rollout_forward_batch_size"],
                field_name="data.options.rollout_forward_batch_size",
            )
        ),
        replay_microbatch_size=(
            None
            if "replay_microbatch_size" not in options
            else _positive_int(
                options["replay_microbatch_size"],
                field_name="data.options.replay_microbatch_size",
            )
        ),
        num_workers=_non_negative_int(
            options.get("num_workers", 0),
            field_name="data.options.num_workers",
        ),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
        snapshot_every_n_steps=_positive_int(
            options.get("snapshot_every_n_steps", 1),
            field_name="data.options.snapshot_every_n_steps",
        ),
        codec_options=codec_options,
    )


def validate_wan_flow_policy_recipe(
    recipe: PostTrainingRecipe,
) -> tuple[FlowPolicyAlgorithmSpec, WanFlowPolicyDataPlan]:
    """Validate the complete model-family execution envelope."""

    if not isinstance(recipe, PostTrainingRecipe):
        raise TypeError("recipe must be PostTrainingRecipe")
    if not isinstance(recipe.algorithm, FlowPolicyAlgorithmSpec):
        raise TypeError("Wan flow-policy materialization requires a flow-policy algorithm")
    if not isinstance(recipe.rollout, LocalRolloutSpec):
        raise ValueError("Wan2.1 flow-policy materialization currently requires local rollout")
    if recipe.model.recipe != "wan2.1-t2v-1.3b":
        raise ValueError("native Wan flow-policy currently requires wan2.1-t2v-1.3b")
    if recipe.data.cache is None:
        raise ValueError("Wan flow-policy requires an immutable data.cache")
    if recipe.distributed.backend not in {"single", "fsdp2"}:
        raise ValueError("Wan flow-policy currently supports single or FSDP2 execution")
    if recipe.distributed.cp != 1 or recipe.distributed.tp != 1:
        raise ValueError("Wan flow-policy context/tensor parallelism is not implemented")
    if recipe.runtime.activation_checkpoint not in {"none", "full"}:
        raise ValueError("Wan flow-policy activation_checkpoint must be 'none' or 'full'")
    if recipe.tuning.mode == "partial":
        raise ValueError("partial Wan flow-policy tuning needs an explicit parameter policy")
    if recipe.data.shuffle_seed < 0:
        raise ValueError("Wan flow-policy data.shuffle_seed must be non-negative")
    plan = build_wan_rollout_data_plan(recipe)
    validate_generation_geometry(
        (
            plan.generation["height"],
            plan.generation["width"],
            plan.generation["num_frames"],
        ),
        frame_factor=recipe.algorithm.reward_model.frame_factor,
    )
    return recipe.algorithm, plan


def validate_generation_geometry(
    geometry: tuple[int, int, int],
    *,
    frame_factor: int,
) -> None:
    height, width, frames = geometry
    if height % 16 or width % 16:
        raise ValueError("Wan flow-policy height and width must be divisible by 16 for VAE and DiT patch geometry")
    if frames < max(5, frame_factor) or (frames - 1) % 4:
        raise ValueError("Wan flow-policy num_frames must be at least five and satisfy 1 + 4k")


def audit_conditioning_cache(
    dataset: RolloutConditioningDataset,
    adapter: WanTrainAdapter,
    *,
    model_recipe: str,
    conditioner: Mapping[str, object],
    tokenizer: Mapping[str, object],
) -> None:
    index = dataset.index
    if index.model_recipe != model_recipe:
        raise ValueError("rollout conditioning cache belongs to another model recipe")
    if index.conditioner != conditioner:
        raise ValueError("rollout conditioning cache uses another conditioner checkpoint")
    if index.tokenizer != tokenizer:
        raise ValueError("rollout conditioning cache uses another tokenizer checkpoint")
    expected_shape = (adapter.expected_text_length, adapter.expected_context_features)
    for entry in index.entries:
        descriptors = entry.artifact.identity.tensors
        if set(descriptors) != {"context"}:
            raise ValueError("Wan rollout conditioning must contain only context")
        descriptor = descriptors["context"]
        if descriptor.shape != expected_shape or descriptor.layout != "sequence-features":
            raise ValueError("Wan rollout context tensor contract is incompatible")


def audit_unconditional_conditioning(
    sample: SharedConditioningSample,
    adapter: WanTrainAdapter,
    *,
    model_recipe: str,
    conditioner: Mapping[str, object],
    tokenizer: Mapping[str, object],
) -> None:
    if not isinstance(sample, SharedConditioningSample):
        raise TypeError("unconditional conditioning must be SharedConditioningSample")
    identity = sample.artifact.identity
    if identity.branch != "unconditional" or identity.prompt != "":
        raise ValueError("Wan CFG cache must be the empty-prompt conditioning branch")
    if (
        identity.model_recipe != model_recipe
        or identity.conditioner != conditioner
        or identity.tokenizer != tokenizer
    ):
        raise ValueError("Wan CFG cache identity differs from the rollout conditioner")
    expected_shape = (adapter.expected_text_length, adapter.expected_context_features)
    if set(identity.tensors) != {"context"} or set(sample.tensors) != {"context"}:
        raise ValueError("Wan CFG cache must contain only context")
    descriptor = identity.tensors["context"]
    if descriptor.shape != expected_shape or descriptor.layout != "sequence-features":
        raise ValueError("Wan CFG empty-prompt context has an incompatible tensor contract")


def audit_component_overrides(
    values: Mapping[str, object] | None,
) -> dict[str, object]:
    """Validate optional local or Hub component checkpoints."""

    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec

    overrides = {str(key): value for key, value in dict(values or {}).items()}
    allowed = {"text-encoder", "tokenizer", "vae"}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown Wan flow-policy component overrides: {unknown}")
    for name, value in overrides.items():
        if not isinstance(value, CheckpointSpec):
            raise TypeError(f"component override {name!r} must be CheckpointSpec")
        if value.sources and not value.files:
            raise ValueError(f"local component override {name!r} must declare loaded files")
        if not value.sources and (not value.repo_id or not value.revision):
            raise ValueError(f"Hub component override {name!r} must declare a repository and revision")
    return overrides


__all__ = [
    "WanFlowPolicyDataPlan",
    "audit_component_overrides",
    "audit_conditioning_cache",
    "audit_unconditional_conditioning",
    "build_wan_rollout_data_plan",
    "validate_generation_geometry",
    "validate_wan_flow_policy_recipe",
]
