"""Public model identities declared by the Echo-Memory release metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from ..networks.echo_memory.schema import EchoMemoryRecipe, get_echo_memory_recipe

ECHO_SOURCE_REPOSITORY = "https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Memory"
ECHO_SOURCE_REVISION = "30eaffb55b264e1d8dfef70a8934d34c86e29947"
ECHO_CHECKPOINT_REPOSITORY = "Echo-Team/Echo-Memory"
ECHO_CHECKPOINT_REVISION = "7645f33efbfd02c58dc471570d0c7bd6a50eec83"
ECHO_BACKBONE_REPOSITORY = "Wan-AI/Wan2.1-T2V-1.3B"


class EchoCheckpointAvailability(str, Enum):
    """Current upstream checkpoint state, kept separate from model identity."""

    PUBLIC_CURRENT = "public_current"
    UNRELEASED_BY_UPSTREAM = "unreleased_by_upstream"
    RETRACTED_MISALIGNED_BY_UPSTREAM = "retracted_misaligned_by_upstream"


@dataclass(frozen=True)
class EchoMemoryModelSpec:
    """One independently selectable WorldFoundry model."""

    model_id: str
    display_name: str
    recipe_id: str
    checkpoint_file: str
    aliases: tuple[str, ...] = ()
    release_tier: str = "paper_primary"
    checkpoint_availability: EchoCheckpointAvailability = EchoCheckpointAvailability.PUBLIC_CURRENT

    @property
    def recipe(self) -> EchoMemoryRecipe:
        """Return the immutable recipe pinned by this model."""

        return get_echo_memory_recipe(self.recipe_id)

    @property
    def has_public_checkpoint(self) -> bool:
        """Whether the official Hugging Face ``main`` branch serves this file."""

        return self.checkpoint_availability is EchoCheckpointAvailability.PUBLIC_CURRENT

    @property
    def integration_status(self) -> str:
        """Catalog status implied by code readiness and upstream weight access."""

        return "integrated" if self.has_public_checkpoint else "blocked"


_SPECS = (
    EchoMemoryModelSpec(
        model_id="echo-memory-context-k1",
        display_name="Echo-Memory Context K=1",
        recipe_id="context-k1",
        checkpoint_file="context_k1/epoch-0.safetensors",
        aliases=("echo-context-k1",),
    ),
    EchoMemoryModelSpec(
        model_id="echo-memory-context-k20",
        display_name="Echo-Memory Context K=20",
        recipe_id="context-k20",
        checkpoint_file="context_k20/epoch-0.safetensors",
        aliases=("echo-context-k20",),
        checkpoint_availability=EchoCheckpointAvailability.UNRELEASED_BY_UPSTREAM,
    ),
    EchoMemoryModelSpec(
        model_id="echo-memory-spatial",
        display_name="Echo-Memory Spatial Memory",
        recipe_id="spatial-grid-64",
        checkpoint_file="spatial_mem/epoch-0.safetensors",
        aliases=("echo-spatial-memory",),
        checkpoint_availability=EchoCheckpointAvailability.RETRACTED_MISALIGNED_BY_UPSTREAM,
    ),
    EchoMemoryModelSpec(
        model_id="echo-memory-block-ssm",
        display_name="Echo-Memory Block-wise SSM",
        recipe_id="block-ssm-k5",
        checkpoint_file="block_wise_ssm_two_chunk/epoch-0.safetensors",
        aliases=("echo-block-ssm",),
        checkpoint_availability=EchoCheckpointAvailability.UNRELEASED_BY_UPSTREAM,
    ),
    EchoMemoryModelSpec(
        model_id="echo-memory-videossm-hybrid",
        display_name="Echo-Memory VideoSSM Hybrid",
        recipe_id="videossm-hybrid-k5",
        checkpoint_file="videossm_hybrid/epoch-0.safetensors",
        aliases=("echo-videossm",),
        checkpoint_availability=EchoCheckpointAvailability.UNRELEASED_BY_UPSTREAM,
    ),
    EchoMemoryModelSpec(
        model_id="echo-memory-spatial-concat-text",
        display_name="Echo-Memory Spatial Concat-Text Ablation",
        recipe_id="spatial-concat-text-64",
        checkpoint_file="spatial_concat_text_two_chunk/epoch-0.safetensors",
        release_tier="research_ablation",
        checkpoint_availability=EchoCheckpointAvailability.RETRACTED_MISALIGNED_BY_UPSTREAM,
    ),
    EchoMemoryModelSpec(
        model_id="echo-memory-spatial-no-injection",
        display_name="Echo-Memory Spatial No-Injection Ablation",
        recipe_id="spatial-no-injection-64",
        checkpoint_file="spatial_inject_none_two_chunk/epoch-0.safetensors",
        release_tier="research_ablation",
        checkpoint_availability=EchoCheckpointAvailability.RETRACTED_MISALIGNED_BY_UPSTREAM,
    ),
    EchoMemoryModelSpec(
        model_id="echo-memory-spatial-cross-attn-t32",
        display_name="Echo-Memory Spatial Cross-Attention T32 Ablation",
        recipe_id="spatial-cross-attn-32",
        checkpoint_file="spatial_cross_attn_readout_t32_g4_two_chunk/epoch-0.safetensors",
        release_tier="research_ablation",
        checkpoint_availability=EchoCheckpointAvailability.RETRACTED_MISALIGNED_BY_UPSTREAM,
    ),
    EchoMemoryModelSpec(
        model_id="echo-memory-ssm-ctx1-every4-hint21",
        display_name="Echo-Memory SSM Ctx1 Every4 Hint21",
        recipe_id="block-ssm-k1-every4-hint21",
        checkpoint_file="ssm_ablation_ctx1_every4_hint21/epoch-0.safetensors",
        release_tier="research_ablation",
        checkpoint_availability=EchoCheckpointAvailability.UNRELEASED_BY_UPSTREAM,
    ),
    EchoMemoryModelSpec(
        model_id="echo-memory-ssm-ctx5-every1-hint21",
        display_name="Echo-Memory SSM Ctx5 Every1 Hint21",
        recipe_id="block-ssm-k5-every1-hint21",
        checkpoint_file="ssm_ablation_ctx5_every1_hint21/epoch-0.safetensors",
        release_tier="research_ablation",
        checkpoint_availability=EchoCheckpointAvailability.UNRELEASED_BY_UPSTREAM,
    ),
    EchoMemoryModelSpec(
        model_id="echo-memory-ssm-ctx5-every4-hint81",
        display_name="Echo-Memory SSM Ctx5 Every4 Hint81",
        recipe_id="block-ssm-k5-every4-hint81",
        checkpoint_file="ssm_ablation_ctx5_every4_hint81/epoch-0.safetensors",
        release_tier="research_ablation",
        checkpoint_availability=EchoCheckpointAvailability.UNRELEASED_BY_UPSTREAM,
    ),
)

ECHO_MEMORY_MODELS: Mapping[str, EchoMemoryModelSpec] = MappingProxyType({spec.model_id: spec for spec in _SPECS})


def get_echo_memory_model_spec(model_id: str) -> EchoMemoryModelSpec:
    """Resolve an independent Echo model ID or a declared alias."""

    normalized = str(model_id).strip().lower().replace("_", "-")
    for spec in ECHO_MEMORY_MODELS.values():
        candidates = (spec.model_id, *spec.aliases)
        if normalized in {candidate.lower().replace("_", "-") for candidate in candidates}:
            return spec
    choices = ", ".join(ECHO_MEMORY_MODELS)
    raise KeyError(f"unknown Echo-Memory model {model_id!r}; choices: {choices}")


__all__ = [
    "ECHO_BACKBONE_REPOSITORY",
    "ECHO_CHECKPOINT_REPOSITORY",
    "ECHO_CHECKPOINT_REVISION",
    "ECHO_MEMORY_MODELS",
    "ECHO_SOURCE_REPOSITORY",
    "ECHO_SOURCE_REVISION",
    "EchoCheckpointAvailability",
    "EchoMemoryModelSpec",
    "get_echo_memory_model_spec",
]
