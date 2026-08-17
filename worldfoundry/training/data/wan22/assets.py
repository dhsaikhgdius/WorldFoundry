"""Official Wan2.2 A14B text assets used by rollout caching."""

from __future__ import annotations

from dataclasses import dataclass

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec

WAN22_T2V_A14B_REPOSITORY = "Wan-AI/Wan2.2-T2V-A14B"
WAN22_TOKENIZER_FILES = (
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)


@dataclass(frozen=True, slots=True)
class Wan22TextCheckpoints:
    text_encoder: CheckpointSpec
    tokenizer: CheckpointSpec


def wan22_text_checkpoints(
    *,
    repository: str = WAN22_T2V_A14B_REPOSITORY,
    revision: str = "main",
) -> Wan22TextCheckpoints:
    """Select UMT5 weights and tokenizer files from the official A14B repo."""

    return Wan22TextCheckpoints(
        text_encoder=CheckpointSpec(
            repo_id=repository,
            revision=revision,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
        ),
        tokenizer=CheckpointSpec(
            repo_id=repository,
            revision=revision,
            files=WAN22_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
        ),
    )


__all__ = [
    "WAN22_T2V_A14B_REPOSITORY",
    "WAN22_TOKENIZER_FILES",
    "Wan22TextCheckpoints",
    "wan22_text_checkpoints",
]
