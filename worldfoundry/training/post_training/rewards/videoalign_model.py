# SPDX-License-Identifier: Apache-2.0
"""Native VideoAlign model topology and strict checkpoint loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import Qwen2VLForConditionalGeneration

from worldfoundry.core.checkpoint import load_tensor_state_dict

from .videoalign_preprocessing import (
    VIDEOALIGN_SPECIAL_TOKEN_IDS,
    pool_videoalign_special_tokens,
)


class NativeVideoAlignRewardModel(Qwen2VLForConditionalGeneration):
    """Qwen2-VL with the official bias-free scalar reward head."""

    def __init__(
        self,
        config: object,
        *,
        special_token_ids: Sequence[int] = VIDEOALIGN_SPECIAL_TOKEN_IDS,
        output_dim: int = 1,
    ) -> None:
        super().__init__(config)
        if int(output_dim) != 1:
            raise ValueError("official VideoAlign checkpoint requires output_dim=1")
        text_config = getattr(config, "text_config", config)
        hidden_size = int(getattr(text_config, "hidden_size"))
        self.rm_head = nn.Linear(hidden_size, 1, bias=False)
        self.special_token_ids = tuple(int(value) for value in special_token_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        rope_deltas: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Mapping[str, torch.Tensor]:
        if labels is not None:
            raise ValueError("VideoAlign reward inference does not accept labels")
        if input_ids is None:
            raise ValueError("VideoAlign special-token pooling requires input_ids")
        kwargs.pop("return_dict", None)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False if use_cache is None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            **kwargs,
        )
        sequence_scores = self.rm_head(outputs[0])
        pooled = pool_videoalign_special_tokens(
            sequence_scores,
            input_ids,
            self.special_token_ids,
        )
        return {"logits": pooled}


def remap_videoalign_checkpoint_keys(
    state_dict: Mapping[str, torch.Tensor],
    target_keys: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Apply the audited Transformers layout migration, then require equality."""

    if not isinstance(state_dict, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state_dict.items()
    ):
        raise TypeError("VideoAlign checkpoint must be a string-to-tensor mapping")
    expected = set(str(value) for value in target_keys)
    if set(state_dict) == expected:
        return dict(state_dict)
    prefix_migrations = (
        ("base_model.model.visual.", "base_model.model.model.visual."),
        (
            "base_model.model.model.",
            "base_model.model.model.language_model.",
        ),
    )
    remapped: dict[str, torch.Tensor] = {}
    for source_key, tensor in state_dict.items():
        target_key = source_key
        for old_prefix, new_prefix in prefix_migrations:
            if source_key.startswith(old_prefix):
                target_key = new_prefix + source_key.removeprefix(old_prefix)
                break
        if target_key in remapped:
            raise ValueError(f"VideoAlign checkpoint key migration collides at {target_key!r}")
        remapped[target_key] = tensor
    missing = sorted(expected - set(remapped))
    unexpected = sorted(set(remapped) - expected)
    if missing or unexpected:
        raise ValueError(
            "VideoAlign checkpoint keys differ from the native model after the audited migration: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    return remapped


def load_videoalign_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    """Weights-only load with exact key and shape enforcement."""

    if not isinstance(model, nn.Module):
        raise TypeError("VideoAlign model must be an nn.Module")
    source = Path(checkpoint_path)
    state_dict = load_tensor_state_dict(source, map_location="cpu", mmap=True)
    remapped = remap_videoalign_checkpoint_keys(state_dict, tuple(model.state_dict()))
    incompatible = model.load_state_dict(remapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict VideoAlign state loading returned incompatible keys")


__all__ = [
    "NativeVideoAlignRewardModel",
    "load_videoalign_checkpoint",
    "remap_videoalign_checkpoint_keys",
]
