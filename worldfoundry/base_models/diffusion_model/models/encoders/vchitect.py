"""Native Vchitect SD3-style three-encoder prompt conditioner."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...components import ComponentBuildContext
from ...contracts import Conditioning, DiffusionRequest
from ...loaders import NativeCheckpointResolver


class VchitectPromptConditioner:
    def __init__(self, root, *, device: torch.device, dtype: torch.dtype) -> None:
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

        self.device = device
        self.dtype = dtype
        self.tokenizer = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer", local_files_only=True)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer_2", local_files_only=True)
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(root, subfolder="tokenizer_3", local_files_only=True)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            root,
            subfolder="text_encoder",
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            root,
            subfolder="text_encoder_2",
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        self.text_encoder_3 = T5EncoderModel.from_pretrained(
            root,
            subfolder="text_encoder_3",
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        for model in (self.text_encoder, self.text_encoder_2, self.text_encoder_3):
            model.eval().requires_grad_(False)

    @torch.no_grad()
    def _clip(self, prompts: Sequence[str], *, second: bool) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.tokenizer_2 if second else self.tokenizer
        encoder = self.text_encoder_2 if second else self.text_encoder
        tokens = tokenizer(
            list(prompts),
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        output = encoder(tokens, output_hidden_states=True)
        return output.hidden_states[-2].to(self.dtype), output.text_embeds.to(self.dtype)

    @torch.no_grad()
    def _t5(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer_3(
            list(prompts),
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return self.text_encoder_3(
            tokens.input_ids.to(self.device),
            attention_mask=tokens.attention_mask.to(self.device),
        ).last_hidden_state.to(self.dtype)

    def _branch(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        clip_1, pooled_1 = self._clip(prompts, second=False)
        clip_2, pooled_2 = self._clip(prompts, second=True)
        clip = torch.cat((clip_1, clip_2), dim=-1)
        clip = torch.nn.functional.pad(clip, (0, 4096 - clip.shape[-1]))
        return {
            "prompt_embeds": torch.cat((clip, self._t5(prompts)), dim=-2),
            "pooled_prompt_embeds": torch.cat((pooled_1, pooled_2), dim=-1),
        }

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        del device, dtype
        negative = request.negative_prompts or ("",) * request.batch_size
        return Conditioning(
            positive=self._branch(request.prompts),
            negative=self._branch(negative),
        )


def build_vchitect_prompt_conditioner(context: ComponentBuildContext) -> VchitectPromptConditioner:
    materialized = NativeCheckpointResolver().materialize(context.require_checkpoint("resources"))
    return VchitectPromptConditioner(
        materialized.root,
        device=context.policy.device,
        dtype=context.policy.dtype,
    )


__all__ = ["VchitectPromptConditioner", "build_vchitect_prompt_conditioner"]
