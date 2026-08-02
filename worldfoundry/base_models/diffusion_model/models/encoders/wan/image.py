"""Checkpoint-shaped Wan image conditioner built from the canonical CLIP layers."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .clip import clip_xlm_roberta_vit_h_14


class WanImageEncoder(torch.nn.Module):
    """Wan CLIP vision encoder used by serialized inference configurations."""

    def __init__(self, image_encoder_pretrained_path: str | None = None) -> None:
        super().__init__()
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device="cpu",
        )
        self.image_encoder_pretrained_path = image_encoder_pretrained_path

    def encode_image(self, videos) -> torch.Tensor:
        size = (self.model.image_size, self.model.image_size)
        images = torch.cat(
            [
                F.interpolate(
                    video,
                    size=size,
                    mode="bicubic",
                    align_corners=False,
                )
                for video in videos
            ]
        )
        images = self.transforms.transforms[-1](images.mul(0.5).add(0.5))
        dtype = next(self.model.visual.parameters()).dtype
        return self.model.visual(images.to(dtype=dtype), use_31_block=True).clone()

    @staticmethod
    def state_dict_converter():
        return WanImageEncoderStateDictConverter()


class WanImageEncoderStateDictConverter:
    def from_diffusers(self, state_dict):
        return state_dict

    def from_civitai(self, state_dict):
        return {
            f"model.{name}": parameter
            for name, parameter in state_dict.items()
        }


__all__ = ["WanImageEncoder", "WanImageEncoderStateDictConverter"]
