"""Frozen prompt encoders for native HunyuanVideo policy rollouts."""

from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.training.models.hunyuan_video import hunyuan_video_model_contract


class HunyuanVideoTextFeatureEncoder:
    """Keep exactly the text tensors consumed by ``HunyuanVideoTrainAdapter``."""

    def __init__(
        self,
        conditioner: object,
        *,
        model_recipe: str,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not callable(getattr(conditioner, "encode", None)):
            raise TypeError("HunyuanVideo cache conditioner must expose encode")
        self.conditioner = conditioner
        self.contract = hunyuan_video_model_contract(model_recipe)
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    def tensor_layouts(self) -> dict[str, str]:
        if self.contract.architecture == "original":
            return {
                "text_states": "sequence-features",
                "text_mask": "sequence",
                "text_states_2": "features",
            }
        return {
            "text_states": "sequence-features",
            "text_mask": "sequence",
            "byt5_text_states": "sequence-features",
            "byt5_text_mask": "sequence",
        }

    def encode(
        self,
        *,
        sample_id: str,
        prompt: str,
        frames: int,
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        request = DiffusionRequest(
            prompt=(prompt,),
            height=height,
            width=width,
            num_frames=frames,
            sampling=SamplingConfig(guidance_scale=self.contract.embedded_guidance_scale),
            metadata={"sample_ids": (sample_id,)},
        )
        with torch.no_grad():
            encoded = self.conditioner.encode(
                request,
                device=self.device,
                dtype=self.dtype,
            )
        if not isinstance(encoded, Conditioning):
            raise TypeError("HunyuanVideo cache conditioner must return Conditioning")

        result: dict[str, torch.Tensor] = {}
        for name in self.tensor_layouts:
            value = encoded.positive.get(name)
            if not isinstance(value, torch.Tensor) or value.ndim < 2 or int(value.shape[0]) != 1:
                raise ValueError(f"HunyuanVideo conditioner returned an incompatible {name} tensor")
            result[name] = value[0].detach().cpu().contiguous()
        return result


__all__ = ["HunyuanVideoTextFeatureEncoder"]
