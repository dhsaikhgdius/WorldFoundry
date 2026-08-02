"""Native StepVideo latent initialization."""

from __future__ import annotations

import torch

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest


class StepVideoLatentInitializer:
    def initialize(self, request, *, generator, device, dtype):
        if request.height % 16 or request.width % 16:
            raise ValueError("StepVideo height and width must be divisible by 16")
        if request.num_frames < 17 or request.num_frames % 17:
            raise ValueError("StepVideo num_frames must be a positive multiple of 17")
        latent_frames = request.num_frames // 17 * 3
        return torch.randn(
            request.batch_size,
            latent_frames,
            64,
            request.height // 16,
            request.width // 16,
            generator=generator,
            device=device,
            dtype=dtype,
        )


def build_step_video_latent_initializer(context: ComponentBuildContext) -> StepVideoLatentInitializer:
    del context
    return StepVideoLatentInitializer()


__all__ = ["StepVideoLatentInitializer", "build_step_video_latent_initializer"]
