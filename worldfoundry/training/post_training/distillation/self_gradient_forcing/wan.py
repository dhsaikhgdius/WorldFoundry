"""Native Wan bridge for Self-Gradient-Forcing's two execution paths."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import torch
from torch import Tensor, nn

from worldfoundry.core.attention.causal_cache import commit_causal_video_cache_block
from worldfoundry.core.nn.diffusion_transformer import velocity_to_denoised

from ..self_forcing.contracts import CachePayload
from ..self_forcing.wan import WanSelfForcingChunkAdapter


class WanSelfGradientForcingAdapter(WanSelfForcingChunkAdapter):
    """Add noisy context commits and full teacher forcing to the causal Wan seam."""

    def __init__(
        self,
        module: nn.Module,
        *,
        frames_per_block: int = 1,
        checkpoint_identity: str | None = None,
    ) -> None:
        super().__init__(
            module,
            frames_per_block=frames_per_block,
            checkpoint_identity=checkpoint_identity,
        )
        if not callable(getattr(self._graph, "forward", None)):
            raise TypeError("causal Wan graph does not expose a forward method")

    def commit_context_chunk(
        self,
        context_chunk: Tensor,
        context_timesteps: Tensor,
        *,
        block_index: int,
        start_frame: int,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        cache: CachePayload,
    ) -> CachePayload:
        """Overwrite the active cache block at the configured context noise."""

        del sample_ids
        if not isinstance(cache, MutableMapping):
            raise TypeError("causal Wan cache must be a mutable mapping")
        with torch.no_grad():
            self._flow_prediction(
                context_chunk.detach(),
                context_timesteps,
                start_frame=start_frame,
                conditioning=conditioning,
                cache=cache,
            )
        commit_causal_video_cache_block(
            cache,
            block_index=block_index,
            start_frame=start_frame,
            frame_count=int(context_chunk.shape[2]),
        )
        return cache

    def predict_clean_teacher_forced(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        sigmas: Tensor,
        *,
        clean_context: Tensor,
        context_timesteps: Tensor,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> Tensor:
        """Run the one parallel replay call that restores context gradients."""

        del sample_ids, training
        if noisy_latents.ndim != 5 or clean_context.shape != noisy_latents.shape:
            raise ValueError("Wan teacher forcing requires matching BCTHW noisy and context latents")
        batch, _, frames, height, width = (int(value) for value in noisy_latents.shape)
        if tuple(timesteps.shape) != (batch,) or tuple(sigmas.shape) != (batch,):
            raise ValueError("Wan teacher forcing requires one target timestep and sigma per sample")
        if tuple(context_timesteps.shape) != (batch, frames):
            raise ValueError("Wan teacher-forced context_timesteps must have shape [B,F]")
        patch = tuple(int(value) for value in self._graph.patch_size)
        if frames % patch[0] or height % patch[1] or width % patch[2]:
            raise ValueError("Wan teacher-forced latents are not divisible by the model patch size")
        sequence_length = (frames // patch[0]) * (height // patch[1]) * (width // patch[2])
        parameter = next(self.module.parameters())
        model_input = noisy_latents.to(device=parameter.device, dtype=parameter.dtype)
        model_context = clean_context.detach().to(device=parameter.device, dtype=parameter.dtype)
        model_timesteps = timesteps.to(device=parameter.device, dtype=torch.float32)
        model_timesteps = model_timesteps[:, None].expand(-1, frames)
        model_context_timesteps = context_timesteps.to(device=parameter.device, dtype=torch.float32)
        velocity = self.module(
            x=model_input,
            t=model_timesteps,
            context=self._context(conditioning, noisy_latents),
            seq_len=sequence_length,
            clean_x=model_context,
            aug_t=model_context_timesteps,
        )
        if not isinstance(velocity, Tensor) or velocity.shape != model_input.shape:
            raise ValueError("causal Wan teacher-forced graph must return a matching flow tensor")
        velocity = velocity.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        sigma = sigmas.to(device=noisy_latents.device, dtype=torch.float32).reshape(
            (batch,) + (1,) * (noisy_latents.ndim - 1)
        )
        return velocity_to_denoised(noisy_latents, velocity, sigma)


__all__ = ["WanSelfGradientForcingAdapter"]
