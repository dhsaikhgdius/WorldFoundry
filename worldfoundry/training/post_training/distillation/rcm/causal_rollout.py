"""Native block/KV self-forcing rollout used by Causal-rCM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import nn

from worldfoundry.core.attention.block_pattern import AttnMaskSpec

from .causal import CausalRolloutRequest
from .contracts import RCMTrainingBatch


@runtime_checkable
class CausalBlockModelAdapter(Protocol):
    """Model-family KV operations needed by the shared rollout algorithm."""

    module: object

    def allocate_cache(
        self,
        *,
        batch_size: int,
        max_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> object: ...

    def synchronize_tensor(self, value: torch.Tensor) -> torch.Tensor: ...

    def predict_block_velocity(
        self,
        block_latents: torch.Tensor,
        rf_timesteps: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        attention_mask: AttnMaskSpec,
        cache: object,
        cache_mode: Literal["read", "append"],
        training: bool,
    ) -> torch.Tensor: ...


class NativeCausalSelfForcingRollout:
    """Execute the fixed official chunk loop over a native causal model."""

    def __init__(self, model: CausalBlockModelAdapter) -> None:
        if not isinstance(model, CausalBlockModelAdapter):
            raise TypeError("model must implement CausalBlockModelAdapter")
        if not isinstance(model.module, nn.Module):
            raise TypeError("model.module must be an nn.Module")
        self.model = model
        self.module = model.module

    def _noise(
        self,
        reference: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        sampled = torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        synchronized = self.model.synchronize_tensor(sampled)
        if not isinstance(synchronized, torch.Tensor) or synchronized.shape != sampled.shape:
            raise ValueError("causal rollout synchronization must preserve tensor shape")
        return synchronized

    @staticmethod
    def _validate_request(
        clean: torch.Tensor,
        request: CausalRolloutRequest,
    ) -> int:
        pattern = request.pattern
        frames = int(clean.shape[2])
        if pattern.frame_tokens <= 0 or pattern.first_chunk_frames <= 0 or pattern.chunk_frames <= 0:
            raise ValueError("causal rollout block pattern must have positive geometry")
        if frames < pattern.first_chunk_frames:
            raise ValueError("causal rollout first chunk exceeds latent frames")
        if (frames - pattern.first_chunk_frames) % pattern.chunk_frames:
            raise ValueError("causal rollout pattern does not cover latent frames")
        blocks = 1 + (frames - pattern.first_chunk_frames) // pattern.chunk_frames
        if len(request.steps_per_block) != blocks or len(request.timesteps_per_block) != blocks:
            raise ValueError("causal rollout request must define every block")
        for steps, times in zip(
            request.steps_per_block,
            request.timesteps_per_block,
            strict=True,
        ):
            if isinstance(steps, bool) or steps <= 0 or len(times) != steps:
                raise ValueError("each causal block needs one RF time per positive step")
            if times[0] != 1.0:
                raise ValueError("each causal rollout block must start from RF time one")
            if any(not 0 < value <= 1 for value in times):
                raise ValueError("causal rollout RF times must be in (0,1]")
            if any(left <= right for left, right in zip(times, times[1:])):
                raise ValueError("causal rollout RF times must be strictly descending")
        return blocks

    def rollout(
        self,
        batch: RCMTrainingBatch,
        request: CausalRolloutRequest,
        *,
        training: bool,
        generator: object | None,
    ) -> torch.Tensor:
        if not isinstance(batch, RCMTrainingBatch):
            raise TypeError("batch must be RCMTrainingBatch")
        clean = batch.clean_latents
        if not isinstance(clean, torch.Tensor) or clean.ndim != 5:
            raise TypeError("causal self-forcing requires [B,C,T,H,W] torch latents")
        if not isinstance(request, CausalRolloutRequest):
            raise TypeError("request must be CausalRolloutRequest")
        if not isinstance(training, bool):
            raise TypeError("training must be a bool")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        blocks = self._validate_request(clean, request)
        pattern = request.pattern
        max_steps = max(request.steps_per_block)
        noises = [self._noise(clean, generator=generator) for _ in range(max_steps)]
        cache = self.model.allocate_cache(
            batch_size=batch.batch_size,
            max_tokens=int(clean.shape[2]) * pattern.frame_tokens,
            device=clean.device,
            dtype=clean.dtype,
        )
        generated_blocks: list[torch.Tensor] = []
        for block_index in range(blocks):
            frame_start = pattern.blocks_to_frames(block_index)
            block_frames = pattern.block_size(block_index)
            frame_end = frame_start + block_frames
            times = request.timesteps_per_block[block_index]
            attention_mask = AttnMaskSpec(
                mode="block_causal",
                pattern=pattern,
                q_block_offset=block_index,
            )
            block: torch.Tensor | None = None
            final_time: torch.Tensor | None = None
            for step, time_value in enumerate(times):
                time = torch.full(
                    (batch.batch_size,),
                    float(time_value),
                    device=clean.device,
                    dtype=torch.float32,
                )
                noise = noises[step][:, :, frame_start:frame_end]
                if block is None:
                    block = noise
                else:
                    coefficient = time.reshape(batch.batch_size, 1, 1, 1, 1)
                    block = (1.0 - coefficient) * block + coefficient * noise
                final = step + 1 == len(times)
                context = torch.enable_grad() if training and final else torch.no_grad()
                with context:
                    velocity = self.model.predict_block_velocity(
                        block,
                        time,
                        sample_ids=batch.sample_ids,
                        conditioning=batch.conditioning,
                        attention_mask=attention_mask,
                        cache=cache,
                        cache_mode="read",
                        training=training and final,
                    )
                    if not isinstance(velocity, torch.Tensor) or velocity.shape != block.shape:
                        raise ValueError("causal block velocity must match its latent block")
                    coefficient = time.reshape(batch.batch_size, 1, 1, 1, 1)
                    block = block - coefficient * velocity
                final_time = time
            assert block is not None and final_time is not None
            generated_blocks.append(block)
            with torch.no_grad():
                appended = self.model.predict_block_velocity(
                    block.detach(),
                    torch.zeros_like(final_time),
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    attention_mask=attention_mask,
                    cache=cache,
                    cache_mode="append",
                    training=False,
                )
                if not isinstance(appended, torch.Tensor) or appended.shape != block.shape:
                    raise ValueError("causal cache append must preserve block velocity shape")
        result = torch.cat(generated_blocks, dim=2)
        if result.shape != clean.shape:
            raise RuntimeError("causal block rollout did not reconstruct the latent frame layout")
        return result


__all__ = ["CausalBlockModelAdapter", "NativeCausalSelfForcingRollout"]
