"""Framework-owned chunked diffusion execution with persistent model caches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from ..contracts import (
    Conditioning,
    DenoiserOutput,
    DiffusionOutput,
    DiffusionRequest,
    EncodedLatentInitializer,
    LatentInitialization,
    SchedulerStep,
)
from ..extensions import DiffusionRunContext
from .base import NativeDiffusionRunner


@runtime_checkable
class ChunkedCacheDenoiser(Protocol):
    """Architecture surface required by the generic chunked-cache runner."""

    def streaming_cache_layout(self) -> tuple[bool, ...]:
        """Return one flag per block; true denotes recurrent state cache."""


class ChunkedKVCacheRunner(NativeDiffusionRunner):
    """Denoise temporal chunks while the framework owns cache lifetime.

    Attention implementations remain model components.  Chunk traversal,
    classifier-free guidance, cache windows, sink-token retention, scheduler
    updates, and decode lifecycle are shared infrastructure.
    """

    def __init__(
        self,
        *,
        base_chunk_frames: int,
        num_cached_chunks: int = 2,
        sink_token: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(self.components.denoiser, ChunkedCacheDenoiser):
            raise TypeError("chunked-kv-cache requires a ChunkedCacheDenoiser component")
        self.cache_denoiser = self.components.denoiser
        self.base_chunk_frames = int(base_chunk_frames)
        self.num_cached_chunks = int(num_cached_chunks)
        self.sink_token = bool(sink_token)
        if self.base_chunk_frames <= 0:
            raise ValueError("base_chunk_frames must be positive")

    @staticmethod
    def _segments(total_frames: int, base_chunk_frames: int) -> tuple[int, ...]:
        remainder = total_frames % base_chunk_frames
        num_chunks = total_frames // base_chunk_frames
        indices = [0]
        for index in range(num_chunks):
            next_index = indices[-1] + base_chunk_frames
            if index == 0:
                next_index += remainder
            indices.append(next_index)
        return tuple(indices)

    @staticmethod
    def _new_cache(num_chunks: int, num_blocks: int) -> list[list[list[Tensor | None]]]:
        return [[[None] * 6 for _ in range(num_blocks)] for _ in range(num_chunks)]

    @staticmethod
    def _promote_full_history(
        cache: list[list[list[Tensor | None]]],
        chunk_index: int,
        state_blocks: tuple[bool, ...],
    ) -> None:
        if chunk_index == 0:
            return
        for block_index, is_state in enumerate(state_blocks):
            if is_state:
                continue
            previous = cache[chunk_index - 1][block_index]
            current = cache[chunk_index][block_index]
            if previous[0] is not None and current[0] is not None:
                current[0] = torch.cat((previous[0], current[0]), dim=-1)
                current[1] = torch.cat((previous[1], current[1]), dim=-1)
                current[2] = torch.cat((previous[2], current[2]), dim=-1)
            elif previous[0] is not None:
                current[0], current[1], current[2] = previous[0], previous[1], previous[2]
            if previous[-1] is not None and current[-1] is not None:
                current[-1] = torch.cat((previous[-1], current[-1]), dim=2)
            elif previous[-1] is not None:
                current[-1] = previous[-1]
            cache[chunk_index - 1][block_index] = [None] * len(previous)

    @staticmethod
    def _accumulate_cache(
        cache: list[list[list[Tensor | None]]],
        chunk_index: int,
        *,
        state_blocks: tuple[bool, ...],
        num_cached_chunks: int,
        sink_token: bool,
        chunk_indices: tuple[int, ...],
        spatial_tokens: int,
    ) -> tuple[list[list[Tensor | None]], int, int]:
        current = cache[chunk_index]
        if chunk_index == 0:
            return current, 0, 0

        start_chunk = max(chunk_index - num_cached_chunks, 0) if num_cached_chunks > 0 else 0
        full_history = num_cached_chunks < 0
        num_cached_frames = 0
        sink_frames = 0

        for block_index, is_state in enumerate(state_blocks):
            if is_state:
                previous = cache[chunk_index - 1][block_index]
                current[block_index][0] = previous[0]
                current[block_index][1] = previous[1]
                current[block_index][-1] = previous[-1]
                continue

            if full_history:
                previous = cache[chunk_index - 1][block_index]
                current[block_index] = [
                    previous[0],
                    previous[1],
                    previous[2],
                    None,
                    None,
                    previous[-1],
                ]
                if previous[0] is not None:
                    num_cached_frames = int(previous[0].shape[-1]) // spatial_tokens
                continue

            valid_chunks = list(range(start_chunk, chunk_index))
            if num_cached_chunks > 0 and sink_token:
                window_start = max(chunk_index - num_cached_chunks + 1, 0)
                if window_start > 0:
                    valid_chunks = [0, *range(window_start, chunk_index)]
                    sink_frames = chunk_indices[1] - chunk_indices[0]

            previous_q = previous_k = previous_v = previous_tconv = None
            for previous_index in range(chunk_index):
                if previous_index not in valid_chunks:
                    cache[previous_index][block_index] = [None] * 6
                    continue
                previous = cache[previous_index][block_index]
                if previous[0] is not None:
                    if previous_q is None:
                        previous_q = previous[0].clone()
                        previous_k = previous[1].clone()
                        previous_v = previous[2].clone()
                    else:
                        previous_q = torch.cat((previous_q, previous[0]), dim=-1)
                        previous_k = torch.cat((previous_k, previous[1]), dim=-1)
                        previous_v = torch.cat((previous_v, previous[2]), dim=-1)
                if previous[-1] is not None:
                    previous_tconv = (
                        previous[-1].clone()
                        if previous_tconv is None
                        else torch.cat((previous_tconv, previous[-1]), dim=2)
                    )
            current[block_index] = [
                previous_q,
                previous_k,
                previous_v,
                None,
                None,
                previous_tconv,
            ]
            if previous_q is not None:
                num_cached_frames = int(previous_q.shape[-1]) // spatial_tokens
        return current, sink_frames, num_cached_frames

    def _prepare(
        self,
        request: DiffusionRequest,
    ) -> tuple[DiffusionRunContext, Tensor, Mapping[str, object]]:
        generator = self._generator(request.sampling.seed)
        conditioning = self.components.conditioner.encode(
            request,
            device=self.device,
            dtype=self.dtype,
        )
        if not isinstance(conditioning, Conditioning):
            raise TypeError("conditioner.encode must return Conditioning")
        context = DiffusionRunContext(
            request=request,
            components=self.components,
            conditioning=conditioning,
            generator=generator,
        )
        for extension in self.extensions:
            extension.on_run_start(context)
        for extension in self.extensions:
            context.conditioning = extension.prepare_conditioning(context, context.conditioning)
            if not isinstance(context.conditioning, Conditioning):
                raise TypeError(f"{extension.extension_id}.prepare_conditioning must return Conditioning")

        initializer = self.components.latent_initializer
        encoder = self.components.latent_encoder
        if encoder is None or not isinstance(initializer, EncodedLatentInitializer):
            raise TypeError("chunked-kv-cache requires an encoded latent initializer")
        initialized = initializer.initialize_with_encoder(
            request,
            latent_encoder=encoder,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        if not isinstance(initialized, LatentInitialization):
            raise TypeError("encoded latent initializer must return LatentInitialization")
        overlap = sorted(set(context.conditioning.shared) & set(initialized.conditioning))
        if overlap:
            raise ValueError(f"initializer conditions overlap conditioner values: {overlap}")
        context.conditioning = Conditioning(
            positive=context.conditioning.positive,
            negative=context.conditioning.negative,
            shared={**context.conditioning.shared, **initialized.conditioning},
        )
        return context, initialized.latents, initialized.artifacts

    @staticmethod
    def _chunk_conditioning(
        values: Mapping[str, object],
        *,
        start: int,
        end: int,
        rope_start: int,
        rope_end: int,
        frame_index: Tensor | None,
        cache: list[list[Tensor | None]],
        save_cache: bool,
    ) -> dict[str, object]:
        result = dict(values)
        source = result.get("image_vae_embeds")
        if isinstance(source, Tensor):
            result["image_vae_embeds"] = source[:, :, start:end]
        result.update(
            {
                "start_f": rope_start,
                "end_f": rope_end,
                "frame_index": frame_index,
                "save_kv_cache": save_cache,
                "kv_cache": cache,
            }
        )
        return result

    def _predict_branch(
        self,
        context: DiffusionRunContext,
        latents: Tensor,
        *,
        branch: str,
        conditioning: Mapping[str, object],
    ) -> DenoiserOutput:
        output = self._call_denoiser_with_conditioning(
            context,
            latents=latents,
            branch=branch,
            conditioning=conditioning,
        )
        return output

    @staticmethod
    def _output_cache(output: DenoiserOutput) -> list[list[Tensor | None]]:
        cache = output.extras.get("kv_cache")
        if not isinstance(cache, list):
            raise TypeError("chunked denoiser must return its kv_cache in DenoiserOutput.extras")
        return cache

    @torch.no_grad()
    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        context: DiffusionRunContext | None = None
        try:
            context, latents, artifacts = self._prepare(request)
            if latents.ndim != 5:
                raise ValueError("chunked-kv-cache requires BCTHW latents")
            total_frames = int(latents.shape[2])
            if total_frames <= self.base_chunk_frames:
                raise ValueError(
                    "chunked-kv-cache requires more latent frames than one base chunk: "
                    f"{total_frames} <= {self.base_chunk_frames}"
                )
            chunk_indices = self._segments(total_frames, self.base_chunk_frames)
            num_chunks = len(chunk_indices) - 1
            state_blocks = self.cache_denoiser.streaming_cache_layout()
            if not state_blocks:
                raise ValueError("chunked denoiser returned an empty cache layout")

            requested_cached = int(request.inputs.get("num_cached_blocks", self.num_cached_chunks))
            requested_sink = bool(request.inputs.get("sink_token", self.sink_token))
            positive_cache = self._new_cache(num_chunks, len(state_blocks))
            negative_cache = (
                self._new_cache(num_chunks, len(state_blocks))
                if context.conditioning.negative and request.sampling.guidance_scale > 1.0
                else None
            )
            spatial_tokens = int(latents.shape[-2] * latents.shape[-1])
            shared = dict(context.conditioning.shared)
            positive_base = {**shared, **context.conditioning.positive}
            negative_base = (
                {**shared, **context.conditioning.negative}
                if negative_cache is not None
                else None
            )

            for chunk_index in range(num_chunks):
                start, end = chunk_indices[chunk_index : chunk_index + 2]
                current_positive, sink_frames, num_cached_frames = self._accumulate_cache(
                    positive_cache,
                    chunk_index,
                    state_blocks=state_blocks,
                    num_cached_chunks=requested_cached,
                    sink_token=requested_sink,
                    chunk_indices=chunk_indices,
                    spatial_tokens=spatial_tokens,
                )
                current_negative = None
                if negative_cache is not None:
                    current_negative, _, _ = self._accumulate_cache(
                        negative_cache,
                        chunk_index,
                        state_blocks=state_blocks,
                        num_cached_chunks=requested_cached,
                        sink_token=requested_sink,
                        chunk_indices=chunk_indices,
                        spatial_tokens=spatial_tokens,
                    )

                cache_start = max(chunk_index - requested_cached, 0) if requested_cached > 0 else 0
                frame_index = None
                if sink_frames > 0:
                    sink_index = torch.arange(sink_frames, device=latents.device)
                    non_sink_count = num_cached_frames - sink_frames + (end - start)
                    window_start = end - non_sink_count
                    frame_index = torch.cat(
                        (sink_index, torch.arange(window_start, end, device=latents.device))
                    )
                    rope_start, rope_end = 0, end
                else:
                    rope_start, rope_end = chunk_indices[cache_start], end

                schedule = tuple(
                    self.components.scheduler.schedule(
                        request.sampling,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                chunk = latents[:, :, start:end].clone()
                for step in schedule:
                    context.step = step
                    model_latents = self.components.scheduler.scale_model_input(chunk, step)
                    positive_conditioning = self._chunk_conditioning(
                        positive_base,
                        start=start,
                        end=end,
                        rope_start=rope_start,
                        rope_end=rope_end,
                        frame_index=frame_index,
                        cache=current_positive,
                        save_cache=False,
                    )
                    positive = self._predict_branch(
                        context,
                        model_latents,
                        branch="positive",
                        conditioning=positive_conditioning,
                    )
                    prediction = positive.sample
                    if negative_base is not None and current_negative is not None:
                        negative_conditioning = self._chunk_conditioning(
                            negative_base,
                            start=start,
                            end=end,
                            rope_start=rope_start,
                            rope_end=rope_end,
                            frame_index=frame_index,
                            cache=current_negative,
                            save_cache=False,
                        )
                        negative = self._predict_branch(
                            context,
                            model_latents,
                            branch="negative",
                            conditioning=negative_conditioning,
                        )
                        scale = request.sampling.guidance_scale
                        prediction = negative.sample + scale * (positive.sample - negative.sample)
                    chunk = self.components.scheduler.step(
                        prediction,
                        step,
                        chunk,
                        generator=context.generator,
                    )
                    for extension in self.extensions:
                        chunk = extension.after_step(context, chunk)
                latents[:, :, start:end] = chunk

                context.step = SchedulerStep(
                    index=max(request.sampling.num_inference_steps - 1, 0),
                    timestep=torch.zeros((), device=self.device),
                    next_timestep=torch.zeros((), device=self.device),
                )
                committed_positive = self._predict_branch(
                    context,
                    chunk,
                    branch="positive-cache-commit",
                    conditioning=self._chunk_conditioning(
                        positive_base,
                        start=start,
                        end=end,
                        rope_start=rope_start,
                        rope_end=rope_end,
                        frame_index=frame_index,
                        cache=current_positive,
                        save_cache=True,
                    ),
                )
                positive_cache[chunk_index] = self._output_cache(committed_positive)
                if requested_cached < 0:
                    self._promote_full_history(positive_cache, chunk_index, state_blocks)

                if negative_base is not None and current_negative is not None and negative_cache is not None:
                    committed_negative = self._predict_branch(
                        context,
                        chunk,
                        branch="negative-cache-commit",
                        conditioning=self._chunk_conditioning(
                            negative_base,
                            start=start,
                            end=end,
                            rope_start=rope_start,
                            rope_end=rope_end,
                            frame_index=frame_index,
                            cache=current_negative,
                            save_cache=True,
                        ),
                    )
                    negative_cache[chunk_index] = self._output_cache(committed_negative)
                    if requested_cached < 0:
                        self._promote_full_history(negative_cache, chunk_index, state_blocks)

            sample = self.components.decoder.decode(latents, request)
            for extension in self.extensions:
                sample = extension.after_decode(context, sample)
            output = DiffusionOutput(
                sample=sample,
                latents=latents,
                artifacts=artifacts,
                metadata={
                    "model_id": self.model_id,
                    "seed": request.sampling.seed,
                    "num_inference_steps": request.sampling.num_inference_steps,
                    "guidance_scale": request.sampling.guidance_scale,
                    "execution_strategy": "chunked-kv-cache",
                    "base_chunk_frames": self.base_chunk_frames,
                    "num_cached_blocks": requested_cached,
                    "sink_token": requested_sink,
                },
            )
            for extension in reversed(self.extensions):
                extension.on_run_end(context)
            return output
        except BaseException as error:
            if context is not None:
                for extension in reversed(self.extensions):
                    extension.on_run_error(context, error)
            raise


__all__ = ["ChunkedCacheDenoiser", "ChunkedKVCacheRunner"]
