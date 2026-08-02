"""Thin training bridge to WorldFoundry's in-tree causal Wan graph."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import torch
from torch import Tensor, nn

from worldfoundry.core.attention.causal_cache import (
    CausalVideoCacheGeometry,
    allocate_causal_video_cache,
    begin_causal_video_cache_block,
    causal_video_cache_geometry,
    causal_video_cache_state,
    commit_causal_video_cache_block,
    finish_causal_video_cache_call,
)
from worldfoundry.core.attention.kv_cache_policy import CacheState
from worldfoundry.core.nn.diffusion_transformer import velocity_to_denoised

from .contracts import CachePayload


def _backbone(module: nn.Module) -> nn.Module:
    """Resolve a causal graph through optional DDP/FSDP/PEFT ownership."""

    required = {"blocks", "dim", "model_type", "num_heads", "num_layers", "patch_size"}
    queue = [module]
    visited: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if required.issubset(set(dir(candidate))):
            return candidate
        for name in ("module", "base_model", "model"):
            child = getattr(candidate, name, None)
            if isinstance(child, nn.Module):
                queue.append(child)
    raise TypeError("wrapped causal Wan module does not expose the native causal graph")


class WanSelfForcingChunkAdapter:
    """Expose the native ``CausalWanModel`` KV forward as a clean predictor.

    This class does not load a checkpoint and does not call a synthesis runner.
    It only bridges the graph's existing ``kv_cache``/``current_start`` forward
    into the model-neutral training rollout.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        frames_per_block: int = 1,
        checkpoint_identity: str | None = None,
    ) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("causal Wan module must be torch.nn.Module")
        graph = _backbone(module)
        required = {
            "blocks",
            "dim",
            "local_attn_size",
            "model_type",
            "num_heads",
            "num_layers",
            "patch_size",
            "text_len",
        }
        missing = sorted(name for name in required if not hasattr(graph, name))
        if missing:
            raise TypeError(f"causal Wan graph is missing required attributes: {missing}")
        if str(graph.model_type) != "t2v":
            raise NotImplementedError("Self-Forcing Wan bridge currently supports the causal T2V graph only")
        patch_size = tuple(int(value) for value in graph.patch_size)
        if len(patch_size) != 3 or patch_size[0] != 1:
            raise ValueError("causal Wan Self-Forcing requires temporal patch size 1")
        if int(graph.num_layers) != len(graph.blocks):
            raise ValueError("causal Wan num_layers differs from its block inventory")
        if int(graph.dim) % int(graph.num_heads):
            raise ValueError("causal Wan hidden dimension must be divisible by num_heads")
        if isinstance(frames_per_block, bool) or int(frames_per_block) <= 0:
            raise ValueError("frames_per_block must be a positive integer")
        self.module = module
        self._graph = graph
        self.frames_per_block = int(frames_per_block)
        if checkpoint_identity is not None:
            if not isinstance(checkpoint_identity, str) or not checkpoint_identity.strip():
                raise ValueError("checkpoint_identity must be a non-empty string")
            self.checkpoint_identity: str | None = checkpoint_identity.strip()
        else:
            self.checkpoint_identity = None

    def _sink_frames(self) -> int:
        values = {int(getattr(getattr(block, "self_attn", None), "sink_size", 0)) for block in self._graph.blocks}
        if len(values) != 1:
            raise ValueError("causal Wan blocks disagree on sink_size")
        return next(iter(values))

    def _context(self, conditioning: Mapping[str, object], reference: Tensor) -> Tensor:
        context = conditioning.get("context")
        prompt_embeds = conditioning.get("prompt_embeds")
        if context is not None and prompt_embeds is not None:
            raise ValueError("Wan conditioning cannot contain both 'context' and 'prompt_embeds'")
        value = context if context is not None else prompt_embeds
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise TypeError("causal Wan conditioning requires [B,L,D] context or prompt_embeds")
        if int(value.shape[0]) != int(reference.shape[0]):
            raise ValueError("causal Wan text conditioning batch differs from the latent batch")
        parameter = next(self.module.parameters())
        return value.to(device=parameter.device, dtype=parameter.dtype)

    def initialize_cache(
        self,
        reference: Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
    ) -> CachePayload:
        del sample_ids
        if reference.ndim != 5:
            raise ValueError("causal Wan Self-Forcing requires BCTHW latents")
        self._context(conditioning, reference)
        batch, _, frames, height, width = (int(value) for value in reference.shape)
        patch = tuple(int(value) for value in self._graph.patch_size)
        if height % patch[1] or width % patch[2]:
            raise ValueError("causal Wan latent spatial shape is not divisible by its patch size")
        frame_sequence_length = (height // patch[1]) * (width // patch[2])
        heads = int(self._graph.num_heads)
        head_dim = int(self._graph.dim) // heads
        parameter = next(self.module.parameters())
        return allocate_causal_video_cache(
            CausalVideoCacheGeometry(
                batch_size=batch,
                total_frames=frames,
                frame_tokens=frame_sequence_length,
                frames_per_block=self.frames_per_block,
                num_layers=int(self._graph.num_layers),
                num_heads=heads,
                head_dim=head_dim,
                local_attention_frames=int(self._graph.local_attn_size),
                sink_frames=self._sink_frames(),
            ),
            device=parameter.device,
            dtype=parameter.dtype,
        )

    @staticmethod
    def _cache(value: CachePayload) -> tuple[list[dict[str, object]], list[dict[str, object]], int, int]:
        if not isinstance(value, Mapping):
            raise TypeError("causal Wan cache must be a mapping")
        geometry = causal_video_cache_geometry(value)
        kv_cache = value.get("kv_cache")
        crossattn_cache = value.get("crossattn_cache")
        if not isinstance(kv_cache, list) or not isinstance(crossattn_cache, list):
            raise TypeError("causal Wan cache mapping has invalid attention caches")
        return (
            kv_cache,
            crossattn_cache,
            geometry.frame_tokens,
            geometry.sequence_tokens,
        )

    def _flow_prediction(
        self,
        chunk: Tensor,
        timesteps: Tensor,
        *,
        start_frame: int,
        conditioning: Mapping[str, object],
        cache: CachePayload,
    ) -> Tensor:
        if chunk.ndim != 5:
            raise ValueError("causal Wan chunk must have BCTHW layout")
        kv_cache, crossattn_cache, frame_sequence_length, sequence_length = self._cache(cache)
        if frame_sequence_length <= 0 or sequence_length <= 0:
            raise ValueError("causal Wan cache geometry is invalid")
        parameter = next(self.module.parameters())
        model_chunk = chunk.to(device=parameter.device, dtype=parameter.dtype)
        model_timesteps = timesteps.to(device=parameter.device, dtype=torch.float32)
        model_timesteps = model_timesteps[:, None].expand(-1, int(chunk.shape[2]))
        prediction = self.module(
            x=model_chunk,
            t=model_timesteps,
            context=self._context(conditioning, chunk),
            seq_len=sequence_length,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=int(start_frame) * frame_sequence_length,
            cache_start=int(start_frame) * frame_sequence_length,
        )
        finish_causal_video_cache_call(
            cache,
            start_frame=start_frame,
            frame_count=int(chunk.shape[2]),
        )
        if not isinstance(prediction, Tensor) or prediction.shape != model_chunk.shape:
            raise ValueError("causal Wan graph must return a flow tensor matching its chunk")
        return prediction.to(device=chunk.device, dtype=chunk.dtype)

    def predict_clean_chunk(
        self,
        noisy_chunk: Tensor,
        timesteps: Tensor,
        sigmas: Tensor,
        *,
        block_index: int,
        start_frame: int,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        cache: CachePayload,
        training: bool,
    ) -> Tensor:
        del sample_ids, training
        if not isinstance(cache, MutableMapping):
            raise TypeError("causal Wan cache must be a mapping")
        begin_causal_video_cache_block(
            cache,
            block_index=block_index,
            start_frame=start_frame,
            frame_count=int(noisy_chunk.shape[2]),
        )
        velocity = self._flow_prediction(
            noisy_chunk,
            timesteps,
            start_frame=start_frame,
            conditioning=conditioning,
            cache=cache,
        )
        sigma = sigmas.to(device=noisy_chunk.device, dtype=torch.float32).reshape(
            (int(noisy_chunk.shape[0]),) + (1,) * (noisy_chunk.ndim - 1)
        )
        return velocity_to_denoised(noisy_chunk, velocity, sigma)

    def commit_clean_chunk(
        self,
        clean_chunk: Tensor,
        *,
        block_index: int,
        start_frame: int,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        cache: CachePayload,
    ) -> CachePayload:
        del sample_ids
        if not isinstance(cache, MutableMapping):
            raise TypeError("causal Wan cache must be a mapping")
        zeros = torch.zeros(int(clean_chunk.shape[0]), device=clean_chunk.device)
        with torch.no_grad():
            self._flow_prediction(
                clean_chunk.detach(),
                zeros,
                start_frame=start_frame,
                conditioning=conditioning,
                cache=cache,
            )
        commit_causal_video_cache_block(
            cache,
            block_index=block_index,
            start_frame=start_frame,
            frame_count=int(clean_chunk.shape[2]),
        )
        return cache

    def cache_state(self, cache: CachePayload) -> CacheState:
        """Expose the core-owned committed block state to the rollout."""

        if not isinstance(cache, Mapping):
            raise TypeError("causal Wan cache must be a mapping")
        return causal_video_cache_state(cache)


__all__ = ["WanSelfForcingChunkAdapter"]
