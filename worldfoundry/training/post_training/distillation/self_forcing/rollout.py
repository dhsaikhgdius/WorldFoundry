"""Causal self-rollout shared by training and inference."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod

import torch
import torch.distributed as dist
from torch import Tensor

from worldfoundry.core.attention.kv_cache_policy import CachedBlock, CacheState
from worldfoundry.training.objectives.flow_matching import flow_interpolate

from ...shared.distributed import PostTrainingParallelContext
from ..dmd.contracts import DMDTrainingBatch
from ..dmd.objective import FewStepPrediction, FewStepSchedule
from .config import SelfForcingConfig
from .contracts import CachePayload, CausalChunkAdapter, SelfForcingRollout


def _normal_like(reference: Tensor, *, generator: torch.Generator | None) -> Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _batch_scalar(reference: Tensor, value: float) -> Tensor:
    return torch.full(
        (int(reference.shape[0]),),
        float(value),
        device=reference.device,
        dtype=torch.float32,
    )


def _detach_cache_payload(value: CachePayload, *, path: str = "cache") -> CachePayload:
    """Detach every visible cache embedding while preserving its container shape."""

    if isinstance(value, Tensor):
        return value.detach()
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {key: _detach_cache_payload(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_cache_payload(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(_detach_cache_payload(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(
        f"{path} has unsupported opaque type {type(value).__name__}; "
        "causal adapters must expose cache tensors in nested mappings/lists/tuples"
    )


def _audit_detached_cache(value: CachePayload, *, path: str = "cache") -> None:
    if isinstance(value, Tensor):
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError(f"{path} retained an autograd graph after context commit")
        return
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    items = value.items() if isinstance(value, Mapping) else enumerate(value)
    for key, item in items:
        _audit_detached_cache(item, path=f"{path}[{key!r}]")


class SelfForcingRolloutSampler:
    """Run Algorithm 1 with the same causal path used for full inference."""

    def __init__(
        self,
        adapter: CausalChunkAdapter,
        config: SelfForcingConfig,
        *,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(adapter, CausalChunkAdapter):
            raise TypeError("adapter must implement CausalChunkAdapter")
        if not isinstance(adapter.module, torch.nn.Module):
            raise TypeError("causal adapter.module must be torch.nn.Module")
        if not isinstance(config, SelfForcingConfig):
            raise TypeError("config must be SelfForcingConfig")
        self.adapter = adapter
        self.config = config
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.execution_digest = config.digest

    def _frame_dim(self, reference: Tensor) -> int:
        resolved = self.config.frame_dim % reference.ndim
        if resolved == 0:
            raise ValueError("resolved frame_dim cannot be the batch dimension")
        return resolved

    def _num_blocks(self, reference: Tensor) -> int:
        frame_dim = self._frame_dim(reference)
        frames = int(reference.shape[frame_dim])
        if frames == 0 or frames % self.config.frames_per_block:
            raise ValueError(
                f"latent frame count {frames} must be divisible by frames_per_block={self.config.frames_per_block}"
            )
        return frames // self.config.frames_per_block

    def _adapter_cache_state(
        self,
        cache: CachePayload,
        *,
        fallback: CacheState,
    ) -> CacheState:
        state_fn = getattr(self.adapter, "cache_state", None)
        if state_fn is None:
            return fallback
        if not callable(state_fn):
            raise TypeError("causal adapter cache_state must be callable")
        state = state_fn(cache)
        if not isinstance(state, CacheState):
            raise TypeError("causal adapter cache_state must return core CacheState")
        return state

    def sample_exit_indices(
        self,
        reference: Tensor,
        *,
        generator: torch.Generator | None,
    ) -> tuple[int, ...]:
        """Draw one synchronized exit for a sequence, or one per block."""

        num_blocks = self._num_blocks(reference)
        draw_count = 1 if self.config.exit_step_mode == "sequence" else num_blocks
        if self.parallel_context.rank == 0:
            indices = torch.randint(
                0,
                len(self.config.schedule.sigmas),
                (draw_count,),
                device=reference.device,
                generator=generator,
            )
        else:
            indices = torch.empty(draw_count, device=reference.device, dtype=torch.long)
        if self.parallel_context.world_size > 1:
            group = self.parallel_context.process_group
            source = 0 if group is None else dist.get_global_rank(group, 0)
            dist.broadcast(indices, src=source, group=group)
        values = tuple(int(value) for value in indices.tolist())
        if self.config.exit_step_mode == "sequence":
            return values * num_blocks
        return values

    def rollout(
        self,
        batch: DMDTrainingBatch,
        initial_noise: Tensor,
        *,
        exit_indices: tuple[int, ...],
        generator: torch.Generator | None,
        training: bool,
    ) -> SelfForcingRollout:
        """Execute the common causal path for explicit block exit indices."""

        if not isinstance(batch, DMDTrainingBatch):
            raise TypeError("batch must be DMDTrainingBatch")
        if not isinstance(initial_noise, Tensor) or initial_noise.shape != batch.clean_latents.shape:
            raise ValueError("initial_noise must match the batch latent shape template")
        num_blocks = self._num_blocks(initial_noise)
        if len(exit_indices) != num_blocks:
            raise ValueError("exit_indices must contain one entry per temporal block")
        if any(
            isinstance(index, bool) or not 0 <= int(index) < len(self.config.schedule.sigmas) for index in exit_indices
        ):
            raise ValueError("an exit index falls outside the few-step schedule")

        frame_dim = self._frame_dim(initial_noise)
        with torch.no_grad():
            cache = self.adapter.initialize_cache(
                initial_noise,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
            )
        cache = _detach_cache_payload(cache)
        _audit_detached_cache(cache)
        frame_tokens = prod(int(size) for size in initial_noise.shape[frame_dim + 1 :]) or 1
        cache_state = self._adapter_cache_state(
            cache,
            fallback=CacheState(
                blocks=(),
                frame_tokens=frame_tokens,
                current_block_idx=-1,
            ),
        )
        blocks = initial_noise.split(self.config.frames_per_block, dim=frame_dim)
        outputs: list[Tensor] = []
        start_frame = 0

        for block_index, (block_noise, selected_value) in enumerate(zip(blocks, exit_indices, strict=True)):
            selected = int(selected_value)
            current = block_noise
            predicted_clean: Tensor | None = None
            for step_index in range(selected + 1):
                timesteps = _batch_scalar(current, self.config.schedule.timesteps[step_index])
                sigmas = _batch_scalar(current, self.config.schedule.sigmas[step_index])
                differentiable = bool(training and step_index == selected)
                with torch.set_grad_enabled(differentiable):
                    predicted_clean = self.adapter.predict_clean_chunk(
                        current,
                        timesteps,
                        sigmas,
                        block_index=block_index,
                        start_frame=start_frame,
                        sample_ids=batch.sample_ids,
                        conditioning=batch.conditioning,
                        cache=cache,
                        training=differentiable,
                    )
                if not isinstance(predicted_clean, Tensor) or predicted_clean.shape != current.shape:
                    raise ValueError("causal adapter must return a clean prediction matching its chunk")
                if step_index < selected:
                    with torch.no_grad():
                        fresh_noise = _normal_like(predicted_clean, generator=generator)
                        current = flow_interpolate(
                            predicted_clean,
                            fresh_noise,
                            _batch_scalar(
                                predicted_clean,
                                self.config.schedule.sigmas[step_index + 1],
                            ),
                        )
            assert predicted_clean is not None
            outputs.append(predicted_clean)
            clean_context = predicted_clean.detach()
            with torch.no_grad():
                cache = self.adapter.commit_clean_chunk(
                    clean_context,
                    block_index=block_index,
                    start_frame=start_frame,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    cache=cache,
                )
            cache = _detach_cache_payload(cache)
            _audit_detached_cache(cache)
            block_frames = int(predicted_clean.shape[frame_dim])
            cache_state = self._adapter_cache_state(
                cache,
                fallback=CacheState(
                    blocks=(
                        *cache_state.blocks,
                        CachedBlock(
                            block_idx=block_index,
                            frame_start=start_frame,
                            frame_count=block_frames,
                        ),
                    ),
                    frame_tokens=cache_state.frame_tokens,
                    current_block_idx=block_index,
                ),
            )
            start_frame += block_frames

        clean_latents = torch.cat(outputs, dim=frame_dim)
        return SelfForcingRollout(
            clean_latents=clean_latents,
            exit_indices=tuple(int(index) for index in exit_indices),
            cache_state=cache_state,
        )

    def sample(
        self,
        batch: DMDTrainingBatch,
        schedule: FewStepSchedule,
        *,
        generator: torch.Generator | None,
        training: bool,
    ) -> FewStepPrediction:
        """DMD student-sampler seam; every invocation creates a fresh rollout."""

        if schedule.digest != self.config.schedule.digest:
            raise ValueError("DMD and self-forcing few-step schedules differ")
        reference = batch.clean_latents
        if not isinstance(reference, Tensor):
            raise TypeError("Self-Forcing latent template must be a torch.Tensor")
        initial_noise = _normal_like(reference, generator=generator)
        exit_indices = self.sample_exit_indices(reference, generator=generator)
        result = self.rollout(
            batch,
            initial_noise,
            exit_indices=exit_indices,
            generator=generator,
            training=training,
        )
        first = result.exit_indices[0]
        return FewStepPrediction(
            clean_latents=result.clean_latents,
            target_index=first,
            timestep=schedule.timesteps[first],
            sigma=schedule.sigmas[first],
        )

    def inference(
        self,
        batch: DMDTrainingBatch,
        initial_noise: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> SelfForcingRollout:
        """Run every denoising step through the exact rollout implementation."""

        exits = (len(self.config.schedule.sigmas) - 1,) * self._num_blocks(initial_noise)
        return self.rollout(
            batch,
            initial_noise,
            exit_indices=exits,
            generator=generator,
            training=False,
        )


__all__ = ["SelfForcingRolloutSampler"]
