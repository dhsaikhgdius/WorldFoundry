"""Bounded two-pass replay for native Self-Gradient-Forcing training."""

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
from ..self_forcing.contracts import CachePayload
from .config import SelfGradientForcingConfig
from .contracts import SelfGradientForcingAdapter, SelfGradientForcingReplay

SELF_GRADIENT_FORCING_RNG_STATE_SCHEMA = "worldfoundry-self-gradient-forcing-rng"


def _normal_like(reference: Tensor, *, generator: torch.Generator) -> Tensor:
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


def _detach_cache(value: CachePayload, *, path: str = "cache") -> CachePayload:
    if isinstance(value, Tensor):
        return value.detach()
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {key: _detach_cache(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_cache(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(_detach_cache(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{path} has unsupported cache payload type {type(value).__name__}")


def _audit_detached_cache(value: CachePayload, *, path: str = "cache") -> None:
    if isinstance(value, Tensor):
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError(f"{path} retained an autograd graph")
        return
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    items = value.items() if isinstance(value, Mapping) else enumerate(value)
    for key, item in items:
        _audit_detached_cache(item, path=f"{path}[{key!r}]")


class SelfGradientForcingSampler:
    """Own the official two-pass replay and all randomness it consumes."""

    def __init__(
        self,
        adapter: SelfGradientForcingAdapter,
        config: SelfGradientForcingConfig,
        *,
        parallel_context: PostTrainingParallelContext | None = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(adapter, SelfGradientForcingAdapter):
            raise TypeError("adapter must implement SelfGradientForcingAdapter")
        if not isinstance(config, SelfGradientForcingConfig):
            raise TypeError("config must be SelfGradientForcingConfig")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        try:
            parameter = next(adapter.module.parameters())
        except StopIteration as error:
            raise ValueError("Self-Gradient-Forcing adapter.module has no parameters") from error
        context = parallel_context or PostTrainingParallelContext.current()
        generator = torch.Generator(device=parameter.device)
        generator.manual_seed((seed + context.rank) % (2**63 - 1))
        self.adapter = adapter
        self.config = config
        self.parallel_context = context
        self._rng = generator
        self._rng_device = str(parameter.device)
        self.last_exit_index = -1
        self.rollout_count = 0

    @property
    def generator(self) -> torch.Generator:
        """The checkpointed generator shared with the enclosing DMD objective."""

        return self._rng

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
                f"latent frame count {frames} must be divisible by "
                f"frames_per_block={self.config.frames_per_block}"
            )
        return frames // self.config.frames_per_block

    def _adapter_cache_state(self, cache: CachePayload, *, fallback: CacheState) -> CacheState:
        state_fn = getattr(self.adapter, "cache_state", None)
        if state_fn is None:
            return fallback
        if not callable(state_fn):
            raise TypeError("Self-Gradient-Forcing adapter cache_state must be callable")
        state = state_fn(cache)
        if not isinstance(state, CacheState):
            raise TypeError("Self-Gradient-Forcing adapter cache_state must return CacheState")
        return state

    def sample_exit_index(self, reference: Tensor) -> int:
        """Draw one exit shared by every temporal block in the sequence."""

        if str(reference.device) != self._rng_device:
            raise ValueError("Self-Gradient-Forcing input device differs from its checkpointed RNG device")
        last = len(self.config.schedule.sigmas) - 1
        if self.config.last_step_only:
            return last
        if self.config.exit_step_rank_mode == "local":
            return int(
                torch.randint(
                    0,
                    len(self.config.schedule.sigmas),
                    (1,),
                    device=reference.device,
                    generator=self._rng,
                ).item()
            )
        if self.parallel_context.rank == 0:
            value = torch.randint(
                0,
                len(self.config.schedule.sigmas),
                (1,),
                device=reference.device,
                generator=self._rng,
            )
        else:
            value = torch.empty(1, device=reference.device, dtype=torch.long)
        if self.parallel_context.world_size > 1:
            group = self.parallel_context.process_group
            source = 0 if group is None else dist.get_global_rank(group, 0)
            dist.broadcast(value, src=source, group=group)
        result = int(value.item())
        if not 0 <= result <= last:
            raise RuntimeError("rank-synchronized Self-Gradient-Forcing exit index is invalid")
        return result

    def replay(
        self,
        batch: DMDTrainingBatch,
        initial_noise: Tensor,
        *,
        exit_index: int,
        training: bool,
    ) -> SelfGradientForcingReplay:
        """Run no-grad AR cache construction, then one parallel live forward."""

        if not isinstance(batch, DMDTrainingBatch):
            raise TypeError("batch must be DMDTrainingBatch")
        if not isinstance(initial_noise, Tensor) or initial_noise.shape != batch.clean_latents.shape:
            raise ValueError("initial_noise must match the batch latent shape template")
        if str(initial_noise.device) != self._rng_device:
            raise ValueError("initial_noise device differs from the checkpointed RNG device")
        if isinstance(exit_index, bool) or not 0 <= int(exit_index) < len(self.config.schedule.sigmas):
            raise ValueError("exit_index falls outside the few-step schedule")
        selected = int(exit_index)
        frame_dim = self._frame_dim(initial_noise)
        num_blocks = self._num_blocks(initial_noise)
        blocks = initial_noise.split(self.config.frames_per_block, dim=frame_dim)

        with torch.no_grad():
            cache = self.adapter.initialize_cache(
                initial_noise,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
            )
        cache = _detach_cache(cache)
        _audit_detached_cache(cache)
        frame_tokens = prod(int(size) for size in initial_noise.shape[frame_dim + 1 :]) or 1
        cache_state = self._adapter_cache_state(
            cache,
            fallback=CacheState(blocks=(), frame_tokens=frame_tokens, current_block_idx=-1),
        )

        exit_inputs: list[Tensor] = []
        cache_targets: list[Tensor] = []
        context_blocks: list[Tensor] = []
        start_frame = 0
        stop_at_exit = (
            self.config.cache_target_mode == "exit"
            and self.config.exit_step_rank_mode == "synchronized"
        )
        with torch.no_grad():
            for block_index, block_noise in enumerate(blocks):
                current = block_noise
                exit_clean: Tensor | None = None
                final_clean: Tensor | None = None
                for step_index, (timestep, sigma) in enumerate(
                    zip(self.config.schedule.timesteps, self.config.schedule.sigmas, strict=True)
                ):
                    if step_index == selected:
                        exit_inputs.append(current.detach().clone())
                    final_clean = self.adapter.predict_clean_chunk(
                        current,
                        _batch_scalar(current, timestep),
                        _batch_scalar(current, sigma),
                        block_index=block_index,
                        start_frame=start_frame,
                        sample_ids=batch.sample_ids,
                        conditioning=batch.conditioning,
                        cache=cache,
                        training=False,
                    )
                    if not isinstance(final_clean, Tensor) or final_clean.shape != current.shape:
                        raise ValueError("causal adapter must return a clean chunk matching its input")
                    if step_index == selected:
                        exit_clean = final_clean
                        if stop_at_exit:
                            break
                    if step_index + 1 < len(self.config.schedule.sigmas):
                        current = flow_interpolate(
                            final_clean,
                            _normal_like(final_clean, generator=self._rng),
                            _batch_scalar(final_clean, self.config.schedule.sigmas[step_index + 1]),
                        )
                if exit_clean is None or final_clean is None:
                    raise RuntimeError("Self-Gradient-Forcing failed to capture its exit state")
                cache_target = final_clean if self.config.cache_target_mode == "final-clean" else exit_clean
                cache_target = cache_target.detach()
                context_chunk = flow_interpolate(
                    cache_target,
                    _normal_like(cache_target, generator=self._rng),
                    _batch_scalar(cache_target, self.config.context_sigma),
                ).detach()
                context_step = _batch_scalar(context_chunk, self.config.context_timestep)
                cache = self.adapter.commit_context_chunk(
                    context_chunk,
                    context_step,
                    block_index=block_index,
                    start_frame=start_frame,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    cache=cache,
                )
                cache = _detach_cache(cache)
                _audit_detached_cache(cache)
                block_frames = int(context_chunk.shape[frame_dim])
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
                cache_targets.append(cache_target)
                context_blocks.append(context_chunk)
                start_frame += block_frames

        if len(exit_inputs) != num_blocks:
            raise RuntimeError("one Self-Gradient-Forcing exit input is required per temporal block")
        noisy_at_exit = torch.cat(exit_inputs, dim=frame_dim).detach()
        cached = torch.cat(cache_targets, dim=frame_dim).detach()
        context_latents = torch.cat(context_blocks, dim=frame_dim).detach()
        frames = int(initial_noise.shape[frame_dim])
        if self.config.match_context:
            pass2_context = context_latents
            context_timestep = self.config.context_timestep
        else:
            pass2_context = cached
            context_timestep = 0.0
        context_timesteps = torch.full(
            (int(initial_noise.shape[0]), frames),
            float(context_timestep),
            device=initial_noise.device,
            dtype=torch.float32,
        )
        timestep = self.config.schedule.timesteps[selected]
        sigma = self.config.schedule.sigmas[selected]
        with torch.set_grad_enabled(training):
            clean = self.adapter.predict_clean_teacher_forced(
                noisy_at_exit,
                _batch_scalar(noisy_at_exit, timestep),
                _batch_scalar(noisy_at_exit, sigma),
                clean_context=pass2_context.detach(),
                context_timesteps=context_timesteps,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                training=training,
            )
        if not isinstance(clean, Tensor) or clean.shape != initial_noise.shape:
            raise ValueError("teacher-forced adapter prediction must match the full latent shape")
        self.last_exit_index = selected
        self.rollout_count += 1
        return SelfGradientForcingReplay(
            clean_latents=clean,
            noisy_at_exit=noisy_at_exit,
            cache_targets=cached,
            context_latents=context_latents,
            exit_index=selected,
            timestep=timestep,
            sigma=sigma,
            cache_state=cache_state,
        )

    def sample(
        self,
        batch: DMDTrainingBatch,
        schedule: FewStepSchedule,
        *,
        generator: object | None,
        training: bool,
    ) -> FewStepPrediction:
        """DMD sampler seam backed exclusively by the checkpointed RNG."""

        if schedule != self.config.schedule:
            raise ValueError("DMD and Self-Gradient-Forcing schedules differ")
        if generator is not None and generator is not self._rng:
            raise ValueError("Self-Gradient-Forcing randomness must use its checkpointed generator")
        reference = batch.clean_latents
        if not isinstance(reference, Tensor):
            raise TypeError("Self-Gradient-Forcing latent template must be a torch.Tensor")
        initial_noise = _normal_like(reference, generator=self._rng)
        selected = self.sample_exit_index(reference)
        replay = self.replay(
            batch,
            initial_noise,
            exit_index=selected,
            training=training,
        )
        return FewStepPrediction(
            clean_latents=replay.clean_latents,
            target_index=replay.exit_index,
            timestep=replay.timestep,
            sigma=replay.sigma,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": SELF_GRADIENT_FORCING_RNG_STATE_SCHEMA,
            "data_parallel_size": self.parallel_context.world_size,
            "rng_device": self._rng_device,
            "rng_state": self._rng.get_state().clone(),
            "last_exit_index": self.last_exit_index,
            "rollout_count": self.rollout_count,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("Self-Gradient-Forcing RNG state must be a mapping")
        expected = {
            "schema",
            "data_parallel_size",
            "rng_device",
            "rng_state",
            "last_exit_index",
            "rollout_count",
        }
        if set(state_dict) != expected:
            raise ValueError("Self-Gradient-Forcing RNG state fields differ from the active schema")
        if state_dict["schema"] != SELF_GRADIENT_FORCING_RNG_STATE_SCHEMA:
            raise ValueError(f"unsupported Self-Gradient-Forcing RNG schema: {state_dict['schema']!r}")
        active = {
            "data_parallel_size": self.parallel_context.world_size,
            "rng_device": self._rng_device,
        }
        for name, value in active.items():
            if state_dict[name] != value:
                raise ValueError(f"saved Self-Gradient-Forcing {name} differs from the active sampler")
        count = state_dict["rollout_count"]
        last = state_dict["last_exit_index"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("saved Self-Gradient-Forcing rollout_count is invalid")
        if isinstance(last, bool) or not isinstance(last, int):
            raise ValueError("saved Self-Gradient-Forcing exit index is invalid")
        if count == 0:
            if last != -1:
                raise ValueError("an unused Self-Gradient-Forcing sampler cannot have an exit index")
        elif not 0 <= last < len(self.config.schedule.sigmas):
            raise ValueError("saved Self-Gradient-Forcing exit index is outside the schedule")
        rng_state = state_dict["rng_state"]
        if not isinstance(rng_state, Tensor) or rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
            raise ValueError("saved Self-Gradient-Forcing RNG state is invalid")
        self._rng.set_state(rng_state.detach().cpu())
        self.last_exit_index = last
        self.rollout_count = count


__all__ = [
    "SELF_GRADIENT_FORCING_RNG_STATE_SCHEMA",
    "SelfGradientForcingSampler",
]
