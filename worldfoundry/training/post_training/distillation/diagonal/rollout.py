"""Causal 4/3/2 rollout for native diagonal distillation."""

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
from ..dmd.objective import FewStepSchedule
from ..self_forcing.contracts import CachePayload
from .config import DiagonalScheduleConfig
from .contracts import DiagonalCausalAdapter, DiagonalFewStepPrediction, DiagonalRollout

DIAGONAL_SAMPLER_STATE_SCHEMA = "worldfoundry-diagonal-sampler"


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


def _frame_dim(reference: Tensor, config: DiagonalScheduleConfig) -> int:
    resolved = config.frame_dim % reference.ndim
    if resolved == 0:
        raise ValueError("resolved frame_dim cannot be the batch dimension")
    return resolved


def _num_blocks(reference: Tensor, config: DiagonalScheduleConfig) -> int:
    frame_dim = _frame_dim(reference, config)
    frames = int(reference.shape[frame_dim])
    if frames == 0 or frames % config.frames_per_block:
        raise ValueError(
            f"latent frame count {frames} must be divisible by frames_per_block={config.frames_per_block}"
        )
    return frames // config.frames_per_block


def _adapter_cache_state(
    adapter: DiagonalCausalAdapter,
    cache: CachePayload,
    *,
    fallback: CacheState,
) -> CacheState:
    state_fn = getattr(adapter, "cache_state", None)
    if state_fn is None:
        return fallback
    if not callable(state_fn):
        raise TypeError("diagonal adapter cache_state must be callable")
    state = state_fn(cache)
    if not isinstance(state, CacheState):
        raise TypeError("diagonal adapter cache_state must return CacheState")
    return state


def _sample_synchronized_exits(
    reference: Tensor,
    config: DiagonalScheduleConfig,
    *,
    generator: torch.Generator,
    parallel_context: PostTrainingParallelContext,
    force_last: bool,
) -> tuple[int, ...]:
    num_blocks = _num_blocks(reference, config)
    schedule_size = len(config.base_schedule.timesteps)
    if parallel_context.rank == 0:
        values = torch.randint(
            0,
            schedule_size,
            (num_blocks,),
            device=reference.device,
            generator=generator,
        )
        if force_last or config.last_step_only:
            values.fill_(schedule_size - 1)
    else:
        values = torch.empty(num_blocks, device=reference.device, dtype=torch.long)
    if parallel_context.world_size > 1:
        group = parallel_context.process_group
        source = 0 if group is None else dist.get_global_rank(group, 0)
        dist.broadcast(values, src=source, group=group)
    exits = tuple(int(value) for value in values.tolist())
    if any(not 0 <= value < schedule_size for value in exits):
        raise RuntimeError("rank-synchronized diagonal exit index is invalid")
    return (exits[0],) * num_blocks if config.exit_step_mode == "sequence" else exits


def _execute_rollout(
    adapter: DiagonalCausalAdapter,
    config: DiagonalScheduleConfig,
    batch: DMDTrainingBatch,
    initial_noise: Tensor,
    *,
    base_exit_indices: tuple[int, ...],
    generator: torch.Generator,
    training: bool,
    complete_block_schedules: bool = False,
) -> DiagonalRollout:
    if not isinstance(batch, DMDTrainingBatch):
        raise TypeError("batch must be DMDTrainingBatch")
    if not isinstance(initial_noise, Tensor) or initial_noise.shape != batch.clean_latents.shape:
        raise ValueError("initial_noise must match the batch latent shape template")
    if not isinstance(complete_block_schedules, bool):
        raise TypeError("complete_block_schedules must be bool")
    num_blocks = _num_blocks(initial_noise, config)
    if len(base_exit_indices) != num_blocks:
        raise ValueError("base_exit_indices must contain one entry per temporal block")
    base_size = len(config.base_schedule.timesteps)
    if any(
        isinstance(index, bool) or not 0 <= int(index) < base_size
        for index in base_exit_indices
    ):
        raise ValueError("a base exit index falls outside the configured schedule")

    frame_dim = _frame_dim(initial_noise, config)
    with torch.no_grad():
        cache = adapter.initialize_cache(
            initial_noise,
            sample_ids=batch.sample_ids,
            conditioning=batch.conditioning,
        )
    cache = _detach_cache(cache)
    _audit_detached_cache(cache)
    frame_tokens = prod(int(size) for size in initial_noise.shape[frame_dim + 1 :]) or 1
    cache_state = _adapter_cache_state(
        adapter,
        cache,
        fallback=CacheState(blocks=(), frame_tokens=frame_tokens, current_block_idx=-1),
    )

    outputs: list[Tensor] = []
    masks: list[Tensor] = []
    clipped_exits: list[int] = []
    block_timesteps: list[tuple[float, ...]] = []
    blocks = initial_noise.split(config.frames_per_block, dim=frame_dim)
    start_frame = 0
    for block_index, (block_noise, raw_exit) in enumerate(
        zip(blocks, base_exit_indices, strict=True)
    ):
        block_schedule = config.block_schedule(block_index)
        selected = (
            len(block_schedule.timesteps) - 1
            if complete_block_schedules
            else min(int(raw_exit), len(block_schedule.timesteps) - 1)
        )
        current = block_noise
        predicted_clean: Tensor | None = None
        for step_index in range(selected + 1):
            differentiable = bool(training and step_index == selected)
            with torch.set_grad_enabled(differentiable):
                predicted_clean = adapter.predict_clean_chunk(
                    current,
                    _batch_scalar(current, block_schedule.timesteps[step_index]),
                    _batch_scalar(current, block_schedule.sigmas[step_index]),
                    block_index=block_index,
                    start_frame=start_frame,
                    sample_ids=batch.sample_ids,
                    conditioning=batch.conditioning,
                    cache=cache,
                    training=differentiable,
                )
            if not isinstance(predicted_clean, Tensor) or predicted_clean.shape != current.shape:
                raise ValueError("diagonal adapter must return a clean chunk matching its input")
            if step_index < selected:
                with torch.no_grad():
                    current = flow_interpolate(
                        predicted_clean,
                        _normal_like(predicted_clean, generator=generator),
                        _batch_scalar(predicted_clean, block_schedule.sigmas[step_index + 1]),
                    )
        assert predicted_clean is not None
        outputs.append(predicted_clean)
        masks.append(torch.full_like(predicted_clean, training, dtype=torch.bool))

        clean_context = predicted_clean.detach()
        with torch.no_grad():
            context_chunk = flow_interpolate(
                clean_context,
                _normal_like(clean_context, generator=generator),
                _batch_scalar(clean_context, config.context_sigma),
            ).detach()
            cache = adapter.commit_context_chunk(
                context_chunk,
                _batch_scalar(context_chunk, config.context_timestep),
                block_index=block_index,
                start_frame=start_frame,
                sample_ids=batch.sample_ids,
                conditioning=batch.conditioning,
                cache=cache,
            )
        cache = _detach_cache(cache)
        _audit_detached_cache(cache)
        block_frames = int(predicted_clean.shape[frame_dim])
        cache_state = _adapter_cache_state(
            adapter,
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
        clipped_exits.append(selected)
        block_timesteps.append(block_schedule.timesteps)
        start_frame += block_frames

    return DiagonalRollout(
        clean_latents=torch.cat(outputs, dim=frame_dim),
        gradient_mask=torch.cat(masks, dim=frame_dim),
        base_exit_indices=tuple(int(value) for value in base_exit_indices),
        block_exit_indices=tuple(clipped_exits),
        block_timesteps=tuple(block_timesteps),
        cache_state=cache_state,
    )


class DiagonalRolloutSampler:
    """Own rank-synchronized exits, re-noising, and checkpointed RNG state."""

    def __init__(
        self,
        adapter: DiagonalCausalAdapter,
        config: DiagonalScheduleConfig,
        *,
        parallel_context: PostTrainingParallelContext | None = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(adapter, DiagonalCausalAdapter):
            raise TypeError("adapter must implement DiagonalCausalAdapter")
        if not isinstance(config, DiagonalScheduleConfig):
            raise TypeError("config must be DiagonalScheduleConfig")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        try:
            parameter = next(adapter.module.parameters())
        except StopIteration as error:
            raise ValueError("diagonal adapter.module has no parameters") from error
        context = parallel_context or PostTrainingParallelContext.current()
        rng = torch.Generator(device=parameter.device)
        rng.manual_seed((seed + context.rank) % (2**63 - 1))
        self.adapter = adapter
        self.config = config
        self.parallel_context = context
        self._rng = rng
        self._rng_device = str(parameter.device)
        self.last_base_exit_indices: tuple[int, ...] = ()
        self.last_block_exit_indices: tuple[int, ...] = ()
        self.rollout_count = 0

    @property
    def generator(self) -> torch.Generator:
        return self._rng

    def _require_generator(self, value: object | None) -> torch.Generator:
        if value is not None and value is not self._rng:
            raise ValueError("diagonal sampler randomness must use its checkpointed generator")
        return self._rng

    def sample_base_exit_indices(self, reference: Tensor) -> tuple[int, ...]:
        if not isinstance(reference, Tensor):
            raise TypeError("diagonal latent template must be a torch.Tensor")
        if str(reference.device) != self._rng_device:
            raise ValueError("diagonal input device differs from its checkpointed RNG device")
        return _sample_synchronized_exits(
            reference,
            self.config,
            generator=self._rng,
            parallel_context=self.parallel_context,
            force_last=False,
        )

    def rollout(
        self,
        batch: DMDTrainingBatch,
        initial_noise: Tensor,
        *,
        base_exit_indices: tuple[int, ...],
        generator: torch.Generator | None = None,
        training: bool,
        complete_block_schedules: bool = False,
    ) -> DiagonalRollout:
        rng = self._require_generator(generator)
        if str(initial_noise.device) != self._rng_device:
            raise ValueError("initial_noise device differs from the checkpointed RNG device")
        result = _execute_rollout(
            self.adapter,
            self.config,
            batch,
            initial_noise,
            base_exit_indices=base_exit_indices,
            generator=rng,
            training=training,
            complete_block_schedules=complete_block_schedules,
        )
        self.last_base_exit_indices = result.base_exit_indices
        self.last_block_exit_indices = result.block_exit_indices
        self.rollout_count += 1
        return result

    def sample(
        self,
        batch: DMDTrainingBatch,
        schedule: FewStepSchedule,
        *,
        generator: object | None,
        training: bool,
    ) -> DiagonalFewStepPrediction:
        if schedule != self.config.base_schedule:
            raise ValueError("DMD and diagonal base schedules differ")
        rng = self._require_generator(generator)
        reference = batch.clean_latents
        if not isinstance(reference, Tensor):
            raise TypeError("diagonal latent template must be a torch.Tensor")
        if str(reference.device) != self._rng_device:
            raise ValueError("diagonal input device differs from its checkpointed RNG device")
        initial_noise = _normal_like(reference, generator=rng)
        exits = self.sample_base_exit_indices(reference)
        result = self.rollout(
            batch,
            initial_noise,
            base_exit_indices=exits,
            generator=rng,
            training=training,
        )
        first_exit = result.block_exit_indices[0]
        first_schedule = self.config.block_schedule(0)
        return DiagonalFewStepPrediction(
            clean_latents=result.clean_latents,
            target_index=result.base_exit_indices[0],
            timestep=first_schedule.timesteps[first_exit],
            sigma=first_schedule.sigmas[first_exit],
            rollout=result,
        )

    def inference(
        self,
        batch: DMDTrainingBatch,
        initial_noise: Tensor,
    ) -> DiagonalRollout:
        exits = (len(self.config.base_schedule.timesteps) - 1,) * _num_blocks(
            initial_noise,
            self.config,
        )
        return self.rollout(
            batch,
            initial_noise,
            base_exit_indices=exits,
            training=False,
            complete_block_schedules=True,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": DIAGONAL_SAMPLER_STATE_SCHEMA,
            "data_parallel_size": self.parallel_context.world_size,
            "rng_device": self._rng_device,
            "rng_state": self._rng.get_state().clone(),
            "last_base_exit_indices": self.last_base_exit_indices,
            "last_block_exit_indices": self.last_block_exit_indices,
            "rollout_count": self.rollout_count,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("diagonal sampler state must be a mapping")
        expected = {
            "schema",
            "data_parallel_size",
            "rng_device",
            "rng_state",
            "last_base_exit_indices",
            "last_block_exit_indices",
            "rollout_count",
        }
        if set(state_dict) != expected:
            raise ValueError("diagonal sampler state fields differ from the active schema")
        if state_dict["schema"] != DIAGONAL_SAMPLER_STATE_SCHEMA:
            raise ValueError(f"unsupported diagonal sampler schema: {state_dict['schema']!r}")
        active = {
            "data_parallel_size": self.parallel_context.world_size,
            "rng_device": self._rng_device,
        }
        for name, value in active.items():
            if state_dict[name] != value:
                raise ValueError(f"saved diagonal {name} differs from the active sampler")
        count = state_dict["rollout_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("saved diagonal rollout_count is invalid")
        base = state_dict["last_base_exit_indices"]
        clipped = state_dict["last_block_exit_indices"]
        if not isinstance(base, tuple) or not isinstance(clipped, tuple):
            raise TypeError("saved diagonal exit indices must be tuples")
        if count == 0:
            if base or clipped:
                raise ValueError("an unused diagonal sampler cannot have exit indices")
        elif not base or len(base) != len(clipped):
            raise ValueError("saved diagonal exit inventories are invalid")
        schedule_size = len(self.config.base_schedule.timesteps)
        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < schedule_size for value in base):
            raise ValueError("saved diagonal base exit falls outside the schedule")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in clipped):
            raise ValueError("saved diagonal clipped exit is invalid")
        rng_state = state_dict["rng_state"]
        if not isinstance(rng_state, Tensor) or rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
            raise ValueError("saved diagonal RNG state is invalid")
        self._rng.set_state(rng_state.detach().cpu())
        self.last_base_exit_indices = base
        self.last_block_exit_indices = clipped
        self.rollout_count = count


class DiagonalFixedTeacherSampler:
    """Frozen four-step target sampler sharing the student's restored RNG."""

    def __init__(
        self,
        adapter: DiagonalCausalAdapter,
        config: DiagonalScheduleConfig,
        *,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(adapter, DiagonalCausalAdapter):
            raise TypeError("adapter must implement DiagonalCausalAdapter")
        if not isinstance(config, DiagonalScheduleConfig):
            raise TypeError("config must be DiagonalScheduleConfig")
        if not config.last_step_only:
            raise ValueError("fixed diagonal teacher must use last_step_only")
        if any(parameter.requires_grad for parameter in adapter.module.parameters()):
            raise ValueError("fixed diagonal teacher parameters must be frozen")
        adapter.module.eval()
        self.adapter = adapter
        self.config = config
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()

    def sample(
        self,
        batch: DMDTrainingBatch,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        if not isinstance(generator, torch.Generator):
            raise TypeError("fixed diagonal teacher requires an explicit torch.Generator")
        reference = batch.clean_latents
        if not isinstance(reference, Tensor):
            raise TypeError("fixed diagonal teacher latent template must be a torch.Tensor")
        generator_device = str(getattr(generator, "device", reference.device))
        if generator_device != str(reference.device):
            raise ValueError("fixed diagonal teacher and restored generator devices differ")
        initial_noise = _normal_like(reference, generator=generator)
        exits = _sample_synchronized_exits(
            reference,
            self.config,
            generator=generator,
            parallel_context=self.parallel_context,
            force_last=True,
        )
        result = _execute_rollout(
            self.adapter,
            self.config,
            batch,
            initial_noise,
            base_exit_indices=exits,
            generator=generator,
            training=False,
        )
        return result.clean_latents.detach()


__all__ = [
    "DIAGONAL_SAMPLER_STATE_SCHEMA",
    "DiagonalFixedTeacherSampler",
    "DiagonalRolloutSampler",
]
