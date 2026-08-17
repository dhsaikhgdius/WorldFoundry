"""Ray rollout workers for native stochastic flow-policy training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch
from torch import nn

from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (
    NativeFlowPolicyTrainingStack,
)
from worldfoundry.training.post_training.rl.contracts import (
    FlowRolloutBatch,
    FlowTrajectory,
)
from worldfoundry.training.post_training.rl.rollout_strategies.transition import (
    flow_transition_strategy_from_identity,
)
from worldfoundry.training.post_training.rl.trajectory import FlowTrajectorySampler
from worldfoundry.training.post_training.shared.contracts import FlowPredictionAdapter
from worldfoundry.training.post_training.shared.partitioning import (
    balanced_contiguous_partitions,
)

from .ray_runtime import RayWorkerContext
from .rollout_runtime import RayPostTrainingRuntime
from .weight_sync import ModuleWeightReceiver, WeightKind, WeightSyncReport


@dataclass(frozen=True, slots=True)
class RayFlowSamplerConfig:
    """Serializable construction fields for one actor-local sampler."""

    transition_identity: Mapping[str, object]
    trajectory_dtype: torch.dtype | None = None
    forward_batch_size: int | None = None

    def __post_init__(self) -> None:
        strategy = flow_transition_strategy_from_identity(self.transition_identity)
        object.__setattr__(self, "transition_identity", MappingProxyType(dict(strategy.identity)))
        if self.forward_batch_size is not None and int(self.forward_batch_size) <= 0:
            raise ValueError("forward_batch_size must be positive")

    @classmethod
    def from_sampler(cls, sampler: object) -> RayFlowSamplerConfig:
        transition_strategy = getattr(sampler, "transition_strategy", None)
        trajectory_dtype = getattr(sampler, "trajectory_dtype", None)
        forward_batch_size = getattr(sampler, "forward_batch_size", None)
        if transition_strategy is None or not hasattr(transition_strategy, "identity"):
            raise TypeError("Ray rollout attachment requires a sampler with a transition strategy")
        return cls(
            transition_identity=dict(transition_strategy.identity),
            trajectory_dtype=trajectory_dtype,
            forward_batch_size=forward_batch_size,
        )


@dataclass(frozen=True, slots=True)
class FlowTrajectoryShardRequest:
    """One complete-group rollout shard sent to a Ray actor."""

    positions: tuple[int, ...]
    batch: FlowRolloutBatch
    sde_step_indices: tuple[int, ...] | None
    generator_seed: int | None


@dataclass(frozen=True, slots=True)
class FlowTrajectoryShardResult:
    """Actor result paired with its positions in the trainer batch."""

    positions: tuple[int, ...]
    trajectory: FlowTrajectory


def partition_complete_flow_groups(
    group_ids: Sequence[str],
    partition_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition ordered prompt groups without splitting any group."""

    count = int(partition_count)
    if count <= 0:
        raise ValueError("partition_count must be positive")
    groups: dict[str, list[int]] = {}
    for position, group_id in enumerate(group_ids):
        groups.setdefault(str(group_id), []).append(position)
    if not groups:
        raise ValueError("group_ids cannot be empty")
    ordered = tuple(tuple(positions) for positions in groups.values())
    partitions = balanced_contiguous_partitions(len(ordered), min(count, len(ordered)))
    return tuple(tuple(position for group in ordered[start:end] for position in group) for start, end in partitions)


def _module_device(module: nn.Module) -> torch.device:
    tensor = next(iter(module.parameters()), None)
    if tensor is None:
        tensor = next(iter(module.buffers()), None)
    return torch.device("cpu") if tensor is None else tensor.device


def _move_tensors(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {str(key): _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    return value


def _select_batched_value(
    value: object,
    positions: tuple[int, ...],
    *,
    batch_size: int,
) -> object:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and int(value.shape[0]) == batch_size:
            indices = torch.tensor(positions, device=value.device, dtype=torch.long)
            return value.index_select(0, indices)
        return value
    if isinstance(value, Mapping):
        return {str(key): _select_batched_value(item, positions, batch_size=batch_size) for key, item in value.items()}
    return value


def _shard_rollout_batch(
    batch: FlowRolloutBatch,
    positions: tuple[int, ...],
) -> FlowRolloutBatch:
    indices = torch.tensor(positions, device=batch.initial_latents.device, dtype=torch.long)
    sigmas = batch.sigmas
    if sigmas.ndim == 2:
        sigma_indices = indices.to(device=sigmas.device)
        sigmas = sigmas.index_select(0, sigma_indices)
    return FlowRolloutBatch(
        sample_ids=tuple(batch.sample_ids[index] for index in positions),
        group_ids=tuple(batch.group_ids[index] for index in positions),
        policy_revision=batch.policy_revision,
        initial_latents=batch.initial_latents.index_select(0, indices),
        sigmas=sigmas,
        conditioning={
            key: _select_batched_value(value, positions, batch_size=batch.batch_size)
            for key, value in batch.conditioning.items()
        },
    )


def _cpu_trajectory(
    trajectory: FlowTrajectory,
    *,
    request_conditioning_keys: frozenset[str],
) -> FlowTrajectory:
    def cpu(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu")
        return value

    return FlowTrajectory(
        sample_ids=trajectory.sample_ids,
        group_ids=trajectory.group_ids,
        policy_revision=trajectory.policy_revision,
        latents=cpu(trajectory.latents),
        sigmas=cpu(trajectory.sigmas),
        step_indices=trajectory.step_indices,
        old_log_probs=cpu(trajectory.old_log_probs),
        transition_means=cpu(trajectory.transition_means),
        transition_scales=cpu(trajectory.transition_scales),
        update_step_mask=(None if trajectory.update_step_mask is None else cpu(trajectory.update_step_mask)),
        conditioning={
            key: _move_tensors(value, torch.device("cpu"))
            for key, value in trajectory.conditioning.items()
            if key not in request_conditioning_keys
        },
        transition_identity=dict(trajectory.transition_identity),
        metadata={},
    )


class RayFlowTrajectoryWorker:
    """Actor-local policy, weight receiver, and real trajectory sampler."""

    def __init__(
        self,
        *,
        context: RayWorkerContext,
        policy_factory: Callable[..., FlowPredictionAdapter],
        sampler_config: RayFlowSamplerConfig,
        policy_factory_kwargs: Mapping[str, object] | None = None,
        sampler_factory: Callable[..., object] | None = None,
        sampler_factory_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(sampler_config, RayFlowSamplerConfig):
            raise TypeError("sampler_config must be RayFlowSamplerConfig")
        policy = policy_factory(context=context, **dict(policy_factory_kwargs or {}))
        if not isinstance(policy, FlowPredictionAdapter):
            raise TypeError("policy_factory must return a FlowPredictionAdapter")
        if not isinstance(policy.module, nn.Module):
            raise TypeError("rollout policy module must be an nn.Module")
        self.context = context
        self.rollout_module = policy.module
        self.weight_receiver = ModuleWeightReceiver(self.rollout_module)
        self.device = _module_device(self.rollout_module)
        transition_strategy = flow_transition_strategy_from_identity(sampler_config.transition_identity)
        if sampler_factory is None:
            sampler = FlowTrajectorySampler(
                policy,
                transition_strategy=transition_strategy,
                trajectory_dtype=sampler_config.trajectory_dtype,
                forward_batch_size=sampler_config.forward_batch_size,
            )
        else:
            sampler = sampler_factory(
                policy,
                transition_strategy=transition_strategy,
                trajectory_dtype=sampler_config.trajectory_dtype,
                forward_batch_size=sampler_config.forward_batch_size,
                **dict(sampler_factory_kwargs or {}),
            )
        sample = getattr(sampler, "sample", None)
        if not callable(sample):
            raise TypeError("sampler_factory must return a trajectory sampler")
        self.sampler = sampler
        self.active_policy_revision: str | None = None

    def activate_policy_revision(self, policy_revision: str, weight_revision: int) -> None:
        if self.weight_receiver.last_revision != int(weight_revision):
            raise ValueError("actor weights do not match the activated revision")
        self.active_policy_revision = str(policy_revision)

    def sample_shard(self, request: FlowTrajectoryShardRequest) -> FlowTrajectoryShardResult:
        if not isinstance(request, FlowTrajectoryShardRequest):
            raise TypeError("request must be FlowTrajectoryShardRequest")
        if request.batch.policy_revision != self.active_policy_revision:
            raise ValueError("rollout request differs from the actor policy revision")
        batch = request.batch
        generator = None
        if request.generator_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(request.generator_seed)
        trajectory = self.sampler.sample(
            batch.initial_latents.to(device=self.device),
            batch.sigmas.to(device=self.device),
            sample_ids=batch.sample_ids,
            group_ids=batch.group_ids,
            conditioning={key: _move_tensors(value, self.device) for key, value in batch.conditioning.items()},
            policy_revision=batch.policy_revision,
            sde_step_indices=request.sde_step_indices,
            generator=generator,
        )
        return FlowTrajectoryShardResult(
            positions=request.positions,
            trajectory=_cpu_trajectory(
                trajectory,
                request_conditioning_keys=frozenset(batch.conditioning),
            ),
        )


def _generator_seeds(
    generator: torch.Generator | None,
    count: int,
) -> tuple[int | None, ...]:
    if generator is None:
        return (None,) * count
    device = torch.device(generator.device)
    return tuple(
        int(
            torch.randint(
                0,
                2**63 - 1,
                (),
                device=device,
                generator=generator,
                dtype=torch.int64,
            ).item()
        )
        for _ in range(count)
    )


def _merge_trajectory_shards(
    results: Sequence[FlowTrajectoryShardResult],
    batch: FlowRolloutBatch,
) -> FlowTrajectory:
    trajectories = tuple(result.trajectory for result in results)
    positions = tuple(position for result in results for position in result.positions)
    source_index = {position: index for index, position in enumerate(positions)}
    if len(positions) != batch.batch_size or set(source_index) != set(range(batch.batch_size)):
        raise ValueError("Ray rollout shards do not cover the trainer batch exactly once")
    order = torch.tensor(
        [source_index[position] for position in range(batch.batch_size)],
        dtype=torch.long,
    )
    device = batch.initial_latents.device

    def merge(name: str) -> torch.Tensor:
        values = tuple(getattr(trajectory, name) for trajectory in trajectories)
        merged = torch.cat(values, dim=0)
        return merged.index_select(0, order.to(device=merged.device)).to(device=device)

    conditioning = dict(batch.conditioning)
    extra_conditioning_keys = set().union(*(set(trajectory.conditioning) for trajectory in trajectories))
    for key in sorted(extra_conditioning_keys):
        values = tuple(trajectory.conditioning.get(key) for trajectory in trajectories)
        if any(value is None for value in values):
            raise ValueError(f"Ray rollout workers returned inconsistent trajectory conditioning {key!r}")
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"Ray rollout trajectory conditioning {key!r} must be a tensor")
        merged = torch.cat(values, dim=0)
        conditioning[key] = merged.index_select(
            0,
            order.to(device=merged.device),
        ).to(device=device)

    first = trajectories[0]
    for trajectory in trajectories[1:]:
        if (
            trajectory.step_indices != first.step_indices
            or dict(trajectory.transition_identity) != dict(first.transition_identity)
            or trajectory.policy_revision != first.policy_revision
        ):
            raise ValueError("Ray rollout workers returned incompatible trajectory shards")
    if batch.sigmas.ndim == 1:
        sigmas = first.sigmas.to(device=device)
    else:
        sigmas = merge("sigmas")
    masks = tuple(trajectory.update_step_mask for trajectory in trajectories)
    update_step_mask = None
    if any(mask is not None for mask in masks):
        if not all(isinstance(mask, torch.Tensor) for mask in masks):
            raise ValueError("Ray rollout workers returned inconsistent update masks")
        update_step_mask = merge("update_step_mask")
    return FlowTrajectory(
        sample_ids=batch.sample_ids,
        group_ids=batch.group_ids,
        policy_revision=batch.policy_revision,
        latents=merge("latents"),
        sigmas=sigmas,
        step_indices=first.step_indices,
        old_log_probs=merge("old_log_probs"),
        transition_means=merge("transition_means"),
        transition_scales=merge("transition_scales"),
        update_step_mask=update_step_mask,
        conditioning=conditioning,
        transition_identity=dict(first.transition_identity),
        metadata=batch.metadata,
    )


class RayFlowTrajectorySampler:
    """Trainer-side sampler proxy with revision-aware weight synchronization."""

    def __init__(
        self,
        runtime: RayPostTrainingRuntime,
        source_module: nn.Module,
        sampler_config: RayFlowSamplerConfig,
        *,
        weight_kind: WeightKind | str = WeightKind.FULL,
    ) -> None:
        if runtime.rollout_group is None:
            raise RuntimeError("RayPostTrainingRuntime must have rollout workers")
        if not isinstance(source_module, nn.Module):
            raise TypeError("source_module must be an nn.Module")
        self.runtime = runtime
        self.module = source_module
        self.sampler_config = sampler_config
        self.transition_strategy = flow_transition_strategy_from_identity(sampler_config.transition_identity)
        self.eta = self.transition_strategy.eta
        self.sigma_max = getattr(self.transition_strategy, "sigma_max", None)
        self.trajectory_dtype = sampler_config.trajectory_dtype
        self.forward_batch_size = sampler_config.forward_batch_size
        self.weight_kind = WeightKind(str(weight_kind).strip().lower())
        self.active_policy_revision: str | None = None
        self.weight_revision = -1
        self.last_sync_report: WeightSyncReport | None = None

    def _activate_revision(self, policy_revision: str) -> None:
        if policy_revision == self.active_policy_revision:
            return
        next_revision = self.weight_revision + 1
        report = self.runtime.sync_rollout_weights(
            self.module,
            revision=next_revision,
            kind=self.weight_kind,
        )
        self.weight_revision = next_revision
        self.last_sync_report = report
        assert self.runtime.rollout_group is not None
        self.runtime.rollout_group.broadcast(
            "activate_policy_revision",
            policy_revision,
            next_revision,
        )
        self.active_policy_revision = policy_revision

    def sample(
        self,
        initial_latents: object,
        sigmas: object,
        *,
        sample_ids: tuple[str, ...],
        group_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        policy_revision: str,
        sde_step_indices: tuple[int, ...] | None = None,
        generator: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> FlowTrajectory:
        if not isinstance(initial_latents, torch.Tensor) or not isinstance(sigmas, torch.Tensor):
            raise TypeError("Ray flow rollout requires tensor initial_latents and sigmas")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        batch = FlowRolloutBatch(
            sample_ids=sample_ids,
            group_ids=group_ids,
            policy_revision=policy_revision,
            initial_latents=initial_latents,
            sigmas=sigmas,
            conditioning=conditioning,
            metadata={} if metadata is None else metadata,
        )
        self._activate_revision(policy_revision)
        assert self.runtime.rollout_group is not None
        partitions = partition_complete_flow_groups(
            batch.group_ids,
            self.runtime.rollout_group.lease.world_size,
        )
        seeds = _generator_seeds(generator, len(partitions))
        requests = tuple(
            FlowTrajectoryShardRequest(
                positions=positions,
                batch=_shard_rollout_batch(batch, positions),
                sde_step_indices=sde_step_indices,
                generator_seed=seed,
            )
            for positions, seed in zip(partitions, seeds, strict=True)
        )
        results = self.runtime.rollout_group.map("sample_shard", requests)
        if not all(isinstance(result, FlowTrajectoryShardResult) for result in results):
            raise TypeError("Ray rollout worker returned an invalid trajectory shard")
        return _merge_trajectory_shards(results, batch)


def attach_ray_flow_policy_rollout(
    stack: NativeFlowPolicyTrainingStack,
    runtime: RayPostTrainingRuntime,
    *,
    rollout_policy_factory: Callable[..., FlowPredictionAdapter],
    rollout_policy_factory_kwargs: Mapping[str, object] | None = None,
    rollout_sampler_factory: Callable[..., object] | None = None,
    rollout_sampler_factory_kwargs: Mapping[str, object] | None = None,
    source_module: nn.Module | None = None,
    weight_kind: WeightKind | str = WeightKind.FULL,
    reward_factory: Callable[..., object] | None = None,
    reward_factory_kwargs: Mapping[str, object] | None = None,
) -> NativeFlowPolicyTrainingStack:
    """Start real rollout actors and replace a native stack's local sampler.

    The returned sampler keeps the runtime alive; the caller closes that
    runtime after the training session finishes.
    """

    if not isinstance(stack, NativeFlowPolicyTrainingStack):
        raise TypeError("stack must be NativeFlowPolicyTrainingStack")
    sampler_config = RayFlowSamplerConfig.from_sampler(stack.sampler)
    try:
        runtime.setup(
            RayFlowTrajectoryWorker,
            rollout_factory_kwargs={
                "policy_factory": rollout_policy_factory,
                "policy_factory_kwargs": dict(rollout_policy_factory_kwargs or {}),
                "sampler_config": sampler_config,
                "sampler_factory": rollout_sampler_factory,
                "sampler_factory_kwargs": dict(rollout_sampler_factory_kwargs or {}),
            },
            reward_factory=reward_factory,
            reward_factory_kwargs=reward_factory_kwargs,
        )
        sampler = RayFlowTrajectorySampler(
            runtime,
            stack.replay.module if source_module is None else source_module,
            sampler_config,
            weight_kind=weight_kind,
        )
    except Exception:
        runtime.shutdown()
        raise
    return replace(stack, sampler=sampler)


__all__ = [
    "FlowTrajectoryShardRequest",
    "FlowTrajectoryShardResult",
    "RayFlowSamplerConfig",
    "RayFlowTrajectorySampler",
    "RayFlowTrajectoryWorker",
    "attach_ray_flow_policy_rollout",
    "partition_complete_flow_groups",
]
