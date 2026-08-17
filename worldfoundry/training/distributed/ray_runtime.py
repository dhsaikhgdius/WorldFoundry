"""Ray-backed device placement and rollout worker groups.

The planner is dependency-free so recipes can be checked before Ray starts.
``RayDevicePool`` owns the placement groups and actors used by dedicated or
GPU-colocated rollout execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from itertools import cycle
from types import TracebackType
from typing import Callable, Iterable, Mapping

import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.tensor import DTensor

from .weight_sync import (
    ModuleWeightReceiver,
    WeightBucket,
    WeightKind,
    WeightUpdateHeader,
    build_weight_buckets,
    materialize_weight_tensors,
)

logger = logging.getLogger(__name__)


class RolloutPlacement(StrEnum):
    """Where rollout workers live relative to a training worker group."""

    SEPARATE = "separate"
    COLOCATE = "colocate"


@dataclass(frozen=True, slots=True)
class DeviceLease:
    """A named set of physical devices and one process slot on each device."""

    role: str
    device_ids: tuple[int, ...]
    slot: int
    placement: RolloutPlacement
    colocated_with: str | None = None

    @property
    def world_size(self) -> int:
        return len(self.device_ids)


@dataclass(frozen=True, slots=True)
class RayDevicePoolConfig:
    """Physical resource shape owned by one training run."""

    num_devices: int
    devices_per_node: int
    workers_per_device: int = 2
    cpus_per_worker: float = 1.0
    accelerator_resource: str = "GPU"
    ray_address: str | None = None

    def __post_init__(self) -> None:
        if self.num_devices <= 0 or self.devices_per_node <= 0:
            raise ValueError("num_devices and devices_per_node must be positive")
        if self.workers_per_device <= 0:
            raise ValueError("workers_per_device must be positive")
        if self.cpus_per_worker <= 0:
            raise ValueError("cpus_per_worker must be positive")
        resource = str(self.accelerator_resource).strip()
        if not resource:
            raise ValueError("accelerator_resource must be non-empty")
        object.__setattr__(self, "accelerator_resource", resource)


class DevicePoolPlanner:
    """Allocate disjoint slabs or additional slots on an existing slab."""

    def __init__(self, config: RayDevicePoolConfig) -> None:
        if not isinstance(config, RayDevicePoolConfig):
            raise TypeError("config must be RayDevicePoolConfig")
        self.config = config
        self._leases: dict[str, DeviceLease] = {}
        self._claimed_devices: set[int] = set()
        self._used_slots: dict[int, set[int]] = {device_id: set() for device_id in range(config.num_devices)}

    @property
    def leases(self) -> Mapping[str, DeviceLease]:
        return dict(self._leases)

    def reserve(
        self,
        role: str,
        count: int,
        *,
        placement: RolloutPlacement | str = RolloutPlacement.SEPARATE,
        colocate_with: str | None = None,
    ) -> DeviceLease:
        name = str(role).strip()
        if not name or name in self._leases:
            raise ValueError(f"device role is empty or already reserved: {role!r}")
        requested = int(count)
        if requested <= 0:
            raise ValueError("device count must be positive")
        mode = RolloutPlacement(str(placement).strip().lower())

        if mode is RolloutPlacement.SEPARATE:
            available = tuple(
                device_id for device_id in range(self.config.num_devices) if device_id not in self._claimed_devices
            )
            if requested > len(available):
                raise ValueError(f"role {name!r} needs {requested} devices but only {len(available)} remain")
            device_ids = available[:requested]
            slot = 0
            self._claimed_devices.update(device_ids)
        else:
            if colocate_with is None or colocate_with not in self._leases:
                raise ValueError("colocated roles require an existing colocate_with role")
            base = self._leases[colocate_with]
            if requested > base.world_size:
                raise ValueError("colocated role cannot exceed the base role's device count")
            device_ids = base.device_ids[:requested]
            common_slots = set(range(self.config.workers_per_device))
            for device_id in device_ids:
                common_slots.difference_update(self._used_slots[device_id])
            if not common_slots:
                raise ValueError("no common worker slot remains on the requested devices")
            slot = min(common_slots)

        lease = DeviceLease(
            role=name,
            device_ids=tuple(device_ids),
            slot=slot,
            placement=mode,
            colocated_with=colocate_with,
        )
        for device_id in lease.device_ids:
            self._used_slots[device_id].add(slot)
        self._leases[name] = lease
        return lease

    def release(self, role: str) -> None:
        lease = self._leases.pop(role)
        for device_id in lease.device_ids:
            self._used_slots[device_id].remove(lease.slot)
        if lease.placement is RolloutPlacement.SEPARATE:
            self._claimed_devices.difference_update(lease.device_ids)


@dataclass(frozen=True, slots=True)
class RayWorkerContext:
    """Stable placement information passed to a rollout role factory."""

    role: str
    rank: int
    world_size: int
    device_id: int
    slot: int
    placement_group_index: int = 0
    bundle_index: int = 0


class RayRoleWorker:
    """One Ray actor process hosting a caller-owned rollout role."""

    def __init__(
        self,
        factory: Callable[..., object],
        context: RayWorkerContext,
        factory_kwargs: Mapping[str, object] | None = None,
        *,
        pass_context: bool = True,
    ) -> None:
        self.context = context
        kwargs = dict(factory_kwargs or {})
        if pass_context:
            kwargs["context"] = context
        self.role = factory(**kwargs)
        self._weight_receiver: object | None = None
        self._prepared_weight_header: WeightUpdateHeader | None = None
        self._prepared_weight_buckets: tuple[WeightBucket, ...] = ()

    def invoke(self, method: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> object:
        target = getattr(self.role, str(method))
        return target(*args, **dict(kwargs))

    def begin_weight_update(self, header: object) -> None:
        self._receiver().begin_weight_update(header)

    def write_weight_bucket(self, bucket: object) -> None:
        self._receiver().write_weight_bucket(bucket)

    def validate_weight_update(self, revision: int) -> None:
        self._receiver().validate_weight_update(revision)

    def commit_weight_update(self, revision: int) -> None:
        self._receiver().commit_weight_update(revision)

    def abort_weight_update(self, revision: int) -> None:
        self._receiver().abort_weight_update(revision)

    def _receiver(self) -> object:
        if self._weight_receiver is not None:
            return self._weight_receiver
        configured = getattr(self.role, "weight_receiver", None)
        if configured is not None:
            self._weight_receiver = configured
            return configured
        candidates = (
            getattr(self.role, "rollout_module", None),
            getattr(self.role, "model", None),
            self.role,
        )
        module = next((value for value in candidates if isinstance(value, nn.Module)), None)
        if module is None:
            raise TypeError("rollout role must expose weight_receiver, rollout_module, or model")
        self._weight_receiver = ModuleWeightReceiver(module)
        return self._weight_receiver

    def _weight_source(self) -> nn.Module:
        configured = getattr(self.role, "weight_source", None)
        if configured is not None:
            configured = configured() if not isinstance(configured, nn.Module) else configured
        candidates = (
            configured,
            getattr(self.role, "trainer_module", None),
            getattr(self.role, "model", None),
            self.role,
        )
        module = next((value for value in candidates if isinstance(value, nn.Module)), None)
        if module is None:
            raise TypeError("trainer role must expose weight_source, trainer_module, or model")
        if isinstance(module, FullyShardedDataParallel) or any(
            isinstance(value, DTensor) for value in (*tuple(module.parameters()), *tuple(module.buffers()))
        ):
            raise TypeError(
                "actor-hosted weight sync requires a replicated module; "
                "sharded modules need collective materialization on every trainer rank"
            )
        return module

    def prepare_weight_update(
        self,
        revision: int,
        kind: WeightKind | str,
        max_bucket_bytes: int,
    ) -> tuple[WeightUpdateHeader, int]:
        """Stage replicated trainer weights inside the source actor."""

        if self._prepared_weight_header is not None:
            raise RuntimeError("another outbound weight update is already active")
        resolved_kind = WeightKind(str(kind).strip().lower())
        source_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        tensors = materialize_weight_tensors(
            self._weight_source(),
            kind=resolved_kind,
            source_rank=source_rank,
        )
        buckets = build_weight_buckets(
            tensors,
            revision=int(revision),
            max_bucket_bytes=int(max_bucket_bytes),
        )
        header = WeightUpdateHeader(
            revision=int(revision),
            kind=resolved_kind,
            tensor_names=tuple(sorted(tensors)),
            bucket_count=len(buckets),
        )
        self._prepared_weight_header = header
        self._prepared_weight_buckets = buckets
        byte_count = sum(value.numel() * value.element_size() for value in tensors.values())
        return header, byte_count

    def read_weight_bucket(self, revision: int, index: int) -> WeightBucket:
        header = self._prepared_weight_header
        if header is None or header.revision != int(revision):
            raise ValueError("no matching outbound weight update is active")
        return self._prepared_weight_buckets[int(index)]

    def clear_weight_update(self, revision: int) -> None:
        header = self._prepared_weight_header
        if header is not None and header.revision == int(revision):
            self._prepared_weight_header = None
            self._prepared_weight_buckets = ()

    def close(self) -> None:
        close = getattr(self.role, "close", None)
        if callable(close):
            close()


class RayWorkerGroup:
    """Controller-side handle for one SPMD rollout role."""

    def __init__(self, ray_module: object, lease: DeviceLease, actors: tuple[object, ...]) -> None:
        self._ray = ray_module
        self.lease = lease
        self.actors = actors
        self._round_robin = cycle(actors)

    def broadcast(self, method: str, *args: object, **kwargs: object) -> tuple[object, ...]:
        refs = [actor.invoke.remote(method, args, kwargs) for actor in self.actors]
        return tuple(self._ray.get(refs))

    def submit(self, method: str, *args: object, **kwargs: object) -> object:
        """Submit one asynchronous call to the next data-parallel worker."""

        actor = next(self._round_robin)
        return actor.invoke.remote(method, args, kwargs)

    def gather(self, refs: Iterable[object]) -> tuple[object, ...]:
        return tuple(self._ray.get(list(refs)))

    def map(self, method: str, items: Iterable[object]) -> tuple[object, ...]:
        refs = [self.submit(method, item) for item in items]
        return self.gather(refs)

    def map_batches(
        self,
        method: str,
        items: Iterable[object],
        *,
        batch_size: int,
    ) -> tuple[object, ...]:
        """Send bounded batches round-robin and restore their original order."""

        size = int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        values = tuple(items)
        batches = tuple(values[index : index + size] for index in range(0, len(values), size))
        results = self.gather(self.submit(method, batch) for batch in batches)
        flattened: list[object] = []
        for batch, result in zip(batches, results):
            output = tuple(result)
            if len(output) != len(batch):
                raise ValueError("worker batch output count differs from its input count")
            flattened.extend(output)
        return tuple(flattened)

    def begin_weight_update(self, header: object) -> None:
        self._ray.get([actor.begin_weight_update.remote(header) for actor in self.actors])

    def write_weight_bucket_ref(self, bucket_ref: object) -> None:
        # Ray dereferences ObjectRef arguments automatically, so this one entry
        # point serves both direct buckets and shared object-store references.
        self._ray.get([actor.write_weight_bucket.remote(bucket_ref) for actor in self.actors])

    def validate_weight_update(self, revision: int) -> None:
        self._ray.get([actor.validate_weight_update.remote(revision) for actor in self.actors])

    def commit_weight_update(self, revision: int) -> None:
        self._ray.get([actor.commit_weight_update.remote(revision) for actor in self.actors])

    def abort_weight_update(self, revision: int) -> None:
        self._ray.get([actor.abort_weight_update.remote(revision) for actor in self.actors])

    def prepare_weight_update(
        self,
        revision: int,
        kind: WeightKind | str,
        max_bucket_bytes: int,
    ) -> tuple[WeightUpdateHeader, int]:
        return self._ray.get(
            self.actors[0].prepare_weight_update.remote(
                revision,
                kind,
                max_bucket_bytes,
            )
        )

    def weight_bucket_ref(self, revision: int, index: int) -> object:
        return self.actors[0].read_weight_bucket.remote(revision, index)

    def clear_weight_update(self, revision: int) -> None:
        self._ray.get(self.actors[0].clear_weight_update.remote(revision))


class RayDevicePool:
    """Own Ray placement groups and rollout workers for one training run."""

    def __init__(self, config: RayDevicePoolConfig) -> None:
        self.config = config
        self.planner = DevicePoolPlanner(config)
        self._ray: object | None = None
        self._placement_groups: tuple[object, ...] = ()
        self._groups: dict[str, RayWorkerGroup] = {}
        self._started_ray = False

    def setup(self) -> None:
        if self._ray is not None:
            return
        ray = import_module("ray")
        if not ray.is_initialized():
            ray.init(address=self.config.ray_address, ignore_reinit_error=True)
            self._started_ray = True
        placement_group = import_module("ray.util.placement_group").placement_group

        groups: list[object] = []
        remaining = self.config.num_devices
        while remaining:
            node_devices = min(self.config.devices_per_node, remaining)
            bundle = {"CPU": self.config.cpus_per_worker * self.config.workers_per_device}
            if self.config.accelerator_resource != "CPU":
                bundle[self.config.accelerator_resource] = 1
            group = placement_group([dict(bundle) for _ in range(node_devices)], strategy="STRICT_PACK")
            groups.append(group)
            remaining -= node_devices
        ray.get([group.ready() for group in groups])
        self._ray = ray
        self._placement_groups = tuple(groups)

    def reserve(
        self,
        role: str,
        count: int,
        *,
        placement: RolloutPlacement | str = RolloutPlacement.SEPARATE,
        colocate_with: str | None = None,
    ) -> DeviceLease:
        return self.planner.reserve(
            role,
            count,
            placement=placement,
            colocate_with=colocate_with,
        )

    def create_worker_group(
        self,
        lease: DeviceLease,
        factory: Callable[..., object],
        *,
        factory_kwargs: Mapping[str, object] | None = None,
        pass_context: bool = True,
        max_concurrency: int = 1,
    ) -> RayWorkerGroup:
        if self._ray is None:
            self.setup()
        if lease.role in self._groups:
            raise ValueError(f"worker group already exists for role {lease.role!r}")
        ray = self._ray
        strategy_type = import_module("ray.util.scheduling_strategies").PlacementGroupSchedulingStrategy
        remote_worker = ray.remote(RayRoleWorker)
        actors: list[object] = []
        for rank, device_id in enumerate(lease.device_ids):
            group_index, bundle_index = divmod(device_id, self.config.devices_per_node)
            resources: dict[str, object] = {
                "num_cpus": self.config.cpus_per_worker,
                "max_concurrency": max(1, int(max_concurrency)),
                "scheduling_strategy": strategy_type(
                    placement_group=self._placement_groups[group_index],
                    placement_group_bundle_index=bundle_index,
                    placement_group_capture_child_tasks=True,
                ),
            }
            fraction = 1.0 / self.config.workers_per_device
            if self.config.accelerator_resource == "GPU":
                resources["num_gpus"] = fraction
            elif self.config.accelerator_resource != "CPU":
                resources["resources"] = {self.config.accelerator_resource: fraction}
            context = RayWorkerContext(
                role=lease.role,
                rank=rank,
                world_size=lease.world_size,
                device_id=device_id,
                slot=lease.slot,
                placement_group_index=group_index,
                bundle_index=bundle_index,
            )
            actor = remote_worker.options(**resources).remote(
                factory,
                context,
                dict(factory_kwargs or {}),
                pass_context=pass_context,
            )
            actors.append(actor)
        result = RayWorkerGroup(ray, lease, tuple(actors))
        self._groups[lease.role] = result
        return result

    def worker_group(self, role: str) -> RayWorkerGroup:
        return self._groups[role]

    def shutdown(self) -> None:
        if self._ray is None:
            return
        ray = self._ray
        close_refs = [actor.close.remote() for group in self._groups.values() for actor in group.actors]
        if close_refs:
            try:
                ray.get(close_refs)
            except Exception:
                # Shutdown stays best-effort (actors are killed next), but a
                # failed role close may hide lost rollout state; keep evidence.
                logger.warning(
                    "rollout worker close() failed during Ray shutdown; killing actors anyway",
                    exc_info=True,
                )
        for group in self._groups.values():
            for actor in group.actors:
                ray.kill(actor, no_restart=True)
        remove_placement_group = import_module("ray.util.placement_group").remove_placement_group
        for group in self._placement_groups:
            remove_placement_group(group)
        self._groups.clear()
        self._placement_groups = ()
        self._ray = None
        self.planner = DevicePoolPlanner(self.config)
        if self._started_ray:
            ray.shutdown()
            self._started_ray = False

    def __enter__(self) -> RayDevicePool:
        self.setup()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.shutdown()


__all__ = [
    "DeviceLease",
    "DevicePoolPlanner",
    "RayDevicePool",
    "RayDevicePoolConfig",
    "RayRoleWorker",
    "RayWorkerContext",
    "RayWorkerGroup",
    "RolloutPlacement",
]
