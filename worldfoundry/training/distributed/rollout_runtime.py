"""Composed Ray rollout placement and trainer-to-worker weight updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import TracebackType

from torch import nn

from .ray_runtime import (
    DeviceLease,
    RayDevicePool,
    RayDevicePoolConfig,
    RayWorkerGroup,
    RolloutPlacement,
)
from .weight_sync import NativeWeightSynchronizer, WeightKind, WeightSyncReport


class TrainerBinding(StrEnum):
    """Whether training runs outside Ray or inside a Ray worker group."""

    EXTERNAL = "external"
    ACTOR = "actor"


@dataclass(frozen=True, slots=True)
class RayPostTrainingRuntimeConfig:
    """Device counts for a trainer and its rollout/reward worker groups."""

    pool: RayDevicePoolConfig
    rollout_devices: int
    trainer_devices: int = 0
    rollout_placement: RolloutPlacement = RolloutPlacement.SEPARATE
    reward_devices: int = 0
    reward_placement: RolloutPlacement = RolloutPlacement.SEPARATE
    weight_bucket_bytes: int = 256 * 1024 * 1024
    trainer_binding: TrainerBinding = TrainerBinding.EXTERNAL

    def __post_init__(self) -> None:
        if self.rollout_devices <= 0:
            raise ValueError("rollout_devices must be positive")
        if self.reward_devices < 0:
            raise ValueError("reward_devices must be non-negative")
        if self.weight_bucket_bytes <= 0:
            raise ValueError("weight_bucket_bytes must be positive")
        object.__setattr__(self, "rollout_placement", RolloutPlacement(self.rollout_placement))
        object.__setattr__(self, "reward_placement", RolloutPlacement(self.reward_placement))
        object.__setattr__(self, "trainer_binding", TrainerBinding(self.trainer_binding))
        if self.trainer_binding is TrainerBinding.ACTOR and self.trainer_devices <= 0:
            raise ValueError("actor trainer binding requires positive trainer_devices")
        if self.trainer_binding is TrainerBinding.EXTERNAL and self.trainer_devices != 0:
            raise ValueError("external trainer binding does not reserve trainer_devices")
        if (
            self.trainer_binding is TrainerBinding.ACTOR
            and self.rollout_placement is RolloutPlacement.COLOCATE
            and self.rollout_devices > self.trainer_devices
        ):
            raise ValueError("colocated rollout_devices cannot exceed trainer_devices")


def ray_runtime_config_from_rollout_spec(spec: object) -> RayPostTrainingRuntimeConfig:
    """Translate the pure-data Ray recipe into executable runtime placement."""

    from worldfoundry.training.recipes.post_training.rollout import RayRolloutSpec

    if not isinstance(spec, RayRolloutSpec):
        raise TypeError("Ray runtime materialization requires RayRolloutSpec")
    return RayPostTrainingRuntimeConfig(
        pool=RayDevicePoolConfig(
            num_devices=spec.pool.num_devices,
            devices_per_node=spec.pool.devices_per_node,
            workers_per_device=spec.pool.workers_per_device,
            cpus_per_worker=spec.pool.cpus_per_worker,
            accelerator_resource=spec.pool.accelerator_resource,
            ray_address=spec.pool.ray_address,
        ),
        trainer_devices=0 if spec.trainer_devices is None else spec.trainer_devices,
        rollout_devices=spec.rollout_devices,
        rollout_placement=RolloutPlacement(spec.placement),
        weight_bucket_bytes=spec.weight_bucket_bytes,
        trainer_binding=TrainerBinding(spec.trainer_binding),
    )


class RayPostTrainingRuntime:
    """Own actor placement and trainer-to-rollout weight synchronization.

    ``external`` binds rollout actors to a controller-local or otherwise
    caller-owned trainer. Ray dedicates the rollout devices, while the caller is
    responsible for ensuring that those devices are physically separate from the
    trainer. Physical separate/colocate placement is runtime-owned only when the
    trainer itself is actor hosted.
    """

    def __init__(self, config: RayPostTrainingRuntimeConfig) -> None:
        self.config = config
        self.pool = RayDevicePool(config.pool)
        self.synchronizer = NativeWeightSynchronizer(
            max_bucket_bytes=config.weight_bucket_bytes,
        )
        self.trainer_lease: DeviceLease | None = None
        self.trainer_group: RayWorkerGroup | None = None
        self.rollout_group: RayWorkerGroup | None = None
        self.reward_group: RayWorkerGroup | None = None

    def setup(
        self,
        rollout_factory: Callable[..., object],
        *,
        rollout_factory_kwargs: Mapping[str, object] | None = None,
        trainer_factory: Callable[..., object] | None = None,
        trainer_factory_kwargs: Mapping[str, object] | None = None,
        reward_factory: Callable[..., object] | None = None,
        reward_factory_kwargs: Mapping[str, object] | None = None,
    ) -> RayPostTrainingRuntime:
        if self.config.reward_devices and reward_factory is None:
            raise ValueError("reward_devices requires a reward_factory")
        actor_hosted = self.config.trainer_binding is TrainerBinding.ACTOR
        if actor_hosted and trainer_factory is None:
            raise ValueError("actor trainer binding requires a trainer_factory")
        if not actor_hosted and trainer_factory is not None:
            raise ValueError("trainer_factory requires actor trainer binding")
        if not actor_hosted and (
            self.config.rollout_placement is RolloutPlacement.COLOCATE
            or (self.config.reward_devices and self.config.reward_placement is RolloutPlacement.COLOCATE)
        ):
            raise ValueError("colocate placement requires an actor-hosted trainer")
        if self.rollout_group is not None or self.trainer_group is not None:
            raise RuntimeError("Ray post-training runtime is already set up")
        try:
            self.pool.setup()
            if actor_hosted:
                trainer = self.pool.reserve("trainer", self.config.trainer_devices)
                assert trainer_factory is not None
                self.trainer_lease = trainer
                self.trainer_group = self.pool.create_worker_group(
                    trainer,
                    trainer_factory,
                    factory_kwargs=trainer_factory_kwargs,
                )
            rollout = self.pool.reserve(
                "rollout",
                self.config.rollout_devices,
                placement=self.config.rollout_placement,
                colocate_with=("trainer" if self.config.rollout_placement is RolloutPlacement.COLOCATE else None),
            )
            self.rollout_group = self.pool.create_worker_group(
                rollout,
                rollout_factory,
                factory_kwargs=rollout_factory_kwargs,
            )
            if self.config.reward_devices:
                reward = self.pool.reserve(
                    "reward",
                    self.config.reward_devices,
                    placement=self.config.reward_placement,
                    colocate_with=("trainer" if self.config.reward_placement is RolloutPlacement.COLOCATE else None),
                )
                assert reward_factory is not None
                self.reward_group = self.pool.create_worker_group(
                    reward,
                    reward_factory,
                    factory_kwargs=reward_factory_kwargs,
                )
            return self
        except Exception:
            self.shutdown()
            raise

    def sync_rollout_weights(
        self,
        module: nn.Module | None = None,
        *,
        revision: int,
        kind: WeightKind | str = WeightKind.FULL,
    ) -> WeightSyncReport:
        if self.rollout_group is None:
            raise RuntimeError("rollout workers have not been created")
        if self.config.trainer_binding is TrainerBinding.ACTOR:
            if module is not None:
                raise ValueError("actor trainer binding synchronizes from its trainer worker group")
            return self._sync_actor_trainer_weights(revision=revision, kind=kind)
        if module is None:
            raise ValueError("external trainer binding requires a source module")
        report = self.synchronizer.sync(
            module,
            [self.rollout_group],
            revision=revision,
            kind=kind,
        )
        return replace(report, receiver_count=len(self.rollout_group.actors))

    def _sync_actor_trainer_weights(
        self,
        *,
        revision: int,
        kind: WeightKind | str,
    ) -> WeightSyncReport:
        trainer_group = self.trainer_group
        rollout_group = self.rollout_group
        if trainer_group is None or rollout_group is None:
            raise RuntimeError("trainer and rollout worker groups must both be active")
        header, byte_count = trainer_group.prepare_weight_update(
            revision,
            kind,
            self.config.weight_bucket_bytes,
        )
        begun = False
        try:
            rollout_group.begin_weight_update(header)
            begun = True
            for index in range(header.bucket_count):
                bucket_ref = trainer_group.weight_bucket_ref(revision, index)
                rollout_group.write_weight_bucket_ref(bucket_ref)
            rollout_group.validate_weight_update(revision)
            rollout_group.commit_weight_update(revision)
        except Exception:
            if begun:
                rollout_group.abort_weight_update(revision)
            raise
        finally:
            trainer_group.clear_weight_update(revision)
        return WeightSyncReport(
            revision=revision,
            kind=header.kind,
            tensor_count=len(header.tensor_names),
            byte_count=byte_count,
            bucket_count=header.bucket_count,
            receiver_count=len(rollout_group.actors),
            transmitted=True,
        )

    def shutdown(self) -> None:
        self.pool.shutdown()
        self.trainer_lease = None
        self.trainer_group = None
        self.rollout_group = None
        self.reward_group = None

    def close(self) -> None:
        """Release every Ray worker and placement group owned by the runtime."""

        self.shutdown()

    def __enter__(self) -> RayPostTrainingRuntime:
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
    "RayPostTrainingRuntime",
    "RayPostTrainingRuntimeConfig",
    "TrainerBinding",
    "ray_runtime_config_from_rollout_spec",
]
