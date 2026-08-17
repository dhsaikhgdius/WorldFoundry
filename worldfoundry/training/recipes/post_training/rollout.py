"""Executable rollout placement for native post-training recipes."""

from __future__ import annotations

from dataclasses import dataclass

from .common import strict_mapping


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class LocalRolloutSpec:
    """Run rollout in the trainer process."""

    backend: str = "local"

    def __post_init__(self) -> None:
        if str(self.backend).strip().lower() != "local":
            raise ValueError("LocalRolloutSpec backend must be 'local'")
        object.__setattr__(self, "backend", "local")


@dataclass(frozen=True, slots=True)
class RayDevicePoolSpec:
    """Physical resources reserved by one Ray rollout runtime."""

    num_devices: int
    devices_per_node: int
    workers_per_device: int = 1
    cpus_per_worker: float = 1.0
    accelerator_resource: str = "GPU"
    ray_address: str | None = None

    def __post_init__(self) -> None:
        for name in ("num_devices", "devices_per_node", "workers_per_device"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), field_name=f"rollout.pool.{name}"),
            )
        cpus = float(self.cpus_per_worker)
        if cpus <= 0:
            raise ValueError("rollout.pool.cpus_per_worker must be positive")
        accelerator = str(self.accelerator_resource).strip()
        if not accelerator:
            raise ValueError("rollout.pool.accelerator_resource must be non-empty")
        address = None if self.ray_address is None else str(self.ray_address).strip()
        if address == "":
            address = None
        object.__setattr__(self, "cpus_per_worker", cpus)
        object.__setattr__(self, "accelerator_resource", accelerator)
        object.__setattr__(self, "ray_address", address)


@dataclass(frozen=True, slots=True)
class RayRolloutSpec:
    """Ray worker placement and trainer-to-rollout weight transfer."""

    pool: RayDevicePoolSpec
    rollout_devices: int
    trainer_devices: int | None = None
    trainer_binding: str = "external"
    placement: str = "separate"
    weight_kind: str = "full"
    weight_bucket_bytes: int = 256 * 1024 * 1024
    backend: str = "ray"

    def __post_init__(self) -> None:
        if not isinstance(self.pool, RayDevicePoolSpec):
            raise TypeError("rollout.pool must be RayDevicePoolSpec")
        if str(self.backend).strip().lower() != "ray":
            raise ValueError("RayRolloutSpec backend must be 'ray'")
        for name in ("rollout_devices", "weight_bucket_bytes"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), field_name=f"rollout.{name}"),
            )
        binding = str(self.trainer_binding).strip().lower().replace("_", "-")
        placement = str(self.placement).strip().lower().replace("_", "-")
        weight_kind = str(self.weight_kind).strip().lower().replace("_", "-")
        if binding not in {"external", "actor"}:
            raise ValueError("rollout.trainer_binding must be 'external' or 'actor'")
        if placement not in {"separate", "colocate"}:
            raise ValueError("rollout.placement must be 'separate' or 'colocate'")
        if binding == "external" and placement == "colocate":
            raise ValueError("colocate rollout placement requires an actor-hosted trainer")
        if binding == "external" and self.trainer_devices is not None:
            raise ValueError("external trainer binding does not reserve trainer_devices")
        if binding == "actor":
            if self.trainer_devices is None:
                raise ValueError("actor trainer binding requires positive trainer_devices")
            object.__setattr__(
                self,
                "trainer_devices",
                _positive_int(self.trainer_devices, field_name="rollout.trainer_devices"),
            )
        if placement == "colocate" and self.pool.workers_per_device < 2:
            raise ValueError("colocate rollout placement requires at least two workers_per_device slots")
        if placement == "colocate" and self.rollout_devices > int(self.trainer_devices):
            raise ValueError("colocated rollout_devices cannot exceed trainer_devices")
        if weight_kind not in {"full", "lora"}:
            raise ValueError("rollout.weight_kind must be 'full' or 'lora'")
        required_devices = (
            self.rollout_devices
            if binding == "external"
            else (
                max(int(self.trainer_devices), self.rollout_devices)
                if placement == "colocate"
                else int(self.trainer_devices) + self.rollout_devices
            )
        )
        if required_devices > self.pool.num_devices:
            raise ValueError("rollout.pool.num_devices cannot satisfy the requested trainer/rollout placement")
        object.__setattr__(self, "backend", "ray")
        object.__setattr__(self, "trainer_binding", binding)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "weight_kind", weight_kind)


RolloutSpec = LocalRolloutSpec | RayRolloutSpec


def parse_rollout_spec(value: object) -> RolloutSpec:
    """Parse a strict local or Ray rollout section."""

    root = strict_mapping(
        value,
        field_name="rollout",
        allowed={
            "backend",
            "pool",
            "trainer_devices",
            "rollout_devices",
            "trainer_binding",
            "placement",
            "weight_kind",
            "weight_bucket_bytes",
        },
    )
    backend = str(root.get("backend", "local")).strip().lower().replace("_", "-")
    if backend == "local":
        if set(root) - {"backend"}:
            raise ValueError("local rollout only accepts the backend field")
        return LocalRolloutSpec()
    if backend != "ray":
        raise ValueError("rollout.backend must be 'local' or 'ray'")
    missing = {"pool", "rollout_devices"} - set(root)
    if missing:
        raise ValueError(f"ray rollout is missing required fields: {sorted(missing)}")
    pool = strict_mapping(
        root["pool"],
        field_name="rollout.pool",
        allowed={
            "num_devices",
            "devices_per_node",
            "workers_per_device",
            "cpus_per_worker",
            "accelerator_resource",
            "ray_address",
        },
    )
    missing_pool = {"num_devices", "devices_per_node"} - set(pool)
    if missing_pool:
        raise ValueError(f"rollout.pool is missing required fields: {sorted(missing_pool)}")
    return RayRolloutSpec(
        pool=RayDevicePoolSpec(**pool),
        rollout_devices=root["rollout_devices"],
        trainer_devices=root.get("trainer_devices"),
        trainer_binding=root.get("trainer_binding", "external"),
        placement=root.get("placement", "separate"),
        weight_kind=root.get("weight_kind", "full"),
        weight_bucket_bytes=root.get("weight_bucket_bytes", 256 * 1024 * 1024),
    )


__all__ = [
    "LocalRolloutSpec",
    "RayDevicePoolSpec",
    "RayRolloutSpec",
    "RolloutSpec",
    "parse_rollout_spec",
]
