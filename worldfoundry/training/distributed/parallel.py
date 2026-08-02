"""Dynamic named DeviceMesh planning and process-group ownership."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from types import TracebackType

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.recipes.spec import DistributedSpec

PARALLEL_PLAN_SCHEMA = "worldfoundry-training-parallel-plan"
_MESH_DIM_NAMES = ("dp_replicate", "dp_shard", "cp", "tp")


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _environment_int(name: str, *, default: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        if default is None:
            raise RuntimeError(f"distributed launch is missing {name}")
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise RuntimeError(f"distributed launch variable {name} must be an integer") from error


@dataclass(frozen=True, slots=True)
class ParallelPlan:
    """One fully resolved four-dimensional training topology."""

    backend: str
    world_size: int
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    schema: str = PARALLEL_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PARALLEL_PLAN_SCHEMA:
            raise ValueError(f"unsupported parallel plan schema: {self.schema!r}")
        backend = str(self.backend).strip().lower().replace("_", "-")
        if backend not in {"single", "ddp", "fsdp2"}:
            raise ValueError(f"unsupported parallel backend: {backend!r}")
        object.__setattr__(self, "backend", backend)
        for name in ("world_size", "dp_replicate", "dp_shard", "cp", "tp"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), field_name=name),
            )
        if self.dp_replicate * self.dp_shard * self.cp * self.tp != self.world_size:
            raise ValueError(
                "parallel dimensions must multiply to world_size: "
                f"{self.dp_replicate}*{self.dp_shard}*{self.cp}*{self.tp} "
                f"!= {self.world_size}"
            )
        if self.backend == "single" and self.mesh_shape != (1, 1, 1, 1):
            raise ValueError("single backend requires every parallel dimension to equal one")
        if self.backend == "ddp" and self.dp_shard != 1:
            raise ValueError("DDP requires dp_shard=1")

    @classmethod
    def resolve(cls, spec: DistributedSpec, *, world_size: int) -> ParallelPlan:
        if not isinstance(spec, DistributedSpec):
            raise TypeError("spec must be DistributedSpec")
        resolved_world_size = _positive_int(world_size, field_name="world_size")
        if spec.backend == "single":
            if resolved_world_size != 1:
                raise ValueError("single backend requires world_size=1")
            return cls(
                backend="single",
                world_size=1,
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=1,
            )
        if spec.backend == "ddp":
            if spec.dp_shard not in {"auto", 1}:
                raise ValueError("DDP distributed.dp_shard must be auto or one")
            fixed = spec.dp_replicate * spec.cp * spec.tp
            if fixed != resolved_world_size:
                raise ValueError("DDP dp_replicate*cp*tp must equal the launched world size")
            return cls(
                backend="ddp",
                world_size=resolved_world_size,
                dp_replicate=spec.dp_replicate,
                dp_shard=1,
                cp=spec.cp,
                tp=spec.tp,
            )

        fixed = spec.dp_replicate * spec.cp * spec.tp
        if isinstance(spec.dp_shard, str):
            if resolved_world_size % fixed:
                raise ValueError("world_size is not divisible by dp_replicate*cp*tp for automatic dp_shard")
            dp_shard = resolved_world_size // fixed
        else:
            dp_shard = spec.dp_shard
        return cls(
            backend="fsdp2",
            world_size=resolved_world_size,
            dp_replicate=spec.dp_replicate,
            dp_shard=dp_shard,
            cp=spec.cp,
            tp=spec.tp,
        )

    @property
    def mesh_shape(self) -> tuple[int, int, int, int]:
        return self.dp_replicate, self.dp_shard, self.cp, self.tp

    @property
    def data_parallel_size(self) -> int:
        return self.dp_replicate * self.dp_shard

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "backend": self.backend,
            "world_size": self.world_size,
            "mesh_dim_names": list(_MESH_DIM_NAMES),
            "mesh_shape": list(self.mesh_shape),
            "dp_replicate": self.dp_replicate,
            "dp_shard": self.dp_shard,
            "cp": self.cp,
            "tp": self.tp,
        }

    def build_device_mesh(self, device_type: str) -> DeviceMesh:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("building a DeviceMesh requires an initialized process group")
        if dist.get_world_size() != self.world_size:
            raise RuntimeError("active process-group world size differs from the parallel plan")
        resolved_device_type = str(device_type).strip().lower()
        if resolved_device_type not in {"cpu", "cuda"}:
            raise ValueError("training DeviceMesh currently supports cpu or cuda")
        if resolved_device_type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA DeviceMesh requested without CUDA availability")
        return init_device_mesh(
            resolved_device_type,
            self.mesh_shape,
            mesh_dim_names=_MESH_DIM_NAMES,
        )

    def fsdp_mesh(self, mesh: DeviceMesh) -> DeviceMesh:
        if self.backend != "fsdp2":
            raise ValueError("an FSDP mesh requires backend='fsdp2'")
        if self.dp_replicate > 1:
            return mesh["dp_replicate", "dp_shard"]
        return mesh["dp_shard"]


class DistributedTrainingContext:
    """Own or validate the process group created by a torchrun launch."""

    def __init__(
        self,
        *,
        device_type: str = "cuda",
        timeout_seconds: int = 1800,
    ) -> None:
        resolved_device_type = str(device_type).strip().lower()
        if resolved_device_type not in {"cpu", "cuda"}:
            raise ValueError("distributed training device_type must be cpu or cuda")
        timeout = _positive_int(timeout_seconds, field_name="timeout_seconds")
        rank = _environment_int("RANK", default=0)
        world_size = _environment_int("WORLD_SIZE", default=1)
        local_rank = _environment_int("LOCAL_RANK", default=rank)
        if world_size <= 0 or not 0 <= rank < world_size:
            raise RuntimeError(f"invalid distributed rank/world size: {rank}/{world_size}")
        if local_rank < 0:
            raise RuntimeError("LOCAL_RANK must be non-negative")
        if resolved_device_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA distributed context requested without CUDA")
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(f"LOCAL_RANK {local_rank} exceeds visible CUDA devices {torch.cuda.device_count()}")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "cpu:gloo,cuda:nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        self._owns_process_group = False
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != rank or dist.get_world_size() != world_size:
                raise RuntimeError("active process group differs from torchrun environment")
        else:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=timeout),
            )
            self._owns_process_group = True
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        dist.barrier()

    def close(self) -> None:
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_process_group = False

    def __enter__(self) -> DistributedTrainingContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


__all__ = [
    "PARALLEL_PLAN_SCHEMA",
    "DistributedTrainingContext",
    "ParallelPlan",
]
