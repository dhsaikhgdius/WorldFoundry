"""Audited, family-declared FSDP2 application for native training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from worldfoundry.training.api.contracts import TrainModelAdapter

from .parallel import ParallelPlan

FSDP2_APPLICATION_SCHEMA = "worldfoundry-training-fsdp2-application"


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"FSDP2 mixed precision does not support dtype {dtype}")
    return str(dtype).removeprefix("torch.")


def _parameter_names(module: nn.Module, *, trainable_only: bool) -> tuple[str, ...]:
    return tuple(name for name, parameter in module.named_parameters() if not trainable_only or parameter.requires_grad)


def _module_device_audit(module: nn.Module, expected: torch.device) -> None:
    mismatches: list[str] = []
    for name, value in (*module.named_parameters(), *module.named_buffers()):
        if value.device != expected:
            mismatches.append(f"{name}={value.device}")
            if len(mismatches) == 5:
                break
    if mismatches:
        raise ValueError(
            "FSDP2 requires the trainable module to be materialized on the current CUDA "
            f"device {expected}; mismatches: {mismatches}"
        )


def _precision_island_modules(
    module: nn.Module,
    *,
    param_dtype: torch.dtype,
    master_parameter_dtype: torch.dtype | None,
) -> dict[torch.dtype, list[tuple[str, nn.Module]]]:
    islands: dict[torch.dtype, list[tuple[str, nn.Module]]] = {}
    for name, child in module.named_modules():
        direct_parameters = tuple(child.parameters(recurse=False))
        non_compute_parameters = tuple(parameter for parameter in direct_parameters if parameter.dtype != param_dtype)
        if not non_compute_parameters:
            continue
        if master_parameter_dtype is not None and all(
            parameter.dtype == master_parameter_dtype for parameter in non_compute_parameters
        ):
            continue
        direct_dtypes = {parameter.dtype for parameter in direct_parameters}
        if len(direct_dtypes) != 1:
            raise ValueError(
                f"FSDP2 precision-island module {name!r} has mixed direct parameter dtypes: "
                f"{sorted(map(str, direct_dtypes))}"
            )
        direct_ids = {id(parameter) for parameter in direct_parameters}
        descendant_parameters = tuple(parameter for parameter in child.parameters() if id(parameter) not in direct_ids)
        if descendant_parameters:
            raise ValueError(
                f"FSDP2 precision-island module {name!r} owns parameterized descendants; "
                "declare a leaf-level model policy instead of guessing a mixed communication group"
            )
        dtype = next(iter(direct_dtypes))
        islands.setdefault(dtype, []).append((name, child))
    return islands


@dataclass(frozen=True, slots=True)
class FSDP2Application:
    """Serializable audit of one in-place FSDP2 transformation."""

    parallel_plan: ParallelPlan
    block_module_names: tuple[str, ...]
    block_class_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    parameter_count: int
    trainable_parameter_count: int
    param_dtype: str
    reduce_dtype: str
    root_reshard_after_forward: bool
    parameter_mode: str
    precision_island_module_names: tuple[str, ...] = ()
    original_parameter_dtypes: tuple[str, ...] = ()
    schema: str = FSDP2_APPLICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FSDP2_APPLICATION_SCHEMA:
            raise ValueError(f"unsupported FSDP2 application schema: {self.schema!r}")
        if self.parallel_plan.backend != "fsdp2":
            raise ValueError("FSDP2 application requires an fsdp2 parallel plan")
        if not self.block_module_names:
            raise ValueError("FSDP2 application must contain at least one block module")
        if len(self.block_module_names) != len(set(self.block_module_names)):
            raise ValueError("FSDP2 block module names cannot contain duplicates")
        if not self.parameter_names or self.parameter_count <= 0:
            raise ValueError("FSDP2 application requires model parameters")
        if self.parameter_mode not in {"trainable", "frozen-reference"}:
            raise ValueError(f"unsupported FSDP2 parameter_mode: {self.parameter_mode!r}")
        if self.parameter_mode == "trainable" and (
            not self.trainable_parameter_names or self.trainable_parameter_count <= 0
        ):
            raise ValueError("trainable FSDP2 applications require trainable parameters")
        if self.parameter_mode == "frozen-reference" and (
            self.trainable_parameter_names or self.trainable_parameter_count != 0
        ):
            raise ValueError("frozen-reference FSDP2 applications cannot contain trainable parameters")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "parallel_plan": self.parallel_plan.to_dict(),
            "block_module_names": list(self.block_module_names),
            "block_class_names": list(self.block_class_names),
            "parameter_names": list(self.parameter_names),
            "trainable_parameter_names": list(self.trainable_parameter_names),
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "param_dtype": self.param_dtype,
            "reduce_dtype": self.reduce_dtype,
            "root_reshard_after_forward": self.root_reshard_after_forward,
            "parameter_mode": self.parameter_mode,
            "precision_island_module_names": list(self.precision_island_module_names),
            "original_parameter_dtypes": list(self.original_parameter_dtypes),
        }


def _apply_fsdp2(
    adapter: TrainModelAdapter,
    *,
    plan: ParallelPlan,
    mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    parameter_mode: str,
    master_parameter_dtype: torch.dtype | None,
) -> FSDP2Application:
    """Apply block-wise FSDP2 in-place and return its execution description.

    The caller must inject/freeze adapters before this function and must build
    the optimizer after it returns. FSDP2 changes parameters to DTensor values,
    so creating the optimizer earlier would give it stale parameter objects.
    """

    if parameter_mode not in {"trainable", "frozen-reference"}:
        raise ValueError(f"unsupported FSDP2 parameter_mode: {parameter_mode!r}")
    if not isinstance(plan, ParallelPlan) or plan.backend != "fsdp2":
        raise TypeError("plan must be an FSDP2 ParallelPlan")
    if not isinstance(mesh, DeviceMesh):
        raise TypeError("mesh must be a DeviceMesh")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("applying FSDP2 requires an initialized process group")
    if dist.get_world_size() != plan.world_size:
        raise RuntimeError("active process-group world size differs from the FSDP2 plan")
    if mesh.device_type != "cuda":
        raise RuntimeError("WorldFoundry FSDP2 training currently requires a CUDA DeviceMesh")
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP2 CUDA application requested without CUDA availability")
    if plan.cp != 1 or plan.tp != 1:
        raise NotImplementedError("context/tensor parallel plans require a family-specific implementation")
    param_dtype_name = _dtype_name(param_dtype)
    reduce_dtype_name = _dtype_name(reduce_dtype)
    if master_parameter_dtype is not None:
        _dtype_name(master_parameter_dtype)
    if tuple(mesh.mesh.shape) != plan.mesh_shape or mesh.mesh_dim_names != (
        "dp_replicate",
        "dp_shard",
        "cp",
        "tp",
    ):
        raise ValueError("DeviceMesh shape/names differ from the resolved parallel plan")

    module = getattr(adapter, "trainable_module", None)
    if not isinstance(module, nn.Module):
        raise TypeError("adapter.trainable_module must be an nn.Module")
    if any(isinstance(parameter, DTensor) for parameter in module.parameters()):
        raise RuntimeError("FSDP2 has already been applied to the trainable module")
    expected_device = torch.device("cuda", torch.cuda.current_device())
    _module_device_audit(module, expected_device)

    raw_block_classes = getattr(adapter, "fsdp_block_classes", None)
    if not isinstance(raw_block_classes, tuple) or not raw_block_classes:
        raise ValueError("adapter must declare a non-empty fsdp_block_classes tuple")
    if any(not isinstance(value, type) or not issubclass(value, nn.Module) for value in raw_block_classes):
        raise TypeError("adapter.fsdp_block_classes must contain nn.Module classes")
    block_classes = tuple(dict.fromkeys(raw_block_classes))

    parameter_names = _parameter_names(module, trainable_only=False)
    trainable_names = _parameter_names(module, trainable_only=True)
    if not parameter_names:
        raise ValueError("FSDP2 requires a non-empty model")
    if parameter_mode == "trainable" and not trainable_names:
        raise ValueError("trainable FSDP2 requires trainable parameters")
    if parameter_mode == "frozen-reference" and trainable_names:
        raise ValueError("frozen-reference FSDP2 requires every parameter to have requires_grad=False")
    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

    matches = [
        (index, name, child)
        for index, (name, child) in enumerate(module.named_modules())
        if name and isinstance(child, block_classes)
    ]
    if not matches:
        declared = [f"{value.__module__}.{value.__qualname__}" for value in block_classes]
        raise ValueError(f"FSDP2 block classes did not match any child modules: {declared}")
    matched_classes = {type(child) for _, _, child in matches}
    unmatched_classes = [
        f"{value.__module__}.{value.__qualname__}"
        for value in block_classes
        if not any(isinstance(child, value) for _, _, child in matches)
    ]
    if unmatched_classes:
        raise ValueError(f"FSDP2 block classes had no exact module matches: {unmatched_classes}")

    policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=param_dtype,
    )
    fsdp_mesh = plan.fsdp_mesh(mesh)
    precision_islands = _precision_island_modules(
        module,
        param_dtype=param_dtype,
        master_parameter_dtype=master_parameter_dtype,
    )

    island_policy = MixedPrecisionPolicy(
        param_dtype=None,
        reduce_dtype=reduce_dtype,
        output_dtype=None,
    )
    precision_island_names: list[str] = []
    for dtype in sorted(precision_islands, key=str):
        entries = precision_islands[dtype]
        precision_island_names.extend(f"{name}:{_dtype_name(dtype)}" for name, _ in entries)
        island_modules = [child for _, child in entries]
        fully_shard(
            island_modules if len(island_modules) > 1 else island_modules[0],
            mesh=fsdp_mesh,
            reshard_after_forward=True,
            mp_policy=island_policy,
        )
    # A declared block class may itself contain another declared block class.
    # Apply the deepest modules first so every parameter belongs to exactly one
    # communication group before the root catches the remaining parameters.
    application_order = sorted(matches, key=lambda value: (-value[1].count("."), value[0]))
    for _, _, child in application_order:
        fully_shard(
            child,
            mesh=fsdp_mesh,
            reshard_after_forward=True,
            mp_policy=policy,
        )
    fully_shard(
        module,
        mesh=fsdp_mesh,
        reshard_after_forward=False,
        mp_policy=policy,
    )

    if _parameter_names(module, trainable_only=False) != parameter_names:
        raise RuntimeError("FSDP2 changed canonical parameter names")
    if _parameter_names(module, trainable_only=True) != trainable_names:
        raise RuntimeError("FSDP2 changed the trainable parameter selection")
    if not all(isinstance(parameter, DTensor) for parameter in module.parameters()):
        raise RuntimeError("FSDP2 did not convert every managed parameter to DTensor")

    block_names = tuple(name for _, name, _ in matches)
    block_class_names = tuple(sorted(f"{value.__module__}.{value.__qualname__}" for value in matched_classes))
    return FSDP2Application(
        parallel_plan=plan,
        block_module_names=block_names,
        block_class_names=block_class_names,
        parameter_names=parameter_names,
        trainable_parameter_names=trainable_names,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        param_dtype=param_dtype_name,
        reduce_dtype=reduce_dtype_name,
        root_reshard_after_forward=False,
        parameter_mode=parameter_mode,
        precision_island_module_names=tuple(precision_island_names),
        original_parameter_dtypes=tuple(
            sorted({str(parameter.dtype).removeprefix("torch.") for parameter in module.parameters()})
        ),
    )


def apply_fsdp2(
    adapter: TrainModelAdapter,
    *,
    plan: ParallelPlan,
    mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    master_parameter_dtype: torch.dtype | None = None,
) -> FSDP2Application:
    """Shard a trainable role, retaining opt-in master dtype for its optimizer."""

    return _apply_fsdp2(
        adapter,
        plan=plan,
        mesh=mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        parameter_mode="trainable",
        master_parameter_dtype=master_parameter_dtype,
    )


def apply_fsdp2_frozen_reference(
    adapter: TrainModelAdapter,
    *,
    plan: ParallelPlan,
    mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
) -> FSDP2Application:
    """Shard a frozen teacher/reference without making it optimizer state."""

    return _apply_fsdp2(
        adapter,
        plan=plan,
        mesh=mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        parameter_mode="frozen-reference",
        master_parameter_dtype=None,
    )


__all__ = [
    "FSDP2_APPLICATION_SCHEMA",
    "FSDP2Application",
    "apply_fsdp2",
    "apply_fsdp2_frozen_reference",
]
