"""T2V-Turbo's native UNet LoRA seam.

The released trainer injects low-rank branches into every ``Linear``,
``Conv2d``, and ``Conv3d`` below the VideoCrafter UNet.  Its convolutional
branches are not PEFT layers, so this module implements that training graph
directly and exports the same ordered ``up, down`` tensor list consumed by the
native inference loader.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class T2VTurboLoraTarget:
    name: str
    rank: int
    up_shape: tuple[int, ...]
    down_shape: tuple[int, ...]

    @property
    def trainable_parameter_count(self) -> int:
        return _numel(self.up_shape) + _numel(self.down_shape)


@dataclass(frozen=True, slots=True)
class T2VTurboLoraAudit:
    targets: tuple[T2VTurboLoraTarget, ...]

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.targets)

    @property
    def expected_trainable_parameter_count(self) -> int:
        return sum(target.trainable_parameter_count for target in self.targets)


@dataclass(frozen=True, slots=True)
class T2VTurboLoraArtifact:
    path: Path
    file_size_bytes: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "file_size_bytes",
            MappingProxyType({str(name): int(size) for name, size in self.file_size_bytes.items()}),
        )


@dataclass(frozen=True, slots=True)
class T2VTurboLoraApplication:
    model: nn.Module
    target_audit: T2VTurboLoraAudit
    targeted_module_names: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int

    def export_adapter(
        self,
        output_dir: str | Path,
        *,
        model_state_dict: Mapping[str, object] | None = None,
    ) -> T2VTurboLoraArtifact:
        return save_t2v_turbo_lora(
            self,
            output_dir,
            model_state_dict=model_state_dict,
        )


def _numel(shape: tuple[int, ...]) -> int:
    count = 1
    for value in shape:
        count *= value
    return count


class _LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, dropout: float) -> None:
        super().__init__()
        resolved_rank = min(int(rank), base.in_features, base.out_features)
        self.linear = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None)
        self.lora_down = nn.Linear(base.in_features, resolved_rank, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_up = nn.Linear(resolved_rank, base.out_features, bias=False)
        self.scale = 1.0
        nn.init.normal_(self.lora_down.weight, std=1.0 / resolved_rank)
        nn.init.zeros_(self.lora_up.weight)
        self.linear.weight = base.weight
        if base.bias is not None:
            self.linear.bias = base.bias
        self.requires_grad_(False)
        self.lora_up.weight.requires_grad_(True)
        self.lora_down.weight.requires_grad_(True)

    def forward(self, value: Tensor) -> Tensor:
        residual = self.lora_up(self.lora_down(value))
        return self.linear(value) + self.dropout(residual) * self.scale


class _LoraConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, *, rank: int, dropout: float) -> None:
        super().__init__()
        resolved_rank = min(int(rank), base.in_channels, base.out_channels)
        self.conv = nn.Conv2d(
            base.in_channels,
            base.out_channels,
            base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=base.bias is not None,
        )
        self.lora_down = nn.Conv2d(
            base.in_channels,
            resolved_rank,
            base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=False,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.lora_up = nn.Conv2d(resolved_rank, base.out_channels, 1, bias=False)
        self.scale = 1.0
        nn.init.normal_(self.lora_down.weight, std=1.0 / resolved_rank)
        nn.init.zeros_(self.lora_up.weight)
        self.conv.weight = base.weight
        if base.bias is not None:
            self.conv.bias = base.bias
        self.requires_grad_(False)
        self.lora_up.weight.requires_grad_(True)
        self.lora_down.weight.requires_grad_(True)

    def forward(self, value: Tensor) -> Tensor:
        residual = self.lora_up(self.lora_down(value))
        return self.conv(value) + self.dropout(residual) * self.scale


class _LoraConv3d(nn.Module):
    def __init__(self, base: nn.Conv3d, *, rank: int) -> None:
        super().__init__()
        resolved_rank = min(int(rank), base.in_channels, base.out_channels)
        # These are the executable released-trainer semantics: the temporal
        # wrapper reconstructs the target with kernel/padding and uses no
        # dropout on its low-rank branch.
        self.conv = nn.Conv3d(
            base.in_channels,
            base.out_channels,
            base.kernel_size,
            padding=base.padding,
            bias=base.bias is not None,
        )
        self.lora_down = nn.Conv3d(
            base.in_channels,
            resolved_rank,
            base.kernel_size,
            padding=base.padding,
            bias=False,
        )
        self.dropout = nn.Dropout(0.0)
        self.lora_up = nn.Conv3d(resolved_rank, base.out_channels, 1, bias=False)
        self.scale = 1.0
        nn.init.normal_(self.lora_down.weight, std=1.0 / resolved_rank)
        nn.init.zeros_(self.lora_up.weight)
        self.conv.weight = base.weight
        if base.bias is not None:
            self.conv.bias = base.bias
        self.requires_grad_(False)
        self.lora_up.weight.requires_grad_(True)
        self.lora_down.weight.requires_grad_(True)

    def forward(self, value: Tensor) -> Tensor:
        residual = self.lora_up(self.lora_down(value))
        return self.conv(value) + self.dropout(residual) * self.scale


_TARGET_TYPES = (nn.Linear, nn.Conv2d, nn.Conv3d)
_INJECTED_TYPES = (_LoraLinear, _LoraConv2d, _LoraConv3d)


def _target_modules(module: nn.Module) -> list[tuple[str, nn.Module, str, nn.Module]]:
    targets: list[tuple[str, nn.Module, str, nn.Module]] = []
    for full_name, child in module.named_modules():
        if not full_name or child.__class__ not in _TARGET_TYPES:
            continue
        parent_name, _, child_name = full_name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        if isinstance(parent, _INJECTED_TYPES):
            continue
        targets.append((full_name, parent, child_name, child))
    return targets


def _target_record(name: str, module: nn.Module, rank: int) -> T2VTurboLoraTarget:
    if isinstance(module, nn.Linear):
        resolved_rank = min(rank, module.in_features, module.out_features)
        down = (resolved_rank, module.in_features)
        up = (module.out_features, resolved_rank)
    elif isinstance(module, nn.Conv2d):
        resolved_rank = min(rank, module.in_channels, module.out_channels)
        down = (resolved_rank, module.in_channels // module.groups, *module.kernel_size)
        up = (module.out_channels, resolved_rank, 1, 1)
    elif isinstance(module, nn.Conv3d):
        resolved_rank = min(rank, module.in_channels, module.out_channels)
        down = (resolved_rank, module.in_channels, *module.kernel_size)
        up = (module.out_channels, resolved_rank, 1, 1, 1)
    else:  # pragma: no cover - callers only pass the closed target set.
        raise TypeError(type(module).__name__)
    return T2VTurboLoraTarget(
        name=name,
        rank=resolved_rank,
        up_shape=up,
        down_shape=down,
    )


def audit_t2v_turbo_lora_targets(module: nn.Module, *, rank: int = 64) -> T2VTurboLoraAudit:
    if isinstance(rank, bool) or int(rank) <= 0:
        raise ValueError("T2V-Turbo LoRA rank must be positive")
    targets = _target_modules(module)
    if not targets:
        raise ValueError("T2V-Turbo UNet contains no LoRA targets")
    records = tuple(_target_record(name, child, int(rank)) for name, _, _, child in targets)
    return T2VTurboLoraAudit(targets=records)


def apply_t2v_turbo_lora(
    module: nn.Module,
    *,
    rank: int = 64,
    dropout: float = 0.1,
) -> T2VTurboLoraApplication:
    resolved_dropout = float(dropout)
    if not 0.0 <= resolved_dropout < 1.0:
        raise ValueError("T2V-Turbo LoRA dropout must be in [0, 1)")
    targets = _target_modules(module)
    audit = audit_t2v_turbo_lora_targets(module, rank=rank)
    module.requires_grad_(False)
    for _, parent, child_name, child in targets:
        if isinstance(child, nn.Linear):
            replacement: nn.Module = _LoraLinear(child, rank=int(rank), dropout=resolved_dropout)
        elif isinstance(child, nn.Conv2d):
            replacement = _LoraConv2d(child, rank=int(rank), dropout=resolved_dropout)
        else:
            replacement = _LoraConv3d(child, rank=int(rank))
        replacement.to(device=child.weight.device, dtype=child.weight.dtype)
        parent._modules[child_name] = replacement

    trainable_names = tuple(name for name, parameter in module.named_parameters() if parameter.requires_grad)
    trainable_count = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    if trainable_count != audit.expected_trainable_parameter_count:
        raise RuntimeError(
            f"T2V-Turbo LoRA exposed {trainable_count} trainable parameters; "
            f"expected {audit.expected_trainable_parameter_count}"
        )
    return T2VTurboLoraApplication(
        model=module,
        target_audit=audit,
        targeted_module_names=audit.module_names,
        trainable_parameter_names=trainable_names,
        trainable_parameter_count=trainable_count,
    )


def _ordered_adapter_tensors(
    application: T2VTurboLoraApplication,
    state_dict: Mapping[str, object],
) -> list[Tensor]:
    tensors: list[Tensor] = []
    for target in application.target_audit.targets:
        for suffix in ("lora_up.weight", "lora_down.weight"):
            key = f"{target.name}.{suffix}"
            value = state_dict.get(key)
            if not isinstance(value, Tensor):
                raise KeyError(f"T2V-Turbo adapter state is missing {key!r}")
            tensors.append(value.detach().to(device="cpu", dtype=torch.float32))
    return tensors


def save_t2v_turbo_lora(
    application: T2VTurboLoraApplication,
    output_dir: str | Path,
    *,
    model_state_dict: Mapping[str, object] | None = None,
) -> T2VTurboLoraArtifact:
    """Save the ordered tensor-list artifact used by the released inference path."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"T2V-Turbo adapter output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-incomplete-", dir=destination.parent))
    try:
        state = application.model.state_dict() if model_state_dict is None else model_state_dict
        torch.save(_ordered_adapter_tensors(application, state), temporary / "unet_lora.pt")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    artifact_path = destination / "unet_lora.pt"
    return T2VTurboLoraArtifact(
        path=destination,
        file_size_bytes={artifact_path.name: artifact_path.stat().st_size},
    )


__all__ = [
    "T2VTurboLoraApplication",
    "T2VTurboLoraArtifact",
    "T2VTurboLoraAudit",
    "T2VTurboLoraTarget",
    "apply_t2v_turbo_lora",
    "audit_t2v_turbo_lora_targets",
    "save_t2v_turbo_lora",
]
