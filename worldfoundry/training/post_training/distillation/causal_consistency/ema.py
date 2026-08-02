"""Frozen target-module EMA used by Causal Consistency Distillation."""

from __future__ import annotations

from math import isfinite

import torch
from torch import nn


class FrozenModuleEMA:
    """Update a distinct frozen target module from trainable source parameters."""

    def __init__(
        self,
        source_module: nn.Module,
        target_module: nn.Module,
        *,
        decay: float,
        initialize_target: bool = True,
    ) -> None:
        if not isinstance(source_module, nn.Module) or not isinstance(target_module, nn.Module):
            raise TypeError("EMA source and target must be nn.Module values")
        if source_module is target_module:
            raise ValueError("EMA source and target modules must be distinct")
        value = float(decay)
        if not isfinite(value) or not 0 <= value < 1:
            raise ValueError("EMA decay must be finite and lie in [0,1)")
        source = dict(source_module.named_parameters())
        target = dict(target_module.named_parameters())
        if tuple(source) != tuple(target):
            raise ValueError("EMA source and target parameter inventories differ")
        tracked = tuple(name for name, parameter in source.items() if parameter.requires_grad)
        if not tracked:
            raise ValueError("EMA source has no trainable parameters")
        for name in tracked:
            if source[name].shape != target[name].shape or source[name].dtype != target[name].dtype:
                raise ValueError(f"EMA parameter {name!r} shape or dtype differs")
            if source[name].device != target[name].device:
                raise ValueError(f"EMA parameter {name!r} must be colocated with its target shard")

        target_module.requires_grad_(False)
        target_module.eval()
        self.source_module = source_module
        self.target_module = target_module
        self.decay = value
        self.tracked_names = tracked
        if initialize_target:
            self.copy_source_to_target()

    @torch.no_grad()
    def copy_source_to_target(self) -> None:
        source = dict(self.source_module.named_parameters())
        target = dict(self.target_module.named_parameters())
        for name in self.tracked_names:
            target[name].copy_(source[name].detach())
        self.target_module.eval()

    @torch.no_grad()
    def update(self) -> None:
        source = dict(self.source_module.named_parameters())
        target = dict(self.target_module.named_parameters())
        one_minus_decay = 1.0 - self.decay
        for name in self.tracked_names:
            target[name].mul_(self.decay).add_(source[name].detach(), alpha=one_minus_decay)
        self.target_module.eval()


__all__ = ["FrozenModuleEMA"]
