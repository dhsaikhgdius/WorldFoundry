"""Checkpointable delayed EMA for AnyFlow on-policy training."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AnyFlowEMA(nn.Module):
    """Snapshot at the configured warmup boundary, then apply fixed decay."""

    def __init__(self, module: nn.Module, *, decay: float, warmup_steps: int) -> None:
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError("AnyFlow EMA module must be nn.Module")
        value = float(decay)
        if not 0 <= value < 1:
            raise ValueError("AnyFlow EMA decay must lie in [0,1)")
        if isinstance(warmup_steps, bool) or int(warmup_steps) < 0:
            raise ValueError("AnyFlow EMA warmup_steps must be non-negative")
        parameters = tuple(
            (name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad
        )
        if not parameters:
            raise ValueError("AnyFlow EMA requires trainable parameters")
        self.parameter_names = tuple(name for name, _ in parameters)
        self.shadow_names = tuple(f"shadow_{index:08d}" for index in range(len(parameters)))
        self.register_buffer("decay", torch.tensor(value, dtype=torch.float32))
        self.register_buffer(
            "warmup_steps",
            torch.tensor(int(warmup_steps), dtype=torch.int64),
        )
        self.register_buffer("optimizer_steps", torch.zeros((), dtype=torch.int64))
        self.register_buffer(
            "started",
            torch.tensor(int(warmup_steps) == 0, dtype=torch.bool),
        )
        for shadow_name, (_, parameter) in zip(
            self.shadow_names,
            parameters,
            strict=True,
        ):
            self.register_buffer(shadow_name, parameter.detach().clone())
        self._stored: tuple[Tensor, ...] = ()

    def _trainable_parameters(self, module: nn.Module) -> tuple[nn.Parameter, ...]:
        active = {name: parameter for name, parameter in module.named_parameters() if parameter.requires_grad}
        if tuple(active) != self.parameter_names:
            raise ValueError("AnyFlow EMA trainable parameter inventory changed")
        return tuple(active.values())

    def _shadows(self) -> tuple[Tensor, ...]:
        buffers = dict(self.named_buffers())
        return tuple(buffers[name] for name in self.shadow_names)

    @torch.no_grad()
    def _snapshot(self, parameters: tuple[nn.Parameter, ...]) -> None:
        for shadow, parameter in zip(self._shadows(), parameters, strict=True):
            shadow.copy_(parameter.detach())

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        """Consume one committed generator optimizer step."""

        parameters = self._trainable_parameters(module)
        self.optimizer_steps.add_(1)
        if not bool(self.started.item()):
            if int(self.optimizer_steps.item()) < int(self.warmup_steps.item()):
                return
            self._snapshot(parameters)
            self.started.fill_(True)
            return
        decay = float(self.decay.item())
        for shadow, parameter in zip(self._shadows(), parameters, strict=True):
            shadow.lerp_(parameter.detach().to(dtype=shadow.dtype), 1.0 - decay)

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> None:
        if not bool(self.started.item()):
            raise RuntimeError("AnyFlow EMA has not reached its warmup boundary")
        for parameter, shadow in zip(
            self._trainable_parameters(module),
            self._shadows(),
            strict=True,
        ):
            parameter.copy_(shadow.to(dtype=parameter.dtype))

    @torch.no_grad()
    def store(self, module: nn.Module) -> None:
        if self._stored:
            raise RuntimeError("AnyFlow EMA already stores live parameters")
        self._stored = tuple(parameter.detach().clone() for parameter in self._trainable_parameters(module))

    @torch.no_grad()
    def restore(self, module: nn.Module) -> None:
        parameters = self._trainable_parameters(module)
        if len(self._stored) != len(parameters):
            raise RuntimeError("AnyFlow EMA has no matching stored parameters")
        for parameter, stored in zip(parameters, self._stored, strict=True):
            parameter.copy_(stored)
        self._stored = ()


__all__ = ["AnyFlowEMA"]
