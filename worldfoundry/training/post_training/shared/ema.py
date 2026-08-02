"""Checkpointable moving averages shared by native post-training loops."""

from __future__ import annotations

import torch
from torch import nn

from worldfoundry.core.nn.ema import LitEma


class DelayedModuleEMA(LitEma):
    """Snapshot trainable parameters at a configured start boundary."""

    def __init__(self, module: nn.Module, *, decay: float) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("EMA module must be torch.nn.Module")
        super().__init__(module, decay=float(decay), use_num_upates=False)
        self.register_buffer("ema_started", torch.tensor(False, dtype=torch.bool))

    def start(self, module: nn.Module) -> None:
        parameters = dict(module.named_parameters())
        shadows = dict(self.named_buffers())
        with torch.no_grad():
            for name, shadow_name in self.m_name2s_name.items():
                try:
                    parameter = parameters[name]
                except KeyError as error:
                    raise ValueError(f"EMA parameter inventory lost {name!r}") from error
                shadows[shadow_name].copy_(parameter.detach())
            self.ema_started.fill_(True)

    def update(self, module: nn.Module) -> None:
        if not bool(self.ema_started.item()):
            raise RuntimeError("delayed EMA cannot update before its start snapshot")
        self.forward(module)


__all__ = ["DelayedModuleEMA"]
