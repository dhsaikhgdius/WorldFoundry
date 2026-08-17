"""Precision policy shared by native Cosmos training loops."""

from __future__ import annotations

import torch
from torch import nn


def promote_trainable_parameters_to_fp32(module: nn.Module) -> None:
    """Keep optimizer-visible parameters in FP32 outside mixed-precision forwards."""

    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype=torch.float32)


__all__ = ["promote_trainable_parameters_to_fp32"]
