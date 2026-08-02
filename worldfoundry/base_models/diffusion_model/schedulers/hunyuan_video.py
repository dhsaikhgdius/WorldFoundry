"""Stateless flow-matching schedule shared by every HunyuanVideo recipe."""

from __future__ import annotations

import torch

from ..components import ComponentBuildContext
from .wan import WanFlowMatchEulerScheduler


class HunyuanVideoFlowMatchEulerScheduler(WanFlowMatchEulerScheduler):
    """Official shifted Euler trajectory exposed through the native contract."""


def build_hunyuan_video_flow_match_scheduler(
    context: ComponentBuildContext,
) -> HunyuanVideoFlowMatchEulerScheduler:
    options = context.component_options
    return HunyuanVideoFlowMatchEulerScheduler(
        num_train_timesteps=int(options.get("num_train_timesteps", 1000)),
        shift=float(options.get("shift", 7.0)),
        sigma_max=float(options.get("sigma_max", 1.0)),
        sigma_min=float(options.get("sigma_min", 0.0)),
        timestep_dtype=torch.float32,
    )


__all__ = [
    "HunyuanVideoFlowMatchEulerScheduler",
    "build_hunyuan_video_flow_match_scheduler",
]
