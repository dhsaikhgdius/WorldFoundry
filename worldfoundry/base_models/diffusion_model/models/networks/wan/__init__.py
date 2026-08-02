"""Native Wan network components shared across model recipes."""

from importlib import import_module

from .adapter import ResidualBlock, SimpleAdapter
from .model import (
    AttentionModule,
    MLP,
    CrossAttention,
    GateModule,
    Head,
    WanModel,
)
from .variants import WanControlNet


def __getattr__(name):
    if name != "WanModelStateDictConverter":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module("worldfoundry.base_models.diffusion_model.models.denoisers.wan").WanModelStateDictConverter
    globals()[name] = value
    return value

__all__ = [
    "CrossAttention",
    "AttentionModule",
    "GateModule",
    "Head",
    "MLP",
    "ResidualBlock",
    "SimpleAdapter",
    "WanModel",
    "WanModelStateDictConverter",
    "WanControlNet",
]
