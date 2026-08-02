"""Wan 2.1 architecture data."""

from .i2v_14b import i2v_14B
from .t2v_14b import t2v_14B
from .t2v_1p3b import t2v_1_3B

WAN_CONFIGS = {
    "t2v-1.3B": t2v_1_3B,
    "t2v-14B": t2v_14B,
    "i2v-14B": i2v_14B,
}

__all__ = ["WAN_CONFIGS", "i2v_14B", "t2v_14B", "t2v_1_3B"]
