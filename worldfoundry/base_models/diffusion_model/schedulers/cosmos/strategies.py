"""Sampling strategy identifiers shared by Cosmos inference models."""

from enum import Enum


class HighSigmaStrategy(str, Enum):
    NONE = "none"
    UNIFORM80_2000 = "uniform80_2000"
    LOGUNIFORM200_100000 = "LOGUNIFORM200_100000"
    SHIFT24 = "shift24"
    BALANCED_TWO_HEADS_V1 = "balanced_two_heads_v1"
    HARDCODED_20steps = "hardcoded_20steps"

    def __str__(self) -> str:
        return self.value


__all__ = ["HighSigmaStrategy"]
