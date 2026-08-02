"""Instance-local memory, control, and research extensions."""

from .base import DiffusionExtension, DiffusionRunContext
from .frozen_context import FrozenContextSuffixExtension
from .frozen_mask import FrozenLatentMaskExtension

__all__ = [
    "DiffusionExtension",
    "DiffusionRunContext",
    "FrozenContextSuffixExtension",
    "FrozenLatentMaskExtension",
]
