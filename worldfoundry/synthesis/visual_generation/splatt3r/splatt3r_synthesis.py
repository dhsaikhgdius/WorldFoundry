"""Synthesis facade for the Splatt3R 3D base-model runtime."""

from __future__ import annotations

from worldfoundry.base_models.three_dimensions.general_3d.splatt3r import Splatt3RRuntime

from ..runtime_facade import RuntimeFacadeSynthesis


class Splatt3RSynthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the base-model Splatt3R runtime."""

    RUNTIME_CLS = Splatt3RRuntime
    MODEL_ID = Splatt3RRuntime.MODEL_ID
    DISPLAY_NAME = Splatt3RRuntime.DISPLAY_NAME


__all__ = ["Splatt3RSynthesis"]
