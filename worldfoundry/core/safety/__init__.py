"""Reusable safety guardrail interfaces.

Exports resolve lazily (same pattern as ``core.checkpoint``/``core.structures``)
so that importing the guardrail Protocols does not pull the video-io stack
(imageio/numpy) or, transitively, torch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "ContentSafetyGuardrail": "worldfoundry.core.safety.guardrails",
    "GuardrailRunner": "worldfoundry.core.safety.guardrails",
    "PostprocessingGuardrail": "worldfoundry.core.safety.guardrails",
    "VideoData": "worldfoundry.core.safety.video_io",
    "get_video_filepaths": "worldfoundry.core.safety.video_io",
    "read_video": "worldfoundry.core.safety.video_io",
    "save_video": "worldfoundry.core.safety.video_io",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORT_MODULES)
