"""On-demand Sana network registration.

Importing a scheduler or a small network utility must not initialize every
image, video, ControlNet, GDN, and Triton implementation.  Model construction
calls :func:`register_networks` with the requested registry type instead.
"""

from __future__ import annotations

import os
from importlib import import_module

_REGISTERED_MODULES: set[str] = set()


def _model_modules(model_type: str | None, profile: str) -> tuple[str, ...]:
    if profile == "v2v" or (model_type or "").startswith("SanaMSVideoV2V"):
        return ("sana_v2v_attn_blocks", "sana_multi_scale_video_v2v")
    if profile == "wm" or (model_type or "").startswith("SanaMSVideoCamCtrl"):
        os.environ.setdefault("WORLDFOUNDRY_SANA_NETS_PROFILE", "wm")
        return (
            "sana_gdn_blocks",
            "sana_gdn_blocks_triton",
            "sana_gdn_camctrl_blocks",
            "sana_multi_scale_video_camctrl",
        )
    if (model_type or "").startswith("SanaMSControlNet"):
        return ("sana_multi_scale_controlnet",)
    if (model_type or "").startswith("SanaMSVideo"):
        return ("sana_multi_scale_video",)
    return ("sana_multi_scale",)


def register_networks(model_type: str | None = None, *, profile: str | None = None) -> None:
    """Import only the modules needed to register ``model_type`` with MMCV."""

    selected_profile = (profile or os.environ.get("WORLDFOUNDRY_SANA_NETS_PROFILE", "")).strip().lower()
    for module_name in _model_modules(model_type, selected_profile):
        if module_name in _REGISTERED_MODULES:
            continue
        import_module(f".{module_name}", __name__)
        _REGISTERED_MODULES.add(module_name)


__all__ = ["register_networks"]
