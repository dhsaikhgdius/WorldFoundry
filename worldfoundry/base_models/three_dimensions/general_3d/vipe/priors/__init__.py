"""Compatibility namespace for ViPE prior modules.

Canonical owners are split by capability:
``three_dimensions.depth``, ``three_dimensions.general_3d.geocalib``, and
``perception_core.tracking.track_anything``.
"""

from worldfoundry.core.io.paths import package_root

_THREE_DIMENSIONS = package_root() / "base_models" / "three_dimensions"

__path__ = [
    str(_THREE_DIMENSIONS),
    str(_THREE_DIMENSIONS / "general_3d"),
    str(_THREE_DIMENSIONS / "optical_flow"),
]
