"""DreamSim runtime and cache paths used by WBench spatial metrics."""

from __future__ import annotations

import sys
from pathlib import Path

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES
from worldfoundry.core.io.paths import package_root

RUNTIME_ROOT = Path(__file__).resolve().parent / "dreamsim_runtime"
GENERAL_PERCEPTION_ROOT = package_root() / "base_models" / "perception_core" / "general_perception"


def add_runtime_to_path() -> Path:
    """Expose the vendored `dreamsim` package for callers that import it directly."""
    for path in (RUNTIME_ROOT, GENERAL_PERCEPTION_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return RUNTIME_ROOT


def cache_dir() -> Path:
    asset = BASE_MODEL_CAPABILITIES["wbench_dreamsim"].assets[0]
    status = asset.check()
    return Path(status["matched_path"] or status["local_path"])


def load_model(*, device: str = "cuda", dreamsim_type: str = "ensemble"):
    add_runtime_to_path()
    from dreamsim import dreamsim

    return dreamsim(pretrained=True, cache_dir=str(cache_dir()), device=device, dreamsim_type=dreamsim_type)


__all__ = ["RUNTIME_ROOT", "add_runtime_to_path", "cache_dir", "load_model"]
