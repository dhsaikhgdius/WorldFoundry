from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from worldfoundry.core.io.paths import package_root

STUDIO_ASSET_DIR = package_root() / "studio" / "assets"
VENDOR_DIR = STUDIO_ASSET_DIR / "vendor"
SPARK_ROOT = VENDOR_DIR / "spark"
THREE_ROOT = VENDOR_DIR / "three"
SPARK_MODULE_PATH = SPARK_ROOT / "spark.module.min.js"
THREE_MODULE_PATH = THREE_ROOT / "three.module.js"
THREE_CORE_MODULE_PATH = THREE_ROOT / "three.core.js"


def local_module_url(path: Path) -> str:
    return f"/gradio_api/file={quote(path.resolve().as_posix(), safe='/')}"


_local_module_url = local_module_url

__all__ = [
    "STUDIO_ASSET_DIR",
    "VENDOR_DIR",
    "SPARK_ROOT",
    "THREE_ROOT",
    "SPARK_MODULE_PATH",
    "THREE_MODULE_PATH",
    "THREE_CORE_MODULE_PATH",
    "local_module_url",
]
