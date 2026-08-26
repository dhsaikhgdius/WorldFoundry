from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote


STUDIO_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
VENDOR_DIR = STUDIO_ASSET_DIR / "vendor"
SPARK_ROOT = VENDOR_DIR / "spark"
THREE_ROOT = VENDOR_DIR / "three"
SPARK_MODULE_PATH = SPARK_ROOT / "spark.module.min.js"
THREE_MODULE_PATH = THREE_ROOT / "three.module.js"
THREE_CORE_MODULE_PATH = THREE_ROOT / "three.core.js"

_REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_MANIFEST_PATH = _REPO_ROOT / "scripts" / "studio" / "vendor_assets.manifest.json"
VENDOR_FETCH_HINT = (
    "Studio Spark/Three vendor JS is gitignored. Provision with: "
    "python scripts/studio/fetch_vendor_assets.py"
)


def local_module_url(path: Path) -> str:
    return f"/gradio_api/file={quote(path.resolve().as_posix(), safe='/')}"


def vendor_manifest_assets() -> list[dict[str, str]]:
    payload = json.loads(VENDOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"invalid vendor manifest: {VENDOR_MANIFEST_PATH}")
    return assets


def missing_vendor_module_paths() -> list[Path]:
    """Return vendor module paths declared by the fetch manifest that are absent."""

    missing: list[Path] = []
    for asset in vendor_manifest_assets():
        relative = asset.get("relative_path")
        if not isinstance(relative, str) or not relative:
            continue
        path = VENDOR_DIR / relative
        if not path.is_file():
            missing.append(path)
    return missing


def require_vendor_modules() -> None:
    """Raise ``FileNotFoundError`` with a fetch hint when Spark/Three JS is missing."""

    missing = missing_vendor_module_paths()
    if not missing:
        return
    rendered = ", ".join(str(path) for path in missing)
    raise FileNotFoundError(f"Missing Studio vendor assets: {rendered}. {VENDOR_FETCH_HINT}")


_local_module_url = local_module_url

__all__ = [
    "STUDIO_ASSET_DIR",
    "VENDOR_DIR",
    "SPARK_ROOT",
    "THREE_ROOT",
    "SPARK_MODULE_PATH",
    "THREE_MODULE_PATH",
    "THREE_CORE_MODULE_PATH",
    "VENDOR_MANIFEST_PATH",
    "VENDOR_FETCH_HINT",
    "local_module_url",
    "vendor_manifest_assets",
    "missing_vendor_module_paths",
    "require_vendor_modules",
]
