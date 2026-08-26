from __future__ import annotations

import re
from pathlib import Path

import yaml

from worldfoundry.runtime.conda import _MODEL_SPECIFIC_ISOLATED_PACKAGES

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_TXT = REPO_ROOT / "requirements" / "worldfoundry-unified.txt"
UNIFIED_YAML = REPO_ROOT / "worldfoundry" / "data" / "models" / "runtime" / "environments" / "_unified.yaml"

_REQ_NAME = re.compile(r"^([A-Za-z0-9_.\-]+)")


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-e ") or " @ " in line:
            continue
        match = _REQ_NAME.match(line)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


def test_unified_yaml_pip_packages_covered_by_unified_txt():
    payload = yaml.safe_load(UNIFIED_YAML.read_text(encoding="utf-8"))
    declared = {
        re.split(r"[<>=!\[]", str(item), maxsplit=1)[0].strip().lower().replace("_", "-")
        for item in (payload.get("pip_packages") or [])
    }
    # Torch stack is installed from the CUDA wheel index before the txt file.
    preinstalled = {"torch", "torchvision", "torchaudio"}
    covered = _requirement_names(UNIFIED_TXT) | preinstalled
    missing = sorted(declared - covered)
    assert missing == [], f"_unified.yaml pip_packages missing from unified txt: {missing}"


def test_shared_unified_packages_not_marked_model_specific_isolated():
    # These ship in worldfoundry-unified.txt and must not force isolation.
    shared = {"denku", "controlnet-aux", "open3d"}
    overlap = shared & {name.lower() for name in _MODEL_SPECIFIC_ISOLATED_PACKAGES}
    assert overlap == set()
