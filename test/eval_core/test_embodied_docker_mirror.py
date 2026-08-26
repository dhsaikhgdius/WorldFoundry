"""Guards for scripts/embodied/mirror_docker_images.py (D-09 supply-chain hygiene)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_mirror_module():
    script_path = REPO_ROOT / "scripts" / "embodied" / "mirror_docker_images.py"
    spec = importlib.util.spec_from_file_location("wf_mirror_docker_images", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass creation resolves the defining module via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_profile(directory: Path, profile_id: str, *, image: str, source_image: str) -> None:
    payload = {"id": profile_id, "docker": {"image": image, "source_image": source_image}}
    (directory / f"{profile_id}.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_load_image_mappings_skips_identity_mirrors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_mirror_module()
    _write_profile(
        tmp_path,
        "libero",
        image="ghcr.io/allenai/vla-evaluation-harness/libero:latest",
        source_image="ghcr.io/allenai/vla-evaluation-harness/libero:latest",
    )
    _write_profile(
        tmp_path,
        "calvin",
        image="ghcr.io/openenvision/calvin:latest",
        source_image="ghcr.io/allenai/vla-evaluation-harness/calvin:latest",
    )

    mappings = module.load_image_mappings(tmp_path, ["all"])
    assert [item.profile_id for item in mappings] == ["calvin"]
    assert "skipping identity mirrors" in capsys.readouterr().err


def test_load_image_mappings_keeps_identity_with_target_prefix(tmp_path: Path) -> None:
    module = _load_mirror_module()
    _write_profile(
        tmp_path,
        "libero",
        image="ghcr.io/allenai/vla-evaluation-harness/libero:latest",
        source_image="ghcr.io/allenai/vla-evaluation-harness/libero:latest",
    )

    mappings = module.load_image_mappings(
        tmp_path, ["all"], target_prefix="registry.cn-wulanchabu.aliyuncs.com/worldfoundry"
    )
    assert len(mappings) == 1
    assert mappings[0].target_image == "registry.cn-wulanchabu.aliyuncs.com/worldfoundry/libero:latest"


def test_mirror_images_refuses_foreign_push_targets(tmp_path: Path) -> None:
    module = _load_mirror_module()
    foreign = module.ImageMapping(
        profile_id="libero",
        source_image="ghcr.io/allenai/vla-evaluation-harness/libero:latest",
        target_image="ghcr.io/allenai/vla-evaluation-harness/libero:latest",
    )
    with pytest.raises(SystemExit, match="refusing to push"):
        module.mirror_images([foreign], push=True, plan_only=True)


def test_mirror_images_allows_owned_or_explicitly_foreign_push(tmp_path: Path) -> None:
    module = _load_mirror_module()
    owned = module.ImageMapping(
        profile_id="libero",
        source_image="ghcr.io/allenai/vla-evaluation-harness/libero:latest",
        target_image="ghcr.io/openenvision/libero:latest",
    )
    foreign = module.ImageMapping(
        profile_id="libero",
        source_image="ghcr.io/allenai/vla-evaluation-harness/libero:latest",
        target_image="ghcr.io/allenai/vla-evaluation-harness/libero:latest",
    )
    # plan_only never shells out to docker; both calls must simply not raise.
    module.mirror_images([owned], push=True, plan_only=True)
    module.mirror_images([foreign], push=True, plan_only=True, allow_foreign_push=True)
