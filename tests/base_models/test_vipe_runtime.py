from __future__ import annotations

from pathlib import Path

import torch

from worldfoundry.base_models.three_dimensions.general_3d.vipe import get_config_path
from worldfoundry.base_models.three_dimensions.general_3d.vipe._imports import import_config_module
from worldfoundry.base_models.three_dimensions.general_3d.vipe.assets import droid_checkpoint, require_asset
from worldfoundry.base_models.three_dimensions.general_3d.vipe.config import parse_typed_config
from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext import build as native_build
from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.specs import UPSTREAM_REVISION, get_sources


def test_vipe_pinned_native_sources_are_complete() -> None:
    sources = [Path(path) for path in get_sources()]
    assert UPSTREAM_REVISION == "157494a2aca56c9f5adbd36977d892e88401b4e2"
    assert len(sources) == 19
    assert all(path.is_file() and "vipe/csrc" in path.as_posix() for path in sources)
    assert any(path.name == "bind.cpp" for path in sources)
    assert any(path.name == "lietorch_gpu.cu" for path in sources)


def test_vipe_v1_1_default_config_is_typed() -> None:
    config = parse_typed_config(
        "default",
        [
            "pipeline=default",
            "streams.base_path=/tmp/input.mp4",
            "pipeline.init.instance=null",
            "pipeline.slam.keyframe_depth=null",
            "pipeline.post.depth_align_model=null",
        ],
        config_dir=get_config_path(),
    )
    assert config.pipeline.slam.ba.fused is False
    assert config.pipeline.slam.ba.robust_kernel is None
    assert config.pipeline.slam.cross_view_idx is None


def test_official_config_module_resolves_in_tree() -> None:
    module = import_config_module("vipe.config.slam")
    assert module.__name__ == "worldfoundry.base_models.three_dimensions.general_3d.vipe.config.slam"


def test_explicit_droid_checkpoint_override(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "droid.pth"
    torch.save({"weight": torch.ones(1)}, checkpoint)
    monkeypatch.setenv("WORLDFOUNDRY_VIPE_DROID_CHECKPOINT", str(checkpoint))
    asset = droid_checkpoint()
    assert asset.path == checkpoint
    assert require_asset(asset) == checkpoint


def test_native_status_fails_closed_when_sources_are_missing(monkeypatch) -> None:
    def _missing_sources() -> list[str]:
        raise RuntimeError("test fixture has no native sources")

    monkeypatch.setattr(native_build, "get_sources", _missing_sources)
    status = native_build.native_extension_status()

    assert status.ready is False
    assert status.module_file is None
    assert status.source_count == 0
    assert "test fixture has no native sources" in (status.reason or "")
