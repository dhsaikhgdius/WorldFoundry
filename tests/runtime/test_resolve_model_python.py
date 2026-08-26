from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from worldfoundry.runtime.conda import (
    clear_runtime_conda_env_cache,
    resolve_model_python,
)


@pytest.fixture(autouse=True)
def _clear_conda_cache():
    clear_runtime_conda_env_cache()
    yield
    clear_runtime_conda_env_cache()


def test_resolve_model_python_prefers_explicit(tmp_path):
    explicit = tmp_path / "custom-python"
    explicit.write_text("#!/bin/sh\n", encoding="utf-8")

    assert resolve_model_python("ac3d", explicit=explicit, env_root=tmp_path) == str(explicit)


def test_resolve_model_python_uses_ready_conda_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_CONDA_ENVS_ROOT", str(tmp_path))
    clear_runtime_conda_env_cache()
    env_bin = tmp_path / "ac3d" / "bin"
    env_bin.mkdir(parents=True)
    python = env_bin / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)

    assert resolve_model_python("ac3d", env_root=tmp_path) == str(python)


def test_resolve_model_python_warns_when_env_declared_but_missing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="worldfoundry.runtime.conda"):
        resolved = resolve_model_python("ac3d", env_root=tmp_path, fallback=sys.executable)

    assert resolved == sys.executable
    assert any("ac3d" in record.getMessage() for record in caplog.records)


def test_resolve_model_python_unknown_model_returns_fallback_quietly(caplog):
    with caplog.at_level(logging.WARNING, logger="worldfoundry.runtime.conda"):
        resolved = resolve_model_python("definitely-not-a-registered-model", fallback="/tmp/fallback-py")

    assert resolved == str(Path("/tmp/fallback-py"))
    assert not caplog.records


def test_sy02_batch2_adapters_wire_resolve_model_python() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "worldfoundry/synthesis/visual_generation/minwm/worldfoundry_runtime.py",
        "worldfoundry/synthesis/visual_generation/kairos/runtime.py",
        "worldfoundry/synthesis/visual_generation/lingbot_video/runtime.py",
        "worldfoundry/synthesis/visual_generation/forcing/runtime.py",
        "worldfoundry/synthesis/visual_generation/solaris/worldfoundry_runtime.py",
        "worldfoundry/synthesis/visual_generation/hydra/worldfoundry_runtime.py",
        "worldfoundry/synthesis/visual_generation/magic_world/worldfoundry_runtime.py",
        "worldfoundry/synthesis/visual_generation/versecrafter/worldfoundry_runtime.py",
        "worldfoundry/synthesis/visual_generation/lingbot_world_v2/runtime.py",
        "worldfoundry/synthesis/visual_generation/pusa_vidgen/adapter.py",
        "worldfoundry/synthesis/visual_generation/liveworld/worldfoundry_runtime.py",
        "worldfoundry/synthesis/visual_generation/dreamdojo/worldfoundry_runtime.py",
    )
    for rel in paths:
        text = (root / rel).read_text(encoding="utf-8")
        assert "resolve_model_python" in text, rel
        assert "from worldfoundry.runtime.conda import resolve_model_python" in text, rel


def test_sy02_batch3_adapters_wire_resolve_model_python() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "worldfoundry/synthesis/visual_generation/video_x_fun/worldfoundry_runtime.py",
        "worldfoundry/synthesis/visual_generation/animatediff/animatediff_synthesis.py",
        "worldfoundry/synthesis/visual_generation/moverse/moverse_synthesis.py",
        "worldfoundry/synthesis/visual_generation/uni3c/uni3c_synthesis.py",
        "worldfoundry/synthesis/visual_generation/hunyuan_world/hy_world_2p0_worldgen_runtime.py",
        "worldfoundry/synthesis/visual_generation/three_d_four_d/runtime.py",
        "worldfoundry/synthesis/visual_generation/multiworld/ittakestwo_runtime.py",
        "worldfoundry/synthesis/visual_generation/bernini/worldfoundry_runtime.py",
        "worldfoundry/synthesis/visual_generation/world_model/open_dreamer/worldfoundry_runtime.py",
    )
    for rel in paths:
        text = (root / rel).read_text(encoding="utf-8")
        assert "resolve_model_python" in text, rel
        assert "from worldfoundry.runtime.conda import resolve_model_python" in text, rel


def test_sy02_batch4_adapters_wire_resolve_model_python() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "worldfoundry/synthesis/visual_generation/official_video_runtime.py",
        "worldfoundry/synthesis/visual_generation/world_model/runtime_manifest.py",
        "worldfoundry/synthesis/visual_generation/inspatio_world/worldfoundry_runtime.py",
        "worldfoundry/synthesis/visual_generation/magi/worldfoundry_runner.py",
        "worldfoundry/synthesis/visual_generation/unianimate_dit/worldfoundry_runner.py",
    )
    for rel in paths:
        text = (root / rel).read_text(encoding="utf-8")
        assert "resolve_model_python" in text, rel
        assert "from worldfoundry.runtime.conda import resolve_model_python" in text, rel
        assert "sys.executable" not in text, rel
