"""SY-03 batch9: matrix_game / pixelsplat adapters use run_logged_subprocess."""

from __future__ import annotations

from pathlib import Path


def test_sy03_batch9_adapters_wire_run_logged_subprocess() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_1_runtime/runtime.py",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_3p5_runtime/runtime.py",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_3_runtime/worldfoundry_runtime.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/pixelsplat/worldfoundry_runtime.py",
    )
    for rel in paths:
        text = (root / rel).read_text(encoding="utf-8")
        assert "run_logged_subprocess" in text, rel
        assert "from worldfoundry.core.process import" in text, rel
        assert "subprocess.run(" not in text, rel
