"""SY-03 batch13: World Explorer setup_dependencies uses run_logged_subprocess."""

from __future__ import annotations

from pathlib import Path


def test_sy03_batch13_setup_dependencies_wire_run_logged_subprocess() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (
        root / "worldfoundry/studio/native/world_explorer/setup_dependencies.py"
    ).read_text(encoding="utf-8")
    assert "run_logged_subprocess" in text
    assert "subprocess.run(" not in text
    # short HEAD probe stays on check_output
    assert "subprocess.check_output(" in text
