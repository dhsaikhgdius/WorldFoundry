"""SY-03 batch12: MegaSAM + World Explorer launcher use run_logged_subprocess."""

from __future__ import annotations

from pathlib import Path


def test_sy03_batch12_megasam_and_launcher_wire_run_logged_subprocess() -> None:
    root = Path(__file__).resolve().parents[2]
    megasam = (root / "worldfoundry/base_models/three_dimensions/slam/megasam.py").read_text(
        encoding="utf-8"
    )
    assert "run_logged_subprocess" in megasam
    assert "subprocess.run(" not in megasam

    launcher = (
        root / "worldfoundry/studio/native/world_explorer/launcher.py"
    ).read_text(encoding="utf-8")
    assert "run_logged_subprocess" in launcher
    assert "subprocess.run(" not in launcher
    assert "client.stdout.log" in launcher
