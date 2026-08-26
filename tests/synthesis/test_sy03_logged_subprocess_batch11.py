"""SY-03 batch11: Studio ffmpeg exporters use run_logged_subprocess."""

from __future__ import annotations

from pathlib import Path


def test_sy03_batch11_studio_ffmpeg_wire_run_logged_subprocess() -> None:
    root = Path(__file__).resolve().parents[2]
    export_vis = (
        root
        / "worldfoundry/studio/visualization/plugins/scene3d/depth_anything_v3/export_vis.py"
    ).read_text(encoding="utf-8")
    assert "run_logged_subprocess" in export_vis
    assert "subprocess.run(" not in export_vis

    lyra = (
        root / "worldfoundry/studio/native/world_explorer/api/lyra_persistent.py"
    ).read_text(encoding="utf-8")
    assert "run_logged_subprocess" in lyra
    assert "ffmpeg_concat.stdout.log" in lyra
    # stdin-piped encode keeps Popen
    assert "subprocess.Popen(" in lyra
    assert "subprocess.run(" not in lyra
