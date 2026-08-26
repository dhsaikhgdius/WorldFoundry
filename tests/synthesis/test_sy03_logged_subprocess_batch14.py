"""SY-03 batch14: core.io media ffmpeg helpers use run_logged_subprocess."""

from __future__ import annotations

from pathlib import Path


def test_sy03_batch14_core_io_media_wire_run_logged_subprocess() -> None:
    root = Path(__file__).resolve().parents[2]
    for rel in (
        "worldfoundry/core/io/video_data.py",
        "worldfoundry/core/io/wan_video_geometry.py",
        "worldfoundry/core/io/audio.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        assert "run_logged_subprocess" in text, rel
        assert "subprocess.run(" not in text, rel

    video = (root / "worldfoundry/core/io/video.py").read_text(encoding="utf-8")
    assert "run_logged_subprocess" in video
    assert "ffmpeg_extract.stdout.log" in video
    # ffprobe metadata probe intentionally keeps subprocess.run
    assert "ffprobe" in video
    assert "subprocess.run(" in video
