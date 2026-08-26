"""SY-03 batch17: anomaly-detector ffmpeg split uses run_logged_subprocess."""

from __future__ import annotations

from pathlib import Path


def test_sy03_batch17_anomaly_split_wires_run_logged_subprocess() -> None:
    root = Path(__file__).resolve().parents[2]
    rel = "worldfoundry/base_models/llm_mllm_core/mllm/instance_anomaly_detector/split.py"
    text = (root / rel).read_text(encoding="utf-8")
    assert "run_logged_subprocess" in text
    assert "from worldfoundry.core.process import run_logged_subprocess" in text
    assert "ffmpeg" in text
    # ffprobe metadata probe intentionally keeps subprocess.run
    assert "ffprobe" in text
    assert "subprocess.run(" in text
