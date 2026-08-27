"""Unit tests for the bounded ffprobe metadata probe in core video I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.core.io.video import probe_video_metadata


def _fake_result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "command": [],
        "stdout": "",
        "stderr": "",
        "returncode": 0,
        "timed_out": False,
        "kill_stuck": False,
        "duration_seconds": 0.01,
    }
    result.update(overrides)
    return result


@pytest.fixture()
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 16)
    return path


def test_probe_uses_run_bounded_command_and_parses_metadata(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    payload = {
        "streams": [
            {
                "width": 640,
                "height": 360,
                "avg_frame_rate": "24000/1001",
                "r_frame_rate": "24/1",
                "nb_frames": "48",
                "duration": "2.002",
            }
        ],
        "format": {"duration": "2.002"},
    }
    captured: dict[str, object] = {}

    def fake_run_bounded_command(command, *, timeout):
        captured["command"] = list(command)
        captured["timeout"] = timeout
        return _fake_result(stdout=json.dumps(payload))

    # probe_video_metadata imports the helper lazily, so patch it at the source.
    monkeypatch.setattr("worldfoundry.runtime.jobs.run_bounded_command", fake_run_bounded_command)

    metadata = probe_video_metadata(video_file, ffprobe_path="/fake/ffprobe", timeout_seconds=12.5)

    assert captured["command"][0] == "/fake/ffprobe"
    assert captured["command"][-1] == str(video_file)
    assert captured["timeout"] == pytest.approx(12.5)
    assert metadata == {
        "width": 640,
        "height": 360,
        "fps": pytest.approx(24000 / 1001),
        "duration_seconds": pytest.approx(2.002),
        "frame_count": 48,
    }


def test_probe_maps_bounded_timeout_to_timeout_error(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    monkeypatch.setattr(
        "worldfoundry.runtime.jobs.run_bounded_command",
        lambda command, *, timeout: _fake_result(
            returncode=124,
            timed_out=True,
            stderr="TimeoutExpired: command exceeded 5s",
        ),
    )

    with pytest.raises(TimeoutError, match="ffprobe timed out after 5s"):
        probe_video_metadata(video_file, ffprobe_path="/fake/ffprobe", timeout_seconds=5.0)


def test_probe_raises_value_error_with_stderr_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, video_file: Path
) -> None:
    monkeypatch.setattr(
        "worldfoundry.runtime.jobs.run_bounded_command",
        lambda command, *, timeout: _fake_result(returncode=1, stderr="moov atom not found\n"),
    )

    with pytest.raises(ValueError, match="moov atom not found"):
        probe_video_metadata(video_file, ffprobe_path="/fake/ffprobe")


def test_probe_raises_value_error_with_status_when_stderr_empty(
    monkeypatch: pytest.MonkeyPatch, video_file: Path
) -> None:
    monkeypatch.setattr(
        "worldfoundry.runtime.jobs.run_bounded_command",
        lambda command, *, timeout: _fake_result(returncode=3),
    )

    with pytest.raises(ValueError, match="ffprobe exited with status 3"):
        probe_video_metadata(video_file, ffprobe_path="/fake/ffprobe")


def test_probe_rejects_invalid_json_stdout(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    monkeypatch.setattr(
        "worldfoundry.runtime.jobs.run_bounded_command",
        lambda command, *, timeout: _fake_result(stdout="not-json"),
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        probe_video_metadata(video_file, ffprobe_path="/fake/ffprobe")
