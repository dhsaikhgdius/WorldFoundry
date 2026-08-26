from __future__ import annotations

import json
import sys

import pytest

from worldfoundry.core.logging_setup import log_context
from worldfoundry.core.process import run_logged_subprocess, synthesis_timeout_seconds


def test_logged_subprocess_persists_lifecycle_and_parent_context(tmp_path):
    stdout_path = tmp_path / "worker.stdout.log"
    stderr_path = tmp_path / "worker.stderr.log"

    with log_context(run_id="run-process", benchmark_id="bench-process"):
        completed = run_logged_subprocess(
            [
                sys.executable,
                "-c",
                (
                    "from worldfoundry.core.logging_setup import configure_logging, get_logger; "
                    "configure_logging(); "
                    "get_logger('child').event('INFO', 'child.ready', 'child logger ready'); "
                    "print('worker stdout')"
                ),
            ],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    assert completed.returncode == 0
    assert stdout_path.read_text() == "worker stdout\n"
    lifecycle_path = tmp_path / "logs" / "worker.stdout.events.jsonl"
    events = [json.loads(line) for line in lifecycle_path.read_text().splitlines() if line]
    assert [event["event"] for event in events] == ["subprocess.started", "child.ready", "subprocess.finished"]
    assert all(event["run_id"] == "run-process" for event in events)
    assert all(event["benchmark_id"] == "bench-process" for event in events)


def test_synthesis_timeout_seconds_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_SYNTHESIS_TIMEOUT_SECONDS", raising=False)
    assert synthesis_timeout_seconds(42.0) == 42.0
    monkeypatch.setenv("WORLDFOUNDRY_SYNTHESIS_TIMEOUT_SECONDS", "60")
    assert synthesis_timeout_seconds(42.0) == 60.0
    monkeypatch.setenv("WORLDFOUNDRY_SYNTHESIS_TIMEOUT_SECONDS", "0")
    assert synthesis_timeout_seconds(42.0) is None


def test_longcat_execute_uses_logged_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from worldfoundry.synthesis.visual_generation.longcat_video import worldfoundry_runtime as mod

    calls: list[dict] = []

    def fake_run(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        return type("C", (), {"returncode": 0})()

    monkeypatch.setattr(mod, "run_logged_subprocess", fake_run)
    monkeypatch.setattr(mod, "_video_files_in", lambda path: [])
    monkeypatch.setattr(mod, "_preferred_video_output", lambda files: None)

    plan = mod.LongCatVideoRuntimePlan(
        command=(sys.executable, "-c", "pass"),
        workdir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "ckpt"),
        output_dir=str(tmp_path / "out"),
        output_path=str(tmp_path / "out" / "video.mp4"),
        env={},
        task_type="t2v",
        script="run_inference_text_to_video.py",
    )
    (tmp_path / "out").mkdir()
    runtime = mod.LongCatVideoRuntime(checkpoint_dir=tmp_path / "ckpt")
    result = runtime.run_plan(plan, timeout_seconds=12, log_dir=tmp_path / "logs")
    assert calls and calls[0]["timeout"] == 12.0
    assert result["returncode"] == 0
