from __future__ import annotations

import json
import sys

from worldfoundry.core.logging_setup import log_context
from worldfoundry.core.process import run_logged_subprocess


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
