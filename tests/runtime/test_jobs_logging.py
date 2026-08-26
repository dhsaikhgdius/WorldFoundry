from __future__ import annotations

import asyncio
import json
import sys

from worldfoundry.runtime.jobs import AsyncCommandJobStore


def test_async_job_persists_streams_events_and_child_context(tmp_path):
    async def run_job():
        store = AsyncCommandJobStore()
        job = store.submit(
            [
                sys.executable,
                "-c",
                (
                    "import os, json; "
                    "from pathlib import Path; "
                    "from worldfoundry.core.logging_setup import configure_logging, get_logger; "
                    "configure_logging(); "
                    "get_logger('child').event('INFO', 'child.ready', 'child logger ready'); "
                    "print(os.environ['WORLDFOUNDRY_LOG_CONTEXT']); "
                    "print('worker error', file=__import__('sys').stderr); "
                    "print(os.environ['WORLDFOUNDRY_LOG_FILE'])"
                ),
            ],
            output_dir=tmp_path,
            metadata={"run_id": "run-job"},
        )
        assert job._task is not None
        await job._task
        return job

    job = asyncio.run(run_job())

    assert job.status == "completed"
    assert job.stdout_log_path is not None
    assert job.stderr_log_path is not None
    assert job.event_log_path is not None
    stdout_lines = job.log_text(stream="stdout").splitlines()
    context = json.loads(stdout_lines[0])
    assert context["run_id"] == "run-job"
    assert context["job_id"] == job.job_id
    child_log_file = stdout_lines[1]
    assert child_log_file.endswith("worker.events.jsonl")
    assert child_log_file != job.event_log_path
    assert "worker error" in open(job.stderr_log_path, encoding="utf-8").read()
    parent_events = [json.loads(line) for line in open(job.event_log_path, encoding="utf-8") if line.strip()]
    assert [event["event"] for event in parent_events] == ["job.queued", "job.started", "job.finished"]
    assert all(event["run_id"] == "run-job" for event in parent_events)
    assert all(event["event"] != "child.ready" for event in parent_events)
    child_events = [json.loads(line) for line in open(child_log_file, encoding="utf-8") if line.strip()]
    assert any(event["event"] == "child.ready" for event in child_events)
