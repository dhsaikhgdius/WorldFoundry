"""LG-07: MetricWriter batches fsync and run.json can record log_run_id."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from worldfoundry.core.logging_setup import bind_log_context, clear_log_context
from worldfoundry.training.engine.sessions.io import MetricWriter, training_log_run_id


def test_metric_writer_batches_fsync(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    writer = MetricWriter(path, fsync_every_n=3, fsync_every_seconds=0)
    with mock.patch("worldfoundry.training.engine.sessions.io.os.fsync") as fsync:
        writer.write({"step": 1})
        writer.write({"step": 2})
        assert fsync.call_count == 0
        writer.write({"step": 3})
        assert fsync.call_count == 1
        writer.force_fsync()
        assert fsync.call_count == 2
        writer.close()
        assert fsync.call_count == 3
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2, 3]


def test_training_log_run_id_reads_context_then_env(monkeypatch) -> None:  # noqa: ANN001
    clear_log_context()
    monkeypatch.delenv("WORLDFOUNDRY_RUN_ID", raising=False)
    assert training_log_run_id() is None
    monkeypatch.setenv("WORLDFOUNDRY_RUN_ID", "env-run")
    assert training_log_run_id() == "env-run"
    token = bind_log_context(run_id="ctx-run")
    try:
        assert training_log_run_id() == "ctx-run"
    finally:
        clear_log_context()
        del token
