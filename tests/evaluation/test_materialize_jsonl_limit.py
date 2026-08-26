"""DS-07: materialize JSONL limit stops reading early."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.datasets.manifest import read_dataset_samples
from worldfoundry.evaluation.tasks.execution.orchestration.materialize import (
    materialize_requests_from_dataset_manifest,
)


def test_read_dataset_samples_jsonl_limit_stops_before_bad_tail(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    # Valid head + corrupt tail: early-stop with limit must not parse the tail.
    path.write_text(
        '{"sample_id": "s0", "prompt": "a"}\n'
        '{"sample_id": "s1", "prompt": "b"}\n'
        '{"sample_id": "s2", "prompt": "c"}\n'
        "NOT-JSON\n"
        '{"sample_id": "s9", "prompt": "z"}\n',
        encoding="utf-8",
    )
    rows = read_dataset_samples(path, limit=2)
    assert [row["sample_id"] for row in rows] == ["s0", "s1"]

    with pytest.raises(Exception):
        read_dataset_samples(path)


def test_materialize_manifest_passes_limit(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        "\n".join(f'{{"sample_id": "s{i}", "prompt": "p{i}"}}' for i in range(10)) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "worldfoundry-dataset-manifest",
        "dataset_id": "demo",
        "samples_path": str(samples),
        "sample_count": 10,
        "sha256": "0" * 64,
        "split": "default",
    }
    batch = materialize_requests_from_dataset_manifest(manifest, task_name="demo", limit=2)
    assert batch.sample_count == 2
    assert [req.sample_id for req in batch.requests] == ["s0", "s1"]
