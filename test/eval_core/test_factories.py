from __future__ import annotations

import json
from pathlib import Path

from test.eval_core.factories import (
    write_benchmark_manifest,
    write_json_document,
    write_model_manifest,
    write_targets_manifest,
)


def test_write_targets_manifest(tmp_path: Path) -> None:
    path = write_targets_manifest(tmp_path / "targets.json", [{"id": "a"}])
    assert json.loads(path.read_text(encoding="utf-8")) == {"targets": [{"id": "a"}]}


def test_write_model_and_benchmark_manifests(tmp_path: Path) -> None:
    models = write_model_manifest(tmp_path / "models", {"model_id": "m"})
    benches = write_benchmark_manifest(tmp_path / "benches", {"benchmark_id": "b"})
    assert models.name == "models.yaml"
    assert benches.name == "benchmarks.yaml"
    assert json.loads(models.read_text(encoding="utf-8"))["model_id"] == "m"
    assert json.loads(benches.read_text(encoding="utf-8"))["benchmark_id"] == "b"


def test_write_json_document_creates_parents(tmp_path: Path) -> None:
    path = write_json_document(tmp_path / "nested" / "doc.json", {"ok": True})
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
