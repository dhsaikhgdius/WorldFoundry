"""DS-06: dataset manifests keep and expand WORLDFOUNDRY path tokens."""

from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.tasks.datasets.manifest import (
    build_dataset_manifest,
    resolve_dataset_samples_path,
    write_dataset_manifest,
)


def _write_samples(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([{"sample_id": "a", "prompt": "one"}, {"sample_id": "b", "prompt": "two"}]),
        encoding="utf-8",
    )
    return path


def test_build_preserves_token_root(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    data_root = tmp_path / "wf-data"
    samples = _write_samples(data_root / "datasets" / "demo" / "samples.json")
    monkeypatch.setenv("WORLDFOUNDRY_DATA_DIR", str(data_root))

    manifest = build_dataset_manifest(
        samples_path=samples,
        dataset_id="demo",
        root="${WORLDFOUNDRY_DATA_DIR}/datasets",
    )
    assert manifest.root == "${WORLDFOUNDRY_DATA_DIR}/datasets"
    assert manifest.samples_path == "demo/samples.json"

    resolved = resolve_dataset_samples_path(manifest)
    assert resolved == samples.resolve()


def test_resolve_expands_token_samples_path(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    data_root = tmp_path / "wf-data"
    samples = _write_samples(data_root / "datasets" / "tok" / "samples.json")
    monkeypatch.setenv("WORLDFOUNDRY_DATA_DIR", str(data_root))

    manifest = build_dataset_manifest(
        samples_path="${WORLDFOUNDRY_DATA_DIR}/datasets/tok/samples.json",
        dataset_id="tok",
    )
    assert manifest.root is None
    assert manifest.samples_path == "${WORLDFOUNDRY_DATA_DIR}/datasets/tok/samples.json"
    assert resolve_dataset_samples_path(manifest) == samples.resolve()

    out = write_dataset_manifest(manifest, tmp_path / "manifest.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["samples_path"].startswith("${WORLDFOUNDRY_DATA_DIR}")
