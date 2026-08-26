"""DS-01: shared HFD dataset root honors WORLDFOUNDRY_HFD_DATASET_ROOT."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import hfd_dataset_root_path, worldfoundry_path_tokens
from worldfoundry.evaluation.utils import worldfoundry_hfd_dataset_root


def test_hfd_dataset_root_prefers_explicit_env(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    explicit = tmp_path / "explicit-datasets"
    monkeypatch.setenv("WORLDFOUNDRY_HFD_DATASET_ROOT", str(explicit))
    monkeypatch.setenv("WORLDFOUNDRY_DATA_DIR", str(tmp_path / "data"))
    assert hfd_dataset_root_path() == explicit.expanduser()
    assert worldfoundry_hfd_dataset_root() == explicit.expanduser()
    tokens = worldfoundry_path_tokens()
    assert tokens["WORLDFOUNDRY_HFD_DATASET_ROOT"] == str(explicit.expanduser())


def test_hfd_dataset_root_defaults_under_data_dir(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    data = tmp_path / "data"
    monkeypatch.delenv("WORLDFOUNDRY_HFD_DATASET_ROOT", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_LOCAL_DATA_ROOT", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_BENCHMARK_DATA_ROOT", raising=False)
    monkeypatch.setenv("WORLDFOUNDRY_DATA_DIR", str(data))
    expected = data / "datasets"
    assert hfd_dataset_root_path() == expected
    assert worldfoundry_hfd_dataset_root() == expected
