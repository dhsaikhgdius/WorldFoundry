"""DS-10: HFD_DATASET_CACHE_ROOT must not freeze at import time."""

from __future__ import annotations

from pathlib import Path

import worldfoundry.evaluation.utils as utils


def test_hfd_dataset_cache_root_is_lazy(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    first = tmp_path / "cache-a"
    second = tmp_path / "cache-b"
    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", str(first))
    assert utils.hfd_dataset_cache_root() == first / "data" / "hfd_datasets"
    # Attribute access stays lazy too.
    assert utils.HFD_DATASET_CACHE_ROOT == first / "data" / "hfd_datasets"

    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", str(second))
    assert utils.hfd_dataset_cache_root() == second / "data" / "hfd_datasets"
    assert utils.HFD_DATASET_CACHE_ROOT == second / "data" / "hfd_datasets"
