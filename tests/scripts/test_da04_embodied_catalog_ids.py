"""DA-04: embodied prepare scripts must track catalog/embodied IDs."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import embodied_benchmark_ids


def test_embodied_benchmark_ids_match_catalog_shard() -> None:
    ids = embodied_benchmark_ids()
    assert len(ids) >= 23
    assert ids == tuple(sorted(ids))
    assert "libero" in ids
    assert "ai2thor" in ids
    assert "larybench" in ids
    assert "_manifest" not in ids


def test_prepare_scripts_use_catalog_embodied_ids() -> None:
    from scripts.setup import prepare_embodied_official_assets as setup_mod
    from scripts.embodied import prepare_official_assets as embodied_mod

    catalog = embodied_benchmark_ids()
    assert setup_mod.EMBODIED_BENCHMARK_IDS == catalog
    assert embodied_mod.ACTIVE_BENCHMARK_IDS == catalog
