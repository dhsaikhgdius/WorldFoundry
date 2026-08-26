"""DA-07: every catalog benchmark is listed, aliased, or explicitly exempt."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.runtime.assets import (
    BENCHMARKS_DATA_ROOT,
    iter_bundled_catalog_benchmark_ids,
    load_local_asset_manifest,
    local_asset_coverage_policy,
    local_asset_manifest_benchmark_ids,
    uncovered_catalog_benchmark_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST = BENCHMARKS_DATA_ROOT / "local_assets.example.yaml"


def test_local_assets_example_covers_or_exempts_every_catalog_id() -> None:
    assert EXAMPLE_MANIFEST.is_file()
    gaps = uncovered_catalog_benchmark_ids(path=EXAMPLE_MANIFEST)
    assert gaps == (), f"catalog ids missing local_assets coverage/exemption: {gaps}"


def test_local_assets_coverage_policy_declares_vbench_aliases() -> None:
    manifest = load_local_asset_manifest(EXAMPLE_MANIFEST)
    policy = local_asset_coverage_policy(manifest)
    aliases = policy.get("aliases") or {}
    assert aliases.get("vbench") == "vchitect"
    assert aliases.get("vbench-2.0") == "vchitect"
    assert aliases.get("vbench-plus-plus") == "vchitect"
    listed = local_asset_manifest_benchmark_ids(manifest)
    assert "vchitect" in listed
    catalog_ids = set(iter_bundled_catalog_benchmark_ids())
    assert {"vbench", "vbench-2.0", "vbench-plus-plus"} <= catalog_ids


def test_uncovered_helper_detects_silent_gap(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    (catalog_root / "video").mkdir(parents=True)
    (catalog_root / "video" / "orphan-bench.yaml").write_text("id: orphan-bench\n", encoding="utf-8")
    manifest = {
        "schema_version": "worldfoundry-local-assets-v1",
        "benchmarks": [{"id": "covered", "assets": [{"id": "dataset", "kind": "dataset", "path": "/tmp/x"}]}],
        "coverage": {"aliases": {}, "exempt_ids": []},
    }
    assert uncovered_catalog_benchmark_ids(manifest=manifest, catalog_root=catalog_root) == ("orphan-bench",)
    manifest["coverage"]["exempt_ids"] = ["orphan-bench"]
    assert uncovered_catalog_benchmark_ids(manifest=manifest, catalog_root=catalog_root) == ()
