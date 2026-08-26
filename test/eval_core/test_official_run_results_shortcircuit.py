"""RP-01: official-run must not silently import preexisting results.

``ManifestBenchmarkRunner._run_official`` used to short-circuit to
``official_results_import`` whenever *any* ``results_path`` existed — including
paths auto-probed out of ``benchmark_data_root``, whose candidates even covered
``annotations.json`` (dataset labels, not model results).  An ``official-run``
against a data root holding stale results therefore "succeeded" without ever
executing the official command.

Contract under test:

* the generic data-root probe never treats ``annotations.json(l)`` as results;
* ``official-run`` ignores data-root-probed results paths (the official
  command actually executes, or the run fails closed when no command exists);
* explicit results imports remain allowed but the scorecard is prominently
  marked with ``official_results_imported`` and an
  ``imported_preexisting_results`` leaderboard blocker.
"""

from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import (
    IMPORTED_PREEXISTING_RESULTS_BLOCKER,
    _default_results_path_from_data_root,
    run_benchmark_execution,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ZOO_DIR = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog"


def _write_manifest_without_official_command(path: Path) -> None:
    """Write a minimal vbench manifest entry that has no official command."""
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "vbench",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {"verification_status": "verified"},
                        "metrics": [
                            {
                                "id": "quality",
                                "leaderboard_key": "quality",
                                "official_results": {
                                    "source_fields": ["quality"],
                                    "required_columns": ["sample_id", "quality"],
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_generic_data_root_probe_ignores_annotations(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "annotations.json").write_text("[]", encoding="utf-8")
    (data_root / "annotations.jsonl").write_text("", encoding="utf-8")

    assert _default_results_path_from_data_root("vbench", data_root) is None

    (data_root / "results.json").write_text("[]", encoding="utf-8")
    assert _default_results_path_from_data_root("vbench", data_root) == data_root / "results.json"


def test_phyground_data_root_probe_keeps_annotations_dir(tmp_path: Path) -> None:
    data_root = tmp_path / "phyground-data"
    (data_root / "annotations").mkdir(parents=True)

    resolved = _default_results_path_from_data_root("phyground", data_root)

    assert resolved == data_root / "annotations"


def test_official_run_executes_command_despite_data_root_results(tmp_path: Path) -> None:
    """Acceptance (round5 RP-01): preset results.json must not short-circuit."""
    stale_root = tmp_path / "stale-data-root"
    stale_root.mkdir()
    (stale_root / "results.json").write_text(
        json.dumps([{"sample_id": "stale", "score": 1.0}]),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample.mp4").write_bytes(b"fake")

    result = run_benchmark_execution(
        "visual-chronometer",
        output_dir=tmp_path / "visual-chronometer-official-run",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-run",
        generated_artifact_dir=generated_dir,
        benchmark_data_root=str(stale_root),
        env_overrides={
            "WORLDFOUNDRY_VISUAL_CHRONOMETER_PREDICT_BACKEND": "mock",
        },
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))
    runtime_spec = result.metadata["runtime"]

    assert result.metadata["run_status"] != "official_results_import"
    assert (tmp_path / "visual-chronometer-official-run" / "results.csv").is_file()
    assert runtime_spec["results_path"] is None
    assert runtime_spec["results_path_source"] is None
    assert runtime_spec["ignored_results_path"] == str(stale_root / "results.json")
    assert runtime_spec["ignored_results_path_source"] == "benchmark_data_root"
    assert scorecard.get("official_results_imported") is not True
    assert IMPORTED_PREEXISTING_RESULTS_BLOCKER not in (scorecard.get("leaderboard_blockers") or [])


def test_official_run_without_command_fails_closed_instead_of_importing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "benchmarks.yaml"
    _write_manifest_without_official_command(manifest_path)
    stale_root = tmp_path / "stale-data-root"
    stale_root.mkdir()
    (stale_root / "results.json").write_text(
        json.dumps([{"sample_id": "stale", "quality": 1.0}]),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "official-run",
        manifest_path=manifest_path,
        mode="official-run",
        benchmark_data_root=str(stale_root),
    )

    assert result.ok is False
    assert result.metadata["run_status"] == "missing_official_command"
    assert result.metadata["runtime"]["results_path"] is None
    assert result.metadata["runtime"]["ignored_results_path"] == str(stale_root / "results.json")
    assert result.metadata["runtime"]["ignored_results_path_source"] == "benchmark_data_root"


def test_explicit_results_import_is_marked_with_blocker(tmp_path: Path) -> None:
    manifest_path = tmp_path / "benchmarks.yaml"
    _write_manifest_without_official_command(manifest_path)
    results_path = tmp_path / "official_results.csv"
    results_path.write_text("sample_id,quality\nsample-a,1.0\n", encoding="utf-8")

    result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "explicit-import",
        manifest_path=manifest_path,
        mode="official-run",
        official_results_path=results_path,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.metadata["official_results_imported"] is True
    assert scorecard["official_results_imported"] is True
    assert IMPORTED_PREEXISTING_RESULTS_BLOCKER in scorecard["leaderboard_blockers"]
    assert result.metadata["runtime"]["results_path"] == str(results_path)
    assert result.metadata["runtime"]["results_path_source"] == "official_results_path"


def test_validation_data_root_import_still_works_and_is_marked(tmp_path: Path) -> None:
    """official-validation keeps the data-root import path but marks it."""
    manifest_path = tmp_path / "benchmarks.yaml"
    _write_manifest_without_official_command(manifest_path)
    data_root = tmp_path / "data-root"
    data_root.mkdir()
    (data_root / "results.json").write_text(
        json.dumps([{"sample_id": "sample-a", "quality": 1.0}]),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "validation-import",
        manifest_path=manifest_path,
        mode="official-validation",
        benchmark_data_root=str(data_root),
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.metadata["runtime"]["results_path"] == str(data_root / "results.json")
    assert result.metadata["runtime"]["results_path_source"] == "benchmark_data_root"
    assert result.metadata["official_results_imported"] is True
    assert scorecard["official_results_imported"] is True
    assert IMPORTED_PREEXISTING_RESULTS_BLOCKER in scorecard["leaderboard_blockers"]
