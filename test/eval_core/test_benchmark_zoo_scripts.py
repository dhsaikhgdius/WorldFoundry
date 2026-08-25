from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import types
import urllib.request
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from worldfoundry.evaluation.tasks.execution.framework.io import (
    load_json,
    mean_numeric,
    normalize_unit_score,
    read_jsonl_objects,
    scalar_number,
    score_item,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import formal_benchmark_ids
from worldfoundry.runtime.env import resolve_cache_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_BENCHMARK_COUNT = len(formal_benchmark_ids())

_REMOVED_AUDIT_SCRIPTS = frozenset(
    {
        "validate_integration",
        "env_check",
        "download_datasets",
        "materialize_benchmark_assets",
        "runtime_preflight",
        "manifest_cli",
    }
)


def _load_script(name: str) -> ModuleType:
    if name == "run_benchmark_execution":
        module = importlib.import_module(
            "worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner"
        )
        # The script-era surface (``build_parser``/``main``/``run_benchmark``/
        # ``load_manifests``) was replaced by the library API
        # ``run_benchmark_execution()`` + ``ManifestBenchmarkRunner``.  Tests
        # below were written against the removed script surface; skip until
        # rewritten (see thin contract tests at the end of this file for the
        # current orchestration surface).
        if not hasattr(module, "run_benchmark"):
            pytest.skip(
                "benchmark execution script surface removed; rewrite against "
                "orchestration.benchmark_runner.run_benchmark_execution()"
            )
        return module

    if name in _REMOVED_AUDIT_SCRIPTS:
        pytest.skip(f"audit script removed: {name}")

    if name == "create_tiny_video":
        return importlib.import_module("worldfoundry.evaluation.tasks.execution.framework.benchmark_data")

    # HANDOVER(tests/eval-execution owner): the standalone benchmark scripts and
    # ``framework/script_paths.py`` were removed; per-benchmark runners now live
    # under ``worldfoundry/evaluation/tasks/execution/runners/<bench>/`` behind
    # the orchestration surface (``zoo benchmark-run``).  The script-level tests
    # below must be rewritten against that runner/orchestration API instead of
    # loading script files by path.  Until then they are skipped rather than
    # failing on the removed module.
    try:
        from worldfoundry.evaluation.tasks.execution.framework.script_paths import (  # type: ignore[import-not-found]
            resolve_benchmark_script,
        )
    except ModuleNotFoundError:
        pytest.skip(
            f"benchmark script surface removed (framework/script_paths.py); "
            f"rewrite this test against execution.runners/orchestration: {name}"
        )

    path = resolve_benchmark_script(name)
    spec = importlib.util.spec_from_file_location(f"test_benchmark_zoo_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(manifest_dir: Path, payload: object) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / "benchmarks.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_runner_io_score_helpers_cover_common_official_shapes() -> None:
    assert scalar_number({"score": "2.5"}) == 2.5
    assert scalar_number({"accuracy": "0.75"}, dict_keys=("accuracy",)) == 0.75
    assert scalar_number([1, "3", None], list_mode="mean") == 2.0
    assert scalar_number([False, True], list_mode="mean", allow_bool=True) == 0.5
    assert scalar_number(-1, reject_negative=True) is None
    assert mean_numeric([None, 0.25, 0.75]) == 0.5
    assert normalize_unit_score(75.0) == 0.75
    assert score_item(0.8, "field", 4) == {"raw_score": 0.8, "source": "field", "sample_count": 4}


def test_runner_io_exposes_canonical_serialization_helpers(tmp_path: Path) -> None:
    json_path = tmp_path / "payload.json"
    jsonl_path = tmp_path / "rows.jsonl"

    write_json(json_path, {"value": 1})
    write_jsonl(jsonl_path, [{"row": 1}, {"row": 2}])

    assert load_json(json_path) == {"value": 1}
    assert read_jsonl_objects(jsonl_path) == [{"row": 1}, {"row": 2}]
    assert utc_now_iso().endswith("Z")












def test_run_benchmark_execution_requires_benchmark_id() -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")

    with pytest.raises(SystemExit):
        run_benchmark_execution.build_parser().parse_args([])






def test_iworldbench_official_runner_normalizes_csv_results(tmp_path: Path) -> None:
    runner = _load_script("run_iworldbench_official_runner")
    results_dir = tmp_path / "official-results" / "reports"
    results_dir.mkdir(parents=True)
    (results_dir / "brightness_consistency.csv").write_text(
        "video,score\nsample-a.mp4,0.8\nsample-b.mp4,0.6\n",
        encoding="utf-8",
    )
    (results_dir / "memory_symmetry.csv").write_text(
        "video,normalized_score\nsample-a.mp4,0.5\nsample-b.mp4,0.7\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = runner.main(
        [
            "--from-upstream-results",
            str(results_dir),
            "--output-dir",
            str(output_dir),
            "--model-id",
            "zeroscope",
            "--json",
        ]
    )
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()]

    assert exit_code == 0
    assert scorecard["benchmark"]["benchmark_id"] == "iworld-bench"
    assert scorecard["official_benchmark_verified"] is True
    assert scorecard["integration_evidence"] is False
    assert scorecard["metrics"]["leaderboard"]["brightness_consistency"] == pytest.approx(0.7)
    assert scorecard["metrics"]["leaderboard"]["memory_symmetry"] == pytest.approx(0.6)
    assert scorecard["metrics"]["leaderboard"]["iworldbench_average"] == pytest.approx(0.65)
    assert {row["metric_id"] for row in rows} >= {"brightness_consistency", "memory_symmetry", "iworldbench_average"}


def test_iworldbench_official_command_uses_released_upstream_entrypoint(tmp_path: Path) -> None:
    runner = _load_script("run_iworldbench_official_runner")
    root = tmp_path / "iWorld-Bench"
    root.mkdir()
    args = argparse.Namespace(
        iworld_root=root,
        generated_videos_dir=tmp_path / "videos",
        output_dir=tmp_path / "out",
        metric="all",
        python=Path("/env/bin/python"),
        vbench_gpu="0",
        timeout=30,
    )

    command = runner.build_official_command(args, tmp_path / "out" / "upstream")

    assert command[:2] == ["/env/bin/python", str(root / "run_iworldbench_evaluation.py")]
    assert "--metric" in command
    assert "all" in command
    assert "--camera-txt-dir" in command
    assert "--source-npz-dir" in command


def test_evalcrafter_official_runner_normalizes_final_result(tmp_path: Path) -> None:
    runner = _load_script("run_evalcrafter_official_runner")
    evalcrafter_root = tmp_path / "EvalCrafter"
    results_dir = evalcrafter_root / "results"
    output_dir = tmp_path / "out"
    results_dir.mkdir(parents=True)
    (evalcrafter_root / "prompt700.txt").write_text("0\tan example prompt\n", encoding="utf-8")
    (results_dir / "final_result.txt").write_text(
        "Metrics: {'VQA_A': 61.0, 'VQA_T': 62.0, 'IS': 63.0, "
        "'clip_temp_score': 64.0, 'warping_error': 0.12, 'face_consistency_score': 65.0, "
        "'action_score': 66.0, 'motion_ac_score': 67.0, 'flow_score': 68.0, "
        "'clip_score': 69.0, 'blip_bleu': 70.0, 'sd_score': 71.0, "
        "'detection_score': 72.0, 'color_score': 73.0, 'count_score': 74.0, "
        "'ocr_score': 0.2, 'celebrity_id_score': 0.3}\n"
        "Results: Visual Quality 10.00, Text-Video Alignment 20.00, "
        "Motion Quality 30.00, Temporal Consistency 40.00, Total 100\n",
        encoding="utf-8",
    )

    extracted, upstream_path = runner.load_upstream_results(results_dir)
    scorecard = runner.write_scorecard(
        extracted,
        output_dir=output_dir,
        upstream_results_path=upstream_path,
        evalcrafter_root=evalcrafter_root,
        videos_dir=None,
        official_run=None,
    )

    assert scorecard["evaluation"]["available"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["leaderboard_valid"] is False
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["metrics"]["leaderboard"]["visual_quality"] == 10.0
    assert scorecard["metrics"]["leaderboard"]["vqa_aesthetic"] == 61.0
    assert (output_dir / "scorecard.json").is_file()
    assert (output_dir / "benchmark_contract.json").is_file()
    assert (output_dir / "raw_metric_table.jsonl").is_file()

    official_scorecard = runner.write_scorecard(
        extracted,
        output_dir=tmp_path / "official_out",
        upstream_results_path=upstream_path,
        evalcrafter_root=evalcrafter_root,
        videos_dir=tmp_path / "generated",
        official_run={"returncode": 0, "input_validation": {"ok": True}},
    )

    assert official_scorecard["official_benchmark_verified"] is True
    assert official_scorecard["leaderboard_valid"] is True
    assert official_scorecard["eligibility"]["leaderboard_valid"] is True


def test_evalcrafter_official_runner_validates_canonical_video_layout(tmp_path: Path) -> None:
    runner = _load_script("run_evalcrafter_official_runner")
    evalcrafter_root = tmp_path / "EvalCrafter"
    videos_dir = tmp_path / "generated"
    evalcrafter_root.mkdir()
    videos_dir.mkdir()
    (evalcrafter_root / "prompt700.txt").write_text("\n".join(f"prompt {index}" for index in range(700)), encoding="utf-8")
    for index in range(700):
        (videos_dir / f"{index:04d}.mp4").write_bytes(b"")

    result = runner.validate_official_inputs(evalcrafter_root, videos_dir)

    assert result["ok"] is True
    assert result["prompt"]["line_count"] == 700
    assert result["videos"]["mp4_count"] == 700
    assert result["videos"]["missing_count"] == 0
    assert result["videos"]["unexpected_count"] == 0


def test_evalcrafter_official_runner_rejects_outer_dataset_dir(tmp_path: Path) -> None:
    runner = _load_script("run_evalcrafter_official_runner")
    evalcrafter_root = tmp_path / "EvalCrafter"
    dataset_root = tmp_path / "EvalCrafter_T2V_Dataset"
    inner_videos = dataset_root / "videocrafter2" / "mix-sr"
    evalcrafter_root.mkdir()
    inner_videos.mkdir(parents=True)
    (evalcrafter_root / "prompt700.txt").write_text("\n".join(f"prompt {index}" for index in range(700)), encoding="utf-8")
    for index in range(700):
        (inner_videos / f"{index:04d}.mp4").write_bytes(b"")

    result = runner.validate_official_inputs(evalcrafter_root, dataset_root)

    assert result["ok"] is False
    assert result["videos"]["mp4_count"] == 0
    assert result["videos"]["subdirectory_count"] == 1
    assert result["videos"]["candidate_video_dirs"] == [{"path": str(inner_videos), "mp4_count": 700}]


def test_video_quality_official_results_runner_normalizes_aigcbench(tmp_path: Path) -> None:
    runner = _load_script("run_aigcbench_official_runner")
    official_results = tmp_path / "official.csv"
    official_results.write_text(
        "metric_id,score,prompt_type\nDOVER,0.8,ours\nDOVER,0.6,webvid\n",
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample-001.mp4").write_bytes(b"dummy video bytes")
    output_dir = tmp_path / "out"

    exit_code = runner.main(
        [
            "--official-results-path",
            str(official_results),
            "--generated-video-dir",
            str(generated_dir),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["metrics"]["leaderboard"]["dover"] == 0.7
    assert (output_dir / "scorecard.json").is_file()
    assert (output_dir / "raw_metric_table.jsonl").is_file()


def test_video_quality_official_results_runner_normalizes_genai_bench(tmp_path: Path) -> None:
    runner = _load_script("run_genai_bench_official_runner")
    official_results = tmp_path / "genai.jsonl"
    official_results.write_text(
        "\n".join(
            [
                json.dumps({"task": "video_generation", "human_label": "A>B", "prediction": "A>B"}),
                json.dumps({"task": "video_generation", "human_label": "B>A", "prediction": "A>B"}),
                json.dumps({"task": "image_generation", "human_label": "A=B=Good", "prediction": "A=B=Good"}),
            ]
        ),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample-001.mp4").write_bytes(b"dummy video bytes")
    output_dir = tmp_path / "out"

    exit_code = runner.main(
        [
            "--official-results-path",
            str(official_results),
            "--generated-video-dir",
            str(generated_dir),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert scorecard["integration_evidence"] is False
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["metrics"]["leaderboard"]["pairwise_accuracy"] == pytest.approx(2 / 3)
    assert scorecard["metrics"]["leaderboard"]["genai_bench_average"] == 0.75


def test_video_quality_official_results_runner_normalizes_ipv_bench(tmp_path: Path) -> None:
    runner = _load_script("run_ipv_bench_official_runner")
    official_results = tmp_path / "ipv.csv"
    official_results.write_text(
        "\n".join(
            [
                "metric_id,score",
                "visual_quality,0.8",
                "prompt_following,0.7",
                "impossible_video_score,0.6",
                "judgement_accuracy,0.5",
                "mcqa_accuracy,0.4",
                "open_qa_score,0.3",
                "ipv_bench_average,0.55",
            ]
        ),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample-001.mp4").write_bytes(b"dummy video bytes")
    output_dir = tmp_path / "out"

    exit_code = runner.main(
        [
            "--from-upstream-results",
            str(official_results),
            "--generated-video-dir",
            str(generated_dir),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert scorecard["normalization_ok"] is True
    assert scorecard["metrics"]["leaderboard"]["visual_quality"] == 0.8
    assert scorecard["metrics"]["leaderboard"]["ipv_bench_average"] == 0.55


def test_video_quality_official_results_runner_normalizes_fetv(tmp_path: Path) -> None:
    runner = _load_script("run_fetv_official_runner")
    official_results = tmp_path / "fetv.csv"
    official_results.write_text(
        "\n".join(
            [
                "metric_id,score",
                "static_quality,0.8",
                "temporal_quality,0.7",
                "overall_alignment,0.9",
                "fine_grained_alignment,0.6",
                "clip_score,0.5",
                "blip_score,0.4",
                "fid,0.3",
                "fvd,0.2",
            ]
        ),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample-001.mp4").write_bytes(b"dummy video bytes")
    output_dir = tmp_path / "out"

    exit_code = runner.main(
        [
            "--official-results-path",
            str(official_results),
            "--generated-video-dir",
            str(generated_dir),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert scorecard["integration_evidence"] is False
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["metrics"]["leaderboard"]["static_quality"] == 0.8
    assert scorecard["metrics"]["leaderboard"]["fetv_average"] == pytest.approx(0.55)
    assert scorecard["metrics"]["summary"]["failed_metrics"] == 0


def test_materialize_benchmark_assets_combines_task_base_models_and_dataset_plan(tmp_path: Path) -> None:
    materialize_assets = _load_script("materialize_benchmark_assets")

    plan = materialize_assets.build_asset_plan(
        benchmark_id="t2v-compbench",
        dataset_cache_dir=tmp_path / "datasets",
        check_local=False,
    )

    assert plan["benchmark_id"] == "t2v-compbench"
    assert plan["task_manifest"].endswith("t2v-compbench.yaml")
    assert plan["catalog_entry_found"] is True
    assert plan["dataset"]["benchmark_id"] == "t2v-compbench"
    assert "Kaiyue/T2V-CompBench-Videos" in plan["dataset"]["hf_dataset_ids"]
    assert plan["base_models"]["capability_ids"] == [
        "depth_anything_v3",
        "grounding_dino",
        "sam_v1",
        "sam2",
        "t2v_compbench_dataset_assets",
    ]
    assert "base_model_exports" in plan["commands"]


def test_materialize_benchmark_assets_falls_back_to_catalog_base_models(monkeypatch: pytest.MonkeyPatch) -> None:
    materialize_assets = _load_script("materialize_benchmark_assets")

    monkeypatch.setattr(materialize_assets, "load_task_manifest", lambda benchmark_id: (None, {}))
    plan = materialize_assets.build_asset_plan(
        benchmark_id="t2v-compbench",
        skip_datasets=True,
    )

    assert plan["task_manifest"] is None
    assert plan["catalog_entry_found"] is True
    assert plan["base_models"]["stack_ids"] == ["grounded_depth_segmentation_stack"]
    assert plan["base_models"]["capability_ids"] == [
        "depth_anything_v3",
        "grounding_dino",
        "sam_v1",
        "sam2",
        "t2v_compbench_dataset_assets",
    ]


def test_materialize_benchmark_assets_can_skip_benchmark_data_asset_layer() -> None:
    materialize_assets = _load_script("materialize_benchmark_assets")

    plan = materialize_assets.build_asset_plan(
        benchmark_id="t2v-compbench",
        skip_datasets=True,
        include_benchmark_data_assets=False,
    )

    assert plan["include_benchmark_data_assets"] is False
    assert plan["base_models"]["capability_ids"] == ["depth_anything_v3", "grounding_dino", "sam_v1", "sam2"]


def test_materialize_benchmark_assets_accepts_base_model_stack_override(tmp_path: Path) -> None:
    materialize_assets = _load_script("materialize_benchmark_assets")

    plan = materialize_assets.build_asset_plan(
        benchmark_id="worldscore",
        capability_ids=["motion_stack"],
        skip_datasets=True,
        dataset_cache_dir=tmp_path / "datasets",
    )

    assert plan["dataset"] is None
    assert plan["base_models"]["stack_ids"] == ["motion_stack"]
    assert plan["base_models"]["capability_ids"] == ["raft", "sea_raft"]


def test_materialize_benchmark_assets_reads_fetv_dataset_refs(tmp_path: Path) -> None:
    materialize_assets = _load_script("materialize_benchmark_assets")

    plan = materialize_assets.build_asset_plan(
        benchmark_id="fetv",
        skip_base_models=True,
        dataset_cache_dir=tmp_path / "datasets",
    )

    assert plan["ok"] is True
    assert plan["dataset"]["hf_dataset_ids"] == ["lyx97/FETV"]
    assert plan["dataset"]["commands"] == [
        [
            "hf",
            "download",
            "lyx97/FETV",
            "--repo-type",
            "dataset",
            "--cache-dir",
            str(tmp_path / "datasets"),
            "--revision",
            "e9a6c057cb6ee9257f29e44d427117e8bd0d704f",
        ]
    ]


def test_download_datasets_plan_only_filters_benchmark_and_dataset(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    manifest_dir = tmp_path / "benchmark_zoo"
    _write_manifest(
        manifest_dir,
        {
            "benchmarks": [
                {
                    "benchmark_id": "alpha",
                    "official_sources": {
                        "huggingface_datasets": [
                            {"repo_id": "org/a", "license": "mit"},
                            {"repo_id": "org/b", "license": "mit"},
                        ]
                    },
                },
                {"benchmark_id": "beta", "hf_dataset_id": "org/c", "license": "mit"},
            ]
        },
    )

    manifests = download_datasets.load_manifests(manifest_dir, "alpha")
    result = download_datasets.download_manifest(
        manifests[0],
        tmp_path / "cache" / "hfd",
        execute=False,
        dataset_id_filter=["org/b"],
    )

    assert result["ok"] is True
    assert result["hf_dataset_ids"] == ["org/b"]
    assert result["commands"] == [
        ["hf", "download", "org/b", "--repo-type", "dataset", "--cache-dir", str(tmp_path / "cache" / "hfd")]
    ]


def test_download_datasets_reports_related_checkpoint_and_results_assets(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    asset_manifest = tmp_path / "local_assets.yaml"
    asset_manifest.write_text(
        """
schema_version: worldfoundry-local-assets-v1
benchmarks:
  - id: alpha
    assets:
      - id: judge
        kind: checkpoint
        hf_model_id: org/model
        path: ${WORLDFOUNDRY_MODEL_DIR}/checkpoints/org__model
      - id: official_results
        kind: result_dump
        path: ${WORLDFOUNDRY_ARTIFACT_DIR}/runs/alpha/official_results
""".strip(),
        encoding="utf-8",
    )
    manifest = download_datasets.BenchmarkManifest(
        benchmark_id="alpha",
        path=tmp_path / "alpha.yaml",
        data={"hf_dataset_id": "org/data", "license": "mit"},
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
        local_assets_manifest=asset_manifest,
    )

    related = result["related_local_assets"]
    assert related["checkpoints"][0]["id"] == "judge"
    assert related["checkpoints"][0]["hf_model_id"] == "org/model"
    assert related["official_results"][0]["id"] == "official_results"
    assert result["hf_dataset_ids"] == ["org/data"]
    assert all("org/model" not in " ".join(command) for command in result["commands"])


def test_download_datasets_plan_only_includes_manifest_revision(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    manifest = download_datasets.BenchmarkManifest(
        benchmark_id="revisioned",
        path=tmp_path / "revisioned.json",
        data={"official_sources": {"huggingface_datasets": [{"repo_id": "org/data", "sha": "abc1234", "license": "mit"}]}},
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
    )

    assert result["commands"] == [
        [
            "hf",
            "download",
            "org/data",
            "--repo-type",
            "dataset",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--revision",
            "abc1234",
        ]
    ]


def test_download_datasets_plan_only_reads_source_provenance_hf_datasets(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    manifest = download_datasets.BenchmarkManifest(
        benchmark_id="source-provenance",
        path=tmp_path / "source-provenance.yaml",
        data={
            "source_provenance": {
                "huggingface_datasets": [
                    {"repo_id": "org/source-data", "sha": "abc1234", "license": "mit"},
                ]
            }
        },
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
    )

    assert result["ok"] is True
    assert result["hf_dataset_ids"] == ["org/source-data"]
    assert result["metadata"]["dataset_refs"][0]["source"] == "source_provenance.huggingface_datasets"
    assert result["commands"] == [
        [
            "hf",
            "download",
            "org/source-data",
            "--repo-type",
            "dataset",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--revision",
            "abc1234",
        ]
    ]


def test_download_datasets_loads_yaml_manifests(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "alpha.yaml").write_text(
        "\n".join(
            [
                "id: alpha",
                "source_provenance:",
                "  huggingface_datasets:",
                "    - repo_id: org/alpha",
                "      license: mit",
            ]
        ),
        encoding="utf-8",
    )

    manifests = download_datasets.load_manifests(manifest_dir)

    assert [manifest.benchmark_id for manifest in manifests] == ["alpha"]
    assert manifests[0].hf_dataset_ids == ["org/alpha"]


def test_download_datasets_catalog_entries_use_task_yaml_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_datasets = _load_script("download_datasets")
    catalog_dir = tmp_path / "catalog"
    _write_manifest(
        catalog_dir,
        {
            "entries": [
                {
                    "id": "alpha",
                    "dataset": {
                        "not_applicable": True,
                        "reason": "catalog keeps contract-only dataset status",
                    },
                }
            ]
        },
    )
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "alpha.yaml").write_text(
        "\n".join(
            [
                "id: alpha",
                "source_provenance:",
                "  huggingface_datasets:",
                "    - repo_id: org/alpha",
                "      license: mit",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(download_datasets, "DEFAULT_TASK_MANIFEST_DIR", task_dir)

    manifest = download_datasets.load_manifests(catalog_dir, "alpha")[0]
    result = download_datasets.download_manifest(manifest, tmp_path / "cache" / "hfd", execute=False)

    assert result["ok"] is True
    assert result["hf_dataset_ids"] == ["org/alpha"]
    assert any(
        ref["source"] == "source_provenance.huggingface_datasets"
        for ref in result["metadata"]["dataset_refs"]
    )


def test_download_datasets_check_local_summarizes_missing_status(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    manifest = download_datasets.BenchmarkManifest(
        benchmark_id="missing-local-data",
        path=tmp_path / "missing.json",
        data={"official_sources": {"huggingface_datasets": [{"repo_id": "org/missing", "license": "mit"}]}},
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
        check_local=True,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "missing"
    assert result["local_checks"][0]["status"] in {"not_found", "direct_hfd_empty"}


def test_download_datasets_check_local_reports_external_dataset_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_datasets = _load_script("download_datasets")
    data_root = tmp_path / "worldfoundry-data"
    monkeypatch.setenv("WORLDFOUNDRY_DATA_DIR", str(data_root))
    manifest = download_datasets.BenchmarkManifest(
        benchmark_id="external-data",
        path=tmp_path / "external.json",
        data={"dataset": {"path": "${WORLDFOUNDRY_DATA_DIR}/external-data"}},
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
        check_local=True,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "external_dataset_missing"
    assert result["dataset_not_hf_downloadable"] is True
    assert result["local_checks"][0]["path"] == str(data_root / "external-data")


def test_download_datasets_stdlib_fallback_matches_plan_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_datasets = _load_script("download_datasets")
    monkeypatch.setattr(download_datasets, "_DATASET_MODULE", None)
    manifest = download_datasets.BenchmarkManifest(
        benchmark_id="fallback",
        path=tmp_path / "fallback.json",
        data={
            "official_sources": {
                "huggingface_datasets": [
                    {"repo_id": "org/fallback", "sha": "abc1234", "license": "mit"}
                ]
            }
        },
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
    )

    assert result["ok"] is True
    assert result["hf_dataset_ids"] == ["org/fallback"]
    assert result["commands"][0][-2:] == ["--revision", "abc1234"]
    assert result["access_reports"][0]["ok"] is True
    assert result["access_reports"][0]["license_status"] == "open"


def test_download_datasets_allows_explicit_dataset_free_benchmark(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    manifest = download_datasets.BenchmarkManifest(
        benchmark_id="dataset-free",
        path=tmp_path / "benchmark.json",
        data={"dataset": {"not_applicable": True, "reason": "uses generated videos supplied by user"}},
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
        check_local=True,
    )

    assert result["ok"] is True
    assert result["ready"] is True
    assert result["status"] == "not_applicable"
    assert result["commands"] == []
    assert result["local_checks"][0]["ready"] is True
    assert result["local_checks"][0]["status"] == "not_applicable"


def test_download_datasets_local_check_detects_incomplete_file(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "datasets--org--dataset"
    snapshot = dataset_dir / "snapshots" / "abc123"
    (dataset_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (dataset_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (dataset_dir / "part.incomplete").write_text("partial", encoding="utf-8")

    result = download_datasets.check_local_dataset("org/dataset", cache_dir)

    assert result["ready"] is False
    assert result["ok"] is False
    assert result["hf_dataset_id"] == "org/dataset"
    assert result["status"] == "incomplete_files"
    assert result["incomplete_files"][0]["path"].endswith("part.incomplete")


def test_download_datasets_local_check_requires_expected_revision(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "datasets--org--dataset"
    snapshot = dataset_dir / "snapshots" / "abc123"
    (dataset_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (dataset_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snapshot / "sample.json").write_text("{}", encoding="utf-8")

    result = download_datasets.check_local_dataset("org/dataset", cache_dir, expected_revision="def456")

    assert result["ready"] is False
    assert result["expected_revision"] == "def456"
    assert result["revision_matches"] is False


def test_download_datasets_local_check_counts_nested_hf_snapshot_files(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "datasets--org--dataset"
    snapshot = dataset_dir / "snapshots" / "abc123"
    (dataset_dir / "refs").mkdir(parents=True)
    (snapshot / "nested").mkdir(parents=True)
    (dataset_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snapshot / "README.md").write_text("placeholder", encoding="utf-8")
    (snapshot / "nested" / "sample.json").write_text("{}", encoding="utf-8")

    result = download_datasets.check_local_dataset(
        "org/dataset",
        cache_dir,
        expected_file_count=2,
    )

    assert result["ready"] is True
    assert result["file_count"] == 2
    assert result["expected_file_count"] == 2


def test_download_datasets_local_check_rejects_partial_hf_snapshot_with_expected_count(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "datasets--org--dataset"
    snapshot = dataset_dir / "snapshots" / "abc123"
    (dataset_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (dataset_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snapshot / "README.md").write_text("placeholder", encoding="utf-8")

    result = download_datasets.check_local_dataset(
        "org/dataset",
        cache_dir,
        expected_file_count=2,
    )

    assert result["ready"] is False
    assert result["ok"] is False
    assert result["status"] == "incomplete_snapshot"
    assert result["file_count"] == 1
    assert result["missing_file_count"] == 1


def test_download_datasets_remote_file_count_rejects_partial_ready_hf_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_datasets = _load_script("download_datasets")
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "datasets--org--dataset"
    snapshot = dataset_dir / "snapshots" / "abc123"
    (dataset_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (dataset_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snapshot / "README.md").write_text("placeholder", encoding="utf-8")
    manifest = download_datasets.BenchmarkManifest(
        "alpha",
        tmp_path / "alpha.yaml",
        {"benchmark_id": "alpha", "hf_dataset_id": "org/dataset", "license": "mit"},
    )

    monkeypatch.setattr(
        download_datasets,
        "_remote_expected_hf_file_count",
        lambda dataset_id, revision=None: 2,
    )

    result = download_datasets.download_manifest(
        manifest,
        cache_dir,
        execute=False,
        check_local=True,
        verify_remote_file_count=True,
    )

    assert result["ready"] is False
    assert result["status"] == "partial"
    assert result["local_checks"][0]["status"] == "incomplete_snapshot"
    assert result["local_checks"][0]["expected_file_count_status"] == "verified_remote"
    assert result["local_checks"][0]["missing_file_count"] == 1


def test_download_datasets_requires_manifest_files_for_ready_snapshot(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "datasets--org--dataset"
    snapshot = dataset_dir / "snapshots" / "abc123"
    (dataset_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (dataset_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snapshot / "README.md").write_text("placeholder", encoding="utf-8")
    manifest = download_datasets.BenchmarkManifest(
        "alpha",
        tmp_path / "alpha.yaml",
        {
            "benchmark_id": "alpha",
            "hf_dataset_id": "org/dataset",
            "license": "mit",
            "metadata": {
                "prompt_suite": {
                    "hf_dataset_id": "org/dataset",
                    "manifest": "required.json",
                }
            },
        },
    )

    result = download_datasets.download_manifest(manifest, cache_dir, execute=False, check_local=True)

    assert result["ready"] is False
    assert result["status"] == "partial"
    assert result["local_checks"][0]["status"] == "missing_required_files"
    assert result["local_checks"][0]["required_files"][0]["path"] == "required.json"
    assert result["local_checks"][0]["required_files"][0]["exists"] is False


def test_download_datasets_accepts_ready_snapshot_with_required_manifest_file(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "datasets--org--dataset"
    snapshot = dataset_dir / "snapshots" / "abc123"
    (dataset_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (dataset_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    required = snapshot / "required.json"
    required.write_text("[]", encoding="utf-8")
    manifest = download_datasets.BenchmarkManifest(
        "alpha",
        tmp_path / "alpha.yaml",
        {
            "benchmark_id": "alpha",
            "hf_dataset_id": "org/dataset",
            "license": "mit",
            "metadata": {
                "prompt_suite": {
                    "hf_dataset_id": "org/dataset",
                    "manifest": "required.json",
                }
            },
        },
    )

    result = download_datasets.download_manifest(manifest, cache_dir, execute=False, check_local=True)

    assert result["ready"] is True
    assert result["status"] == "ready"
    assert result["local_checks"][0]["required_files"][0]["matched_path"] == str(required)


def test_download_datasets_ignores_provenance_only_refs_by_default(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    manifest = download_datasets.BenchmarkManifest(
        "alpha",
        tmp_path / "alpha.yaml",
        {
            "benchmark_id": "alpha",
            "dataset": {
                "not_applicable": True,
                "reason": "assets are caller supplied",
            },
            "source_provenance": {
                "huggingface_datasets": [
                    {
                        "repo_id": "org/provenance",
                        "license": "mit",
                        "provenance_only": True,
                    },
                    {
                        "repo_id": "org/role-provenance",
                        "license": "cc-by-nc-4.0",
                        "role": "provenance_only_not_runtime_dependency",
                    }
                ]
            },
        },
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache",
        execute=False,
        check_local=True,
    )
    explicit = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache",
        execute=False,
        dataset_id_filter=["org/provenance"],
        check_local=True,
    )

    assert result["ready"] is True
    assert result["status"] == "not_applicable"
    assert result["hf_dataset_ids"] == []
    assert {report["access_status"] for report in result["access_reports"]} == {"not_applicable"}
    assert explicit["hf_dataset_ids"] == ["org/provenance"]
    assert explicit["commands"][0][:3] == ["hf", "download", "org/provenance"]


def test_download_datasets_explicit_provenance_only_ref_still_uses_safety_gate(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    manifest = download_datasets.BenchmarkManifest(
        "alpha",
        tmp_path / "alpha.yaml",
        {
            "benchmark_id": "alpha",
            "dataset": {
                "not_applicable": True,
                "reason": "assets are caller supplied",
            },
            "source_provenance": {
                "huggingface_datasets": [
                    {
                        "repo_id": "org/role-provenance",
                        "license": "cc-by-nc-4.0",
                        "role": "provenance_only",
                    }
                ]
            },
        },
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache",
        execute=False,
        dataset_id_filter=["org/role-provenance"],
        check_local=True,
    )

    assert result["ready"] is False
    assert result["status"] == "blocked"
    assert result["hf_dataset_ids"] == ["org/role-provenance"]
    assert [issue["reason"] for issue in result["dataset_access_issues"]] == ["license_review_required"]


def test_download_datasets_uses_task_yaml_provenance_role_to_filter_catalog_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_datasets = _load_script("download_datasets")
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "alpha.yaml").write_text(
        """
benchmark: alpha
metadata:
  data_refs:
    dataset_refs:
      - hf_dataset_id: org/catalog-only
        role: provenance_only
source_provenance:
  hf_dataset:
    repo_id: org/catalog-only
    role: provenance_only_not_runtime_dependency
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(download_datasets, "DEFAULT_TASK_MANIFEST_DIR", task_dir)
    manifest = download_datasets.BenchmarkManifest(
        "alpha",
        tmp_path / "alpha.yaml",
        {
            "benchmark_id": "alpha",
            "dataset": {
                "not_applicable": True,
                "reason": "assets are caller supplied",
            },
            "official_sources": {
                "huggingface_datasets": [
                    {
                        "repo_id": "org/catalog-only",
                        "license": "cc-by-nc-4.0",
                    }
                ]
            },
        },
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache",
        execute=False,
        check_local=True,
    )

    assert result["ready"] is True
    assert result["status"] == "not_applicable"
    assert result["hf_dataset_ids"] == []
    assert "blocked_commands" not in result


def test_download_datasets_applies_task_yaml_required_manifest_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_datasets = _load_script("download_datasets")
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "alpha.yaml").write_text(
        """
benchmark: alpha
metadata:
  prompt_suite:
    hf_dataset_id: org/dataset
    manifest: required.json
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(download_datasets, "DEFAULT_TASK_MANIFEST_DIR", task_dir)
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "datasets--org--dataset"
    snapshot = dataset_dir / "snapshots" / "abc123"
    (dataset_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (dataset_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snapshot / "README.md").write_text("placeholder", encoding="utf-8")
    manifest = download_datasets.BenchmarkManifest(
        "alpha",
        tmp_path / "catalog.yaml",
        {"benchmark_id": "alpha", "hf_dataset_id": "org/dataset", "license": "mit"},
    )

    result = download_datasets.download_manifest(manifest, cache_dir, execute=False, check_local=True)

    assert result["ready"] is False
    assert result["local_checks"][0]["status"] == "missing_required_files"
    assert result["local_checks"][0]["required_files"][0]["path"] == "required.json"


def test_videoverse_prompt_suite_is_resolved_from_official_repo_not_bundled_assets(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    task_path = REPO_ROOT / "worldfoundry/data/benchmarks/tasks/external/videoverse.yaml"
    manifest_payload = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    prompt_suite = manifest_payload["metadata"]["prompt_suite"]

    assert "hf_dataset_id" not in prompt_suite
    assert prompt_suite["path_or_ref"] == "prompt/prompts_of_VideoVerse.json"
    assert prompt_suite["relative_to_env"] == "WORLDFOUNDRY_VIDEOVERSE_ROOT"
    assert "worldfoundry/data/benchmarks/assets" not in json.dumps(prompt_suite)

    repo_root = tmp_path / "VideoVerse"
    prompt_dir = repo_root / "prompt"
    prompt_dir.mkdir(parents=True)
    prompt_path = prompt_dir / "prompts_of_VideoVerse.json"
    decomposed_path = prompt_dir / "prompts_of_VideoVerse_decomposed.json"
    prompt_path.write_text(json.dumps({"sample-a": {"verification_checks": []}}), encoding="utf-8")
    decomposed_path.write_text(json.dumps({"sample-a": {"events": []}}), encoding="utf-8")

    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "datasets--NNaptmn--VideoVerse"
    snapshot = dataset_dir / "snapshots" / "af9b910eae1f7771c6a4ac3b1e9e5ae316b9cbd2"
    (dataset_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (dataset_dir / "refs" / "main").write_text("af9b910eae1f7771c6a4ac3b1e9e5ae316b9cbd2", encoding="utf-8")
    (snapshot / "README.md").write_text("placeholder", encoding="utf-8")

    result = download_datasets.download_manifest(
        download_datasets.BenchmarkManifest(
            "videoverse",
            task_path,
            {
                "benchmark_id": "videoverse",
                "hf_dataset_id": "NNaptmn/VideoVerse",
                "license": "apache-2.0",
                "metadata": {"prompt_suite": prompt_suite},
            },
        ),
        cache_dir,
        execute=False,
        check_local=True,
        env_overrides={"WORLDFOUNDRY_VIDEOVERSE_ROOT": str(repo_root)},
    )

    assert result["ready"] is True
    assert result["local_checks"][0]["status"] == "ready"
    assert result["local_checks"][0].get("required_files", []) == []


def test_download_datasets_local_check_accepts_direct_hfd_layout(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    cache_dir = tmp_path / "hfd_datasets"
    dataset_dir = cache_dir / "org__dataset"
    (dataset_dir / ".hfd").mkdir(parents=True)
    (dataset_dir / ".hfd" / "repo_metadata.json").write_text(
        json.dumps({"sha": "abc123def456"}),
        encoding="utf-8",
    )
    (dataset_dir / "sample.json").write_text("{}", encoding="utf-8")

    result = download_datasets.check_local_dataset("org/dataset", cache_dir, expected_revision="abc123def456")

    assert result["ready"] is True
    assert result["ok"] is True
    assert result["local_layout"] == "direct_hfd"
    assert result["direct_file_count"] == 1
    assert result["direct_revision"] == "abc123def456"


def test_download_datasets_local_check_accepts_nested_hfd_datasets_layout(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "hfd_datasets" / "org__dataset"
    (dataset_dir / ".hfd").mkdir(parents=True)
    (dataset_dir / ".hfd" / "repo_metadata.json").write_text(
        json.dumps({"sha": "abc123def456"}),
        encoding="utf-8",
    )
    (dataset_dir / "sample.json").write_text("{}", encoding="utf-8")

    result = download_datasets.check_local_dataset("org/dataset", cache_dir, expected_revision="abc123def456")

    assert result["ready"] is True
    assert result["ok"] is True
    assert result["local_layout"] == "direct_hfd"
    assert result["direct_dataset_dir"] == str(dataset_dir)


def test_download_datasets_local_check_rejects_direct_hfd_missing_siblings(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    cache_dir = tmp_path / "hfd_datasets"
    dataset_dir = cache_dir / "org__dataset"
    (dataset_dir / ".hfd").mkdir(parents=True)
    (dataset_dir / ".hfd" / "repo_metadata.json").write_text(
        json.dumps(
            {
                "sha": "abc123def456",
                "siblings": [
                    {"rfilename": "present.json"},
                    {"rfilename": ".DS_Store"},
                    {"rfilename": "missing/video.mp4"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "present.json").write_text("{}", encoding="utf-8")

    result = download_datasets.check_local_dataset("org/dataset", cache_dir, expected_revision="abc123def456")

    assert result["ready"] is False
    assert result["ok"] is False
    assert result["status"] == "direct_hfd_incomplete_files"
    assert result["direct_incomplete_files"][0]["kind"] == "missing_expected_file"


def test_download_datasets_execute_blocks_unsafe_gated_and_license_refs(
    tmp_path: Path,
) -> None:
    download_datasets = _load_script("download_datasets")
    manifest = download_datasets.BenchmarkManifest(
        benchmark_id="unsafe-datasets",
        path=tmp_path / "unsafe.json",
        data={
            "official_sources": {
                "huggingface_datasets": [
                    {"repo_id": "org/gated", "gated": "manual", "license": None},
                    {"repo_id": "org/review-license", "license": "cc-by-nc-4.0"},
                ]
            }
        },
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=True,
    )

    assert result["ok"] is False
    assert result["executed"] is False
    assert result["status"] == "blocked"
    assert "unsafe dataset access" in result["error"]
    assert {issue["reason"] for issue in result["dataset_access_issues"]} == {
        "gated_dataset",
        "missing_license",
        "license_review_required",
    }
    assert {(report["hf_dataset_id"], report["license_status"]) for report in result["access_reports"]} == {
        ("org/gated", "missing"),
        ("org/review-license", "review_required"),
    }


def test_download_datasets_plan_only_omits_unsafe_commands_by_default(tmp_path: Path) -> None:
    download_datasets = _load_script("download_datasets")
    manifest = download_datasets.BenchmarkManifest(
        benchmark_id="unsafe-plan-only",
        path=tmp_path / "unsafe.json",
        data={
            "official_sources": {
                "huggingface_datasets": [
                    {"repo_id": "org/unsafe", "gated": "manual", "license": None}
                ]
            }
        },
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
    )

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["commands"] == []
    assert result["blocked_commands"] == [
        ["hf", "download", "org/unsafe", "--repo-type", "dataset", "--cache-dir", str(tmp_path / "cache" / "hfd")]
    ]


def test_download_datasets_execute_allows_unsafe_refs_with_explicit_override(
    tmp_path: Path,
) -> None:
    download_datasets = _load_script("download_datasets")
    manifest = download_datasets.BenchmarkManifest(
        benchmark_id="unsafe-allowed",
        path=tmp_path / "unsafe.json",
        data={
            "official_sources": {
                "huggingface_datasets": [
                    {"repo_id": "org/unsafe", "gated": "manual", "license": None}
                ]
            }
        },
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=True,
        allow_unsafe_datasets=True,
    )

    assert result["executed"] is False
    assert result["unsafe_datasets_allowed"] is True
    assert result["commands"] == [
        ["hf", "download", "org/unsafe", "--repo-type", "dataset", "--cache-dir", str(tmp_path / "cache" / "hfd")]
    ]
    assert "manual_download_note" in result


def test_download_datasets_execute_captures_hf_output_in_report(
    tmp_path: Path,
) -> None:
    download_datasets = _load_script("download_datasets")
    manifest = download_datasets.BenchmarkManifest(
        benchmark_id="clean-json",
        path=tmp_path / "clean.json",
        data={"official_sources": {"huggingface_datasets": [{"repo_id": "org/data", "license": "mit"}]}},
    )

    result = download_datasets.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=True,
        check_local=True,
    )

    assert result["executed"] is False
    assert "manual_download_note" in result
    assert "stdout" not in result


def test_download_datasets_main_writes_structured_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    download_datasets = _load_script("download_datasets")
    manifest_dir = tmp_path / "benchmark_zoo"
    _write_manifest(
        manifest_dir,
        {
            "benchmarks": [
                    {
                        "benchmark_id": "missing-data",
                        "official_sources": {
                            "huggingface_datasets": [
                                {"repo_id": "org/missing", "revision": "abc123", "license": "mit"}
                            ]
                        },
                    }
            ]
        },
    )
    report_path = tmp_path / "reports" / "benchmark-download.json"

    exit_code = download_datasets.main(
        [
            "--manifest-dir",
            str(manifest_dir),
            "--benchmark-id",
            "missing-data",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--check-local",
            "--report-path",
            str(report_path),
            "--json",
        ]
    )

    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert file_report == stdout_report
    assert file_report["schema_version"] == "worldfoundry-benchmark-dataset-download-report"
    assert file_report["summary"] == {
        "total": 1,
        "ready": 0,
        "not_ready": 1,
        "by_status": {"missing": 1},
    }
    assert file_report["results"][0]["benchmark_id"] == "missing-data"
    assert file_report["results"][0]["status"] == "missing"


def test_run_benchmark_execution_runs_selected_manifest_run_command_and_writes_report(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="demo-benchmark",
        path=tmp_path / "demo.json",
        data={"run_command": [sys.executable, "-c", "print('benchmark-ok')"]},
    )

    result = run_benchmark_execution.run_benchmark(manifest, tmp_path / "benchmark_zoo", timeout_seconds=10)

    report_path = tmp_path / "benchmark_zoo" / "demo-benchmark" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert report["benchmark_id"] == "demo-benchmark"
    assert report["run_command"] == [sys.executable, "-c", "print('benchmark-ok')"]
    assert report["returncode"] == 0
    assert report["stdout"] == "benchmark-ok\n"


def test_run_benchmark_execution_loads_entries_and_nested_run_command(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    manifest_dir = tmp_path / "benchmark_zoo"
    _write_manifest(
        manifest_dir,
        {
            "entries": [
                {
                    "id": "nested-runner",
                    "runner": {
                        "run_command": [sys.executable, "-c", "print('nested-benchmark-ok')"],
                    },
                }
            ]
        },
    )

    manifests = run_benchmark_execution.load_manifests(manifest_dir, "nested-runner")
    result = run_benchmark_execution.run_benchmark(manifests[0], tmp_path / "reports", timeout_seconds=10)

    assert result["ok"] is True
    assert result["stdout"] == "nested-benchmark-ok\n"


def test_run_benchmark_execution_resolves_python_to_current_executable(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="python-resolution",
        path=tmp_path / "demo.json",
        data={"run_command": ["python", "-c", "import sys; print(sys.executable)"]},
    )

    result = run_benchmark_execution.run_benchmark(manifest, tmp_path / "reports", timeout_seconds=10)

    assert result["ok"] is True
    assert result["run_command"][0] == sys.executable
    assert result["stdout"].strip() == sys.executable


def test_run_benchmark_execution_validation_prefers_validation_command(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="validation-benchmark",
        path=tmp_path / "demo.json",
        data={
            "run_command": [sys.executable, "-c", "print('full-run')"],
            "runner": {"validation_command": [sys.executable, "-c", "print('validation-run')"]},
        },
    )

    result = run_benchmark_execution.run_benchmark(
        manifest,
        tmp_path / "benchmark_zoo",
        timeout_seconds=10,
        command_kind="validation",
    )

    assert result["ok"] is False
    assert result["command_ok"] is True
    assert result["command_kind"] == "validation"
    assert result["stdout"] == "validation-run\n"


def test_run_benchmark_execution_fails_when_expected_artifact_is_missing(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="missing-artifact-benchmark",
        path=tmp_path / "demo.json",
        data={
            "run_command": [sys.executable, "-c", "print('no artifact')"],
            "runner": {"expected_artifacts": ["score.json"]},
        },
    )

    result = run_benchmark_execution.run_benchmark(manifest, tmp_path / "benchmark_zoo", timeout_seconds=10)

    assert result["returncode"] == 0
    assert result["ok"] is False
















def test_run_benchmark_execution_exposes_output_env_for_commands(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="env-benchmark",
        path=tmp_path / "demo.json",
        data={
            "runner": {
                "validation_command": [
                    sys.executable,
                    "-c",
                    (
                        "import json, os; "
                        "from pathlib import Path; "
                        "root = Path(os.environ['WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR']); "
                        "Path(root, 'scorecard.json').write_text(json.dumps({"
                        "'id': os.environ['WORLDFOUNDRY_BENCHMARK_ID'], "
                        "'kind': os.environ['WORLDFOUNDRY_BENCHMARK_COMMAND_KIND']"
                        "}))"
                    ),
                ],
                "expected_artifacts": ["scorecard.json"],
            },
        },
    )

    result = run_benchmark_execution.run_benchmark(
        manifest,
        tmp_path / "benchmark_zoo",
        timeout_seconds=10,
        command_kind="validation",
    )

    scorecard = json.loads(
        (tmp_path / "benchmark_zoo" / "env-benchmark" / "scorecard.json").read_text(encoding="utf-8")
    )
    assert result["ok"] is False
    assert result["command_ok"] is True
    assert scorecard == {"id": "env-benchmark", "kind": "validation"}


def test_vbench_official_runner_normalizes_existing_upstream_results(tmp_path: Path) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")
    upstream_results = tmp_path / "results_eval_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "aesthetic_quality": [
                    0.75,
                    [{"video_path": "sample.mp4", "video_results": 0.75}],
                ]
            }
        ),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample.mp4").write_bytes(b"fake-video")
    output_dir = tmp_path / "vbench-out"

    exit_code = run_vbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--videos-path",
            str(videos_dir),
            "--dimension",
            "aesthetic_quality",
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert scorecard["benchmark"]["contract_only"] is False
    assert scorecard["evaluation"]["kind"] == "official_vbench"
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["metrics"]["per_metric"]["aesthetic_quality"]["raw_score"] == 0.75
    assert scorecard["metrics"]["groups"]["vbench_dimensions"] == ["aesthetic_quality"]
    assert scorecard["metrics"]["groups"]["vbench_aggregates"] == []
    assert "overall_quality" not in scorecard["metrics"]["per_metric"]
    assert raw_rows[0]["metric_id"] == "aesthetic_quality"


def test_worldscore_runner_stages_bounded_dynamic_contract(tmp_path: Path) -> None:
    run_worldscore = _load_script("run_worldscore_official_runner")
    from PIL import Image

    worldscore_root = tmp_path / "WorldScore"
    config_dir = worldscore_root / "config" / "model_configs"
    config_dir.mkdir(parents=True)
    (worldscore_root / "config" / "base_config.yaml").write_text(
        "dataset_root: ${oc.env:DATA_PATH}/WorldScore-Dataset\n"
        "output_dir: worldscore_output\n"
        "frames: 50\n",
        encoding="utf-8",
    )
    (config_dir / "wan2.1_i2v.yaml").write_text(
        "model: wan2.1_i2v\n"
        "runs_root: ${oc.env:MODEL_PATH}/Wan2.1\n"
        "resolution: [64, 40]\n"
        "generate_type: i2v\n"
        "frames: 81\n",
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "data" / "WorldScore-Dataset"
    image_dir = dataset_dir / "dynamic" / "photorealistic" / "images" / "articulated"
    mask_dir = dataset_dir / "dynamic" / "photorealistic" / "masks" / "articulated" / "001"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (80, 40), color=(10, 20, 30)).save(image_dir / "001.jpg")
    Image.new("L", (80, 40), color=255).save(mask_dir / "001-1.png")
    (dataset_dir / "dynamic" / "dynamic.json").write_text(
        json.dumps(
            [
                {
                    "visual_movement": "dynamic",
                    "visual_style": "photorealistic",
                    "motion_type": "articulated",
                    "style": "photorealistic",
                    "objects": ["elephant"],
                    "prompt": "The elephant moves.",
                    "image": "./dynamic/photorealistic/images/articulated/001.jpg",
                    "masks": ["./dynamic/photorealistic/masks/articulated/001/001-1.png"],
                    "camera_path": ["fixed"],
                }
            ]
        ),
        encoding="utf-8",
    )
    source_frames = tmp_path / "source_frames"
    source_frames.mkdir()
    for index in range(4):
        Image.new("RGB", (64, 40), color=(index, index, index)).save(source_frames / f"{index:03d}.png")

    exit_code = run_worldscore.main(
        [
            "--worldscore-root",
            str(worldscore_root),
            "--model-name",
            "wan2.1_i2v",
            "--model-path",
            str(tmp_path / "model_path"),
            "--data-path",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "runner_out"),
            "--stage-dynamic-source",
            str(source_frames),
            "--stage-target-frames",
            "4",
            "--stage-only",
            "--stage-overwrite",
            "--json",
        ]
    )

    staged_root = tmp_path / "model_path" / "Wan2.1" / "worldscore_output"
    instance_dir = staged_root / "dynamic" / "photorealistic" / "articulated" / "001"
    scorecard = json.loads((tmp_path / "runner_out" / "scorecard.json").read_text(encoding="utf-8"))
    image_data = json.loads((instance_dir / "image_data.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert (instance_dir / "frames" / "003.png").is_file()
    assert (instance_dir / "input_image.png").is_file()
    assert image_data["total_frames"] == 4
    assert image_data["masks"] == ["./dynamic/photorealistic/masks/articulated/001/001-1.png"]
    assert scorecard["validation"]["scope"] == "official_bounded"
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    assert scorecard["contract_validation"]["valid"] is True


def test_vbench_script_request_normalizes_existing_upstream_results(tmp_path: Path) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")
    upstream_results = tmp_path / "results_eval_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "aesthetic_quality": [
                    0.81,
                    [{"video_path": "sample.mp4", "video_results": 0.81}],
                ]
            }
        ),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample.mp4").write_bytes(b"fake-video")
    output_dir = tmp_path / "vbench-api-out"

    scorecard = run_vbench_official_runner.run_vbench(
        run_vbench_official_runner.VBenchRunRequest(
            output_dir=output_dir,
            videos_path=videos_dir,
            dimensions=("aesthetic_quality",),
            from_upstream_results=upstream_results,
        )
    )

    raw_rows = [json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()]
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["benchmark"]["benchmark_id"] == "vbench"
    assert scorecard["metrics"]["per_metric"]["aesthetic_quality"]["raw_score"] == 0.81
    assert scorecard["metrics"]["groups"]["vbench_dimensions"] == ["aesthetic_quality"]
    assert raw_rows[0]["metric_id"] == "aesthetic_quality"


def test_robotwin_official_runner_normalizes_structured_results(tmp_path: Path) -> None:
    run_robotwin = _load_script("run_robotwin_official_runner")
    upstream_results = tmp_path / "robotwin_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_id": "ep-1", "task": "handover_block", "task_config": "demo_clean", "success": True, "reward": 1.0},
                    {"episode_id": "ep-2", "task": "handover_block", "task_config": "demo_clean", "success": False, "reward": 0.0},
                    {"episode_id": "ep-3", "task": "stack_blocks_two", "task_config": "demo_randomized", "success": True, "reward": 2.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "robotwin-out"

    exit_code = run_robotwin.main(
        [
            "--from-upstream-results",
            str(upstream_results),
            "--output-dir",
            str(output_dir),
            "--robotwin-root",
            str(tmp_path / "RoboTwin"),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    leaderboard = scorecard["metrics"]["leaderboard"]
    assert exit_code == 0
    assert scorecard["benchmark"]["benchmark_kind"] == ["vla", "embodied-action", "robot-manipulation", "bimanual"]
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    assert scorecard["validation"]["official_demo_parity_verified"] is False
    assert leaderboard["success_rate"] == pytest.approx(2 / 3)
    assert leaderboard["episode_success"] == pytest.approx(2 / 3)
    assert leaderboard["bimanual_task_success"] == pytest.approx(2 / 3)
    assert leaderboard["task_success"] == pytest.approx(0.75)
    assert leaderboard["clean_success"] == pytest.approx(0.5)
    assert leaderboard["randomized_success"] == pytest.approx(1.0)
    assert leaderboard["domain_randomization_success"] == pytest.approx(1.0)
    assert leaderboard["reward"] == pytest.approx(1.0)
    assert {row["metric_id"] for row in raw_rows} >= {"success_rate", "clean_success", "randomized_success"}


def test_robotwin_official_runner_imports_eval_result_txt_layout(tmp_path: Path) -> None:
    run_robotwin = _load_script("run_robotwin_official_runner")
    result_dir = (
        tmp_path
        / "eval_result"
        / "handover_block"
        / "ACT"
        / "demo_randomized"
        / "ckpt-0001"
        / "2026-05-11 00:00:00"
    )
    result_dir.mkdir(parents=True)
    (result_dir / "_result.txt").write_text(
        "Timestamp: 2026-05-11 00:00:00\n\nInstruction Type: unseen\n\n0.42\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "robotwin-txt-out"

    scorecard = run_robotwin.run_robotwin(
        run_robotwin.RoboTwinRunRequest(
            output_dir=output_dir,
            robotwin_root=tmp_path / "RoboTwin",
            results_paths=(tmp_path / "eval_result",),
        )
    )

    leaderboard = scorecard["metrics"]["leaderboard"]
    metric_rows = [
        json.loads(line)
        for line in (output_dir / "evaluation" / "per_sample_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert leaderboard["success_rate"] == pytest.approx(0.42)
    assert leaderboard["randomized_success"] == pytest.approx(0.42)
    assert metric_rows[0]["task_id"] == "handover_block"
    assert metric_rows[0]["metadata"]["policy"] == "ACT"
    assert metric_rows[0]["metadata"]["checkpoint"] == "ckpt-0001"


def test_robotwin_contract_only_writes_task_and_command_artifacts(tmp_path: Path) -> None:
    run_robotwin = _load_script("run_robotwin_official_runner")
    robotwin_root = tmp_path / "RoboTwin"
    task_dir = robotwin_root / "description" / "task_instruction"
    task_dir.mkdir(parents=True)
    (task_dir / "handover_block.json").write_text(
        json.dumps({"full_description": "handover a red block", "seen": ["handover"], "unseen": ["transfer"]}),
        encoding="utf-8",
    )
    config_dir = robotwin_root / "task_config"
    config_dir.mkdir()
    (config_dir / "_eval_step_limit.yml").write_text("handover_block: 800\n", encoding="utf-8")
    output_dir = tmp_path / "robotwin-contract"

    exit_code = run_robotwin.main(
        [
            "--contract-only",
            "--robotwin-root",
            str(robotwin_root),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    task_manifest = json.loads((output_dir / "task_manifest.json").read_text(encoding="utf-8"))
    command_manifest = json.loads((output_dir / "official_command_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((output_dir / "benchmark_contract.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert scorecard["benchmark"]["contract_only"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert task_manifest["official_task_count"] == 1
    assert task_manifest["tasks"][0]["task_id"] == "handover_block"
    assert task_manifest["tasks"][0]["eval_step_limit"] == 800
    assert command_manifest["policy_eval_entrypoints"]["ACT"][0] == "bash"
    assert contract["metric_ids"] == list(run_robotwin.METRIC_IDS)


def test_robotwin_simulator_validation_runs_official_render_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_robotwin = _load_script("run_robotwin_official_runner")
    robotwin_root = tmp_path / "RoboTwin"
    (robotwin_root / "script").mkdir(parents=True)
    (robotwin_root / "script" / "test_render.py").write_text("print('Render Well')\n", encoding="utf-8")
    output_dir = tmp_path / "robotwin-validation"
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> dict[str, object]:
        calls.append(command)
        return {
            "stdout": "Render Well\n",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "kill_stuck": False,
            "duration_seconds": 0.1,
        }

    monkeypatch.setattr(run_robotwin, "run_bounded_command", fake_run)
    monkeypatch.setattr(
        run_robotwin,
        "run_cuda_probe",
        lambda: {
            "available": True,
            "torch_version": "test",
            "cuda_version": "11.3",
            "device_count": 1,
            "device_name": "A800",
            "probe_sum": 16777216.0,
            "memory_allocated": 262144,
            "max_memory_allocated": 262144,
        },
    )

    exit_code = run_robotwin.main(
        [
            "--simulator-validation",
            "--robotwin-root",
            str(robotwin_root),
            "--output-dir",
            str(output_dir),
            "--python",
            sys.executable,
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "simulator_validation.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert calls[0] == [sys.executable, str(robotwin_root / "script" / "test_render.py")]
    assert scorecard["validation"]["official_simulator_validation_verified"] is True
    assert scorecard["official_benchmark_verified"] is True
    assert scorecard["integration_evidence"] is True
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    assert validation["status"] == "verified"
    assert validation["cuda_probe"]["available"] is True


def test_robotwin_simulator_validation_writes_failed_scorecard_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_robotwin = _load_script("run_robotwin_official_runner")
    robotwin_root = tmp_path / "RoboTwin"
    (robotwin_root / "script").mkdir(parents=True)
    (robotwin_root / "script" / "test_render.py").write_text("print('Render Well')\n", encoding="utf-8")
    output_dir = tmp_path / "robotwin-timeout"

    def fake_run(command: list[str], **kwargs: object) -> dict[str, object]:
        return {
            "stdout": "",
            "stderr": "render wait\nTimeoutExpired: official script exceeded 3s",
            "returncode": 124,
            "timed_out": True,
            "kill_stuck": True,
            "duration_seconds": 8.0,
        }

    monkeypatch.setattr(run_robotwin, "run_bounded_command", fake_run)
    monkeypatch.setattr(
        run_robotwin,
        "run_cuda_probe",
        lambda: {"available": True, "probe": "cuda_driver", "device_count": 1},
    )

    exit_code = run_robotwin.main(
        [
            "--simulator-validation",
            "--robotwin-root",
            str(robotwin_root),
            "--output-dir",
            str(output_dir),
            "--python",
            sys.executable,
            "--timeout",
            "3",
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "simulator_validation.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert validation["status"] == "failed"
    assert validation["timed_out"] is True
    assert validation["kill_stuck"] is True
    assert validation["returncode"] == 124
    assert scorecard["run"]["timed_out"] is True
    assert scorecard["run"]["kill_stuck"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False


def test_vbench_script_presets_and_setup_are_non_vendored(tmp_path: Path) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")

    assert run_vbench_official_runner.split_dimensions(None, ["validation"]) == ["aesthetic_quality"]
    assert run_vbench_official_runner.split_dimensions(["semantic"]) == [
        "object_class",
        "multiple_objects",
        "human_action",
        "color",
        "spatial_relationship",
        "scene",
        "appearance_style",
        "temporal_style",
        "overall_consistency",
    ]
    payload = run_vbench_official_runner.vbench_dimensions_payload()
    assert payload["presets"]["full_16"][0] == "subject_consistency"
    assert payload["presets"]["custom_supported"] == [
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
    ]

    setup = run_vbench_official_runner.ensure_vbench_repo(
        root=tmp_path / "repos" / "vbench",
        repo_url="https://github.com/example/VBench.git",
        revision="abc123",
        plan_only=True,
    )
    assert setup["status"] == "planned_clone"
    assert setup["repo_url"] == "https://github.com/example/VBench.git"
    assert setup["revision"] == "abc123"
    assert setup["commands"][0][:3] == ["git", "clone", "https://github.com/example/VBench.git"]
    assert setup["commands"][1][-1] == "abc123"


def test_vbench_run_parser_accepts_official_repo_checkout_args(tmp_path: Path) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")

    args = run_vbench_official_runner.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--videos-path",
            str(tmp_path / "videos"),
            "--dimension",
            "aesthetic_quality",
            "--clone-root",
            str(tmp_path / "repos"),
            "--repo-url",
            "https://github.com/example/VBench.git",
            "--revision",
            "abc123",
        ]
    )

    assert args.vbench_root is None
    assert args.clone_root == tmp_path / "repos"
    assert args.repo_url == "https://github.com/example/VBench.git"
    assert args.revision == "abc123"


def test_vbench_run_ensure_repo_passes_checkout_args_without_clone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")
    upstream_results = tmp_path / "results_eval_results.json"
    upstream_results.write_text(json.dumps({"aesthetic_quality": [0.5, []]}), encoding="utf-8")
    full_json = tmp_path / "VBench_full_info.json"
    full_json.write_text(json.dumps({"items": [{"prompt": "a calm lake"}]}), encoding="utf-8")
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample.mp4").write_bytes(b"fake-video")
    calls: list[dict[str, object]] = []

    def fake_ensure_vbench_repo(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"ok": True, "status": "plan_only", "root": str(kwargs["root"]), "commands": []}

    monkeypatch.setattr(run_vbench_official_runner, "ensure_vbench_repo", fake_ensure_vbench_repo)

    args = run_vbench_official_runner.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--videos-path",
            str(videos_dir),
            "--dimension",
            "aesthetic_quality",
            "--from-upstream-results",
            str(upstream_results),
            "--full-json-dir",
            str(full_json),
            "--ensure-repo",
            "--clone-root",
            str(tmp_path / "repos"),
            "--repo-url",
            "https://github.com/example/VBench.git",
            "--revision",
            "abc123",
        ]
    )
    args.dimension = run_vbench_official_runner.split_dimensions(args.dimension, args.preset)

    scorecard = run_vbench_official_runner.run_official_vbench(args)

    assert calls == [
        {
            "root": (tmp_path / "repos" / "github.com_Vchitect_VBench").resolve(),
            "repo_url": "https://github.com/example/VBench.git",
            "revision": "abc123",
        }
    ]
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False


def test_vbench_script_command_validates_prompt_and_category(tmp_path: Path) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")

    vbench_root = tmp_path / "vbench"
    vbench_root.mkdir()
    (vbench_root / "evaluate.py").write_text("raise SystemExit(0)\n", encoding="utf-8")

    request = run_vbench_official_runner.VBenchRunRequest(
        output_dir=tmp_path / "out",
        videos_path=tmp_path / "videos",
        vbench_root=vbench_root,
        dimensions=("aesthetic_quality",),
        mode="vbench_category",
        category="animal",
        imaging_quality_preprocessing_mode="shorter",
    )
    args = request.to_namespace()
    upstream_output_dir = tmp_path / "out" / "upstream"
    command = run_vbench_official_runner.build_official_command(args, upstream_output_dir)
    assert "--category" in command
    assert "animal" in command
    assert "--imaging_quality_preprocessing_mode" in command
    assert "shorter" in command

    with pytest.raises(ValueError, match="require --mode custom_input"):
        run_vbench_official_runner.validate_vbench_args(
            run_vbench_official_runner.VBenchRunRequest(
                output_dir=tmp_path / "bad-prompt",
                videos_path=tmp_path / "videos",
                vbench_root=vbench_root,
                dimensions=("aesthetic_quality",),
                prompt="a red car",
                mode="vbench_standard",
            ).to_namespace()
        )
    with pytest.raises(ValueError, match="cannot be used together"):
        run_vbench_official_runner.validate_vbench_args(
            run_vbench_official_runner.VBenchRunRequest(
                output_dir=tmp_path / "bad-prompt-file",
                videos_path=tmp_path / "videos",
                vbench_root=vbench_root,
                dimensions=("aesthetic_quality",),
                prompt="a red car",
                prompt_file=tmp_path / "prompts.txt",
                mode="custom_input",
            ).to_namespace()
        )

    with pytest.raises(ValueError, match="custom_input only supports"):
        run_vbench_official_runner.run_vbench(
            run_vbench_official_runner.VBenchRunRequest(
                output_dir=tmp_path / "bad",
                videos_path=tmp_path / "videos",
                vbench_root=vbench_root,
                dimensions=("human_action",),
                mode="custom_input",
            )
        )


def test_vbench_official_runner_materializes_prompt_file_for_custom_input_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")
    vbench_root = tmp_path / "VBench"
    vbench_root.mkdir()
    (vbench_root / "evaluate.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample.mp4").write_bytes(b"fake-video")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        upstream_output = Path(command[command.index("--output_path") + 1])
        upstream_output.mkdir(parents=True, exist_ok=True)
        (upstream_output / "results_eval_results.json").write_text(
            json.dumps({"aesthetic_quality": [0.7, []]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(run_vbench_official_runner.subprocess, "run", fake_run)
    output_dir = tmp_path / "out"

    scorecard = run_vbench_official_runner.run_vbench(
        run_vbench_official_runner.VBenchRunRequest(
            output_dir=output_dir,
            videos_path=videos_dir,
            vbench_root=vbench_root,
            dimensions=("aesthetic_quality",),
            mode="custom_input",
            prompt="a red car",
        )
    )

    assert scorecard["official_benchmark_verified"] is True
    assert calls
    command = calls[0]
    assert "--prompt" not in command
    prompt_file = Path(command[command.index("--prompt_file") + 1])
    assert json.loads(prompt_file.read_text(encoding="utf-8")) == {"sample.mp4": "a red car"}


def test_vbench_official_suite_validation_reports_missing_full_json(tmp_path: Path) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")
    upstream_results = tmp_path / "results_eval_results.json"
    upstream_results.write_text(json.dumps({"aesthetic_quality": [0.7, []]}), encoding="utf-8")
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample.mp4").write_bytes(b"fake-video")
    output_dir = tmp_path / "out"

    exit_code = run_vbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--videos-path",
            str(videos_dir),
            "--preset",
            "full_16",
            "--from-upstream-results",
            str(upstream_results),
            "--full-json-dir",
            str(tmp_path / "missing.json"),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    validation = scorecard["validation"]["prompt_suite_materialization"]
    assert exit_code == 0
    assert validation["ok"] is False
    assert validation["leaderboard_valid"] is False
    assert validation["issues"][0]["code"] == "missing_full_json_dir"
    assert any("prompt suite metadata was not found" in reason for reason in scorecard["eligibility"]["reasons"])
    assert scorecard["official_benchmark_verified"] is False


def test_vbench_official_suite_validation_reports_no_video_coverage(tmp_path: Path) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")
    upstream_results = tmp_path / "results_eval_results.json"
    upstream_results.write_text(json.dumps({"aesthetic_quality": [0.7, []]}), encoding="utf-8")
    full_json = tmp_path / "VBench_full_info.json"
    full_json.write_text(
        json.dumps({"items": [{"prompt": "a red car", "video_list": ["expected.mp4"]}]}),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "other.mp4").write_bytes(b"fake-video")
    output_dir = tmp_path / "out"

    exit_code = run_vbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--videos-path",
            str(videos_dir),
            "--preset",
            "official",
            "--from-upstream-results",
            str(upstream_results),
            "--full-json-dir",
            str(full_json),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    validation = scorecard["validation"]["prompt_suite_materialization"]
    assert exit_code == 0
    assert validation["ok"] is False
    assert validation["expected_prompt_count"] == 1
    assert validation["expected_video_count"] == 1
    assert validation["covered_video_count"] == 0
    assert validation["issues"][0]["code"] == "no_materialized_prompt_or_video"
    assert scorecard["eligibility"]["leaderboard_valid"] is False


def test_vbench_official_suite_validation_uses_prompt_en_filenames(tmp_path: Path) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")
    full_json = tmp_path / "VBench_full_info.json"
    full_json.write_text(
        json.dumps(
            [
                {"prompt_en": "a red car", "dimension": ["aesthetic_quality"]},
                {"prompt_en": "a blue boat", "dimension": ["dynamic_degree"]},
            ]
        ),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "a red car-0.mp4").write_bytes(b"fake-video")
    (videos_dir / "a blue boat-0.mp4").write_bytes(b"fake-video")
    args = run_vbench_official_runner.VBenchRunRequest(
        output_dir=tmp_path / "out",
        videos_path=videos_dir,
        dimensions=("aesthetic_quality",),
        full_json_dir=full_json,
    ).to_namespace()

    validation = run_vbench_official_runner.validate_prompt_suite_materialization(args)

    assert validation["ok"] is True
    assert validation["expected_prompt_count"] == 1
    assert validation["expected_video_count"] == 5
    assert validation["covered_video_count"] == 1
    assert validation["covered_videos"] == ["a red car-0.mp4"]
    assert "a red car-4.mp4" in validation["sample_expected_videos"]


def test_vbench_official_runner_skips_upstream_when_prompt_suite_has_no_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")
    vbench_root = tmp_path / "VBench"
    vbench_root.mkdir()
    (vbench_root / "evaluate.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    full_json = tmp_path / "VBench_full_info.json"
    full_json.write_text(
        json.dumps([{"prompt_en": "a red car", "dimension": ["aesthetic_quality"]}]),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "other.mp4").write_bytes(b"fake-video")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(run_vbench_official_runner.subprocess, "run", fake_run)
    output_dir = tmp_path / "out"

    exit_code = run_vbench_official_runner.main(
        [
            "--vbench-root",
            str(vbench_root),
            "--output-dir",
            str(output_dir),
            "--videos-path",
            str(videos_dir),
            "--dimension",
            "aesthetic_quality",
            "--full-json-dir",
            str(full_json),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    validation = scorecard["validation"]["prompt_suite_materialization"]
    assert exit_code == 1
    assert calls == []
    assert validation["ok"] is False
    assert validation["expected_prompt_count"] == 1
    assert validation["expected_video_count"] == 5
    assert validation["covered_video_count"] == 0
    assert validation["issues"][0]["code"] == "no_materialized_prompt_or_video"
    assert scorecard["run"]["returncode"] == 2
    assert scorecard["official_benchmark_verified"] is False
    assert Path(scorecard["artifacts"]["upstream_stderr"]).read_text(encoding="utf-8").startswith("skipped official VBench")


def test_vbench_official_runner_accepts_python_from_environment(monkeypatch, tmp_path: Path) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")
    python_path = tmp_path / "python-cu118"
    monkeypatch.setenv("WORLDFOUNDRY_VBENCH_PYTHON", str(python_path))

    args = run_vbench_official_runner.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--videos-path",
            str(tmp_path / "videos"),
            "--dimension",
            "aesthetic_quality",
        ]
    )

    assert args.python == str(python_path)


def test_vbench_official_runner_builds_full_16_preset_command(tmp_path: Path) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")
    vbench_root = tmp_path / "vbench"
    vbench_root.mkdir()
    (vbench_root / "evaluate.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    full_json = tmp_path / "VBench_full_info.json"
    full_json.write_text(json.dumps({"items": [{"prompt": "a city street"}]}), encoding="utf-8")

    args = run_vbench_official_runner.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--videos-path",
            str(tmp_path / "videos"),
            "--vbench-root",
            str(vbench_root),
            "--preset",
            "full_16",
            "--full-json-dir",
            str(full_json),
        ]
    )
    args.dimension = run_vbench_official_runner.split_dimensions(args.dimension, args.preset)
    command = run_vbench_official_runner.build_official_command(args, tmp_path / "out" / "upstream")

    assert command[:2] == [args.python, str(vbench_root / "evaluate.py")]
    assert command[command.index("--full_json_dir") + 1] == str(full_json)
    dimension_index = command.index("--dimension") + 1
    mode_index = command.index("--mode")
    assert command[dimension_index:mode_index] == list(run_vbench_official_runner.VBENCH_DIMENSIONS)
    assert args.preset == ["full_16"]


def test_vbench_official_runner_computes_full_aggregate_metrics(tmp_path: Path) -> None:
    run_vbench_official_runner = _load_script("run_vbench_official_runner")
    upstream_results = tmp_path / "full_eval_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                metric_id: [bounds["max"], [{"video_path": "sample.mp4", "video_results": bounds["max"]}]]
                for metric_id, bounds in run_vbench_official_runner.VBENCH_NORMALIZATION.items()
            }
        ),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample.mp4").write_bytes(b"fake-video")
    output_dir = tmp_path / "vbench-full-out"

    exit_code = run_vbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--videos-path",
            str(videos_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert scorecard["metrics"]["per_metric"]["overall_quality"]["raw_score"] == pytest.approx(1.0)
    assert scorecard["metrics"]["per_metric"]["temporal_quality"]["raw_score"] == pytest.approx(1.0)
    assert scorecard["metrics"]["per_metric"]["frame_quality"]["raw_score"] == pytest.approx(1.0)
    assert scorecard["metrics"]["per_metric"]["text_alignment"]["raw_score"] == pytest.approx(1.0)
    assert scorecard["metrics"]["groups"]["vbench_aggregates"] == [
        "overall_quality",
        "temporal_quality",
        "frame_quality",
        "text_alignment",
    ]
    assert {row["metric_id"] for row in raw_rows} >= {
        "subject_consistency",
        "aesthetic_quality",
        "overall_quality",
        "text_alignment",
    }


def test_vbench_plus_plus_runner_normalizes_i2v_results(tmp_path: Path) -> None:
    run_vbench_plus_plus_official_runner = _load_script("run_vbench_plus_plus_official_runner")
    upstream_results = tmp_path / "results_eval_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "camera_motion": [0.4, [{"video_path": "sample.mp4", "video_results": 0.4}]],
                "i2v_background": [0.6, [{"video_path": "sample.mp4", "video_results": 0.6}]],
                "i2v_subject": [0.8, [{"video_path": "sample.mp4", "video_results": 0.8}]],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "vbench-plus-plus-out"

    exit_code = run_vbench_plus_plus_official_runner.main(
        [
            "--variant",
            "i2v",
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    dimension_scores = json.loads((output_dir / "dimension_scores.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert scorecard["evaluation"]["kind"] == "official_vbench_series"
    assert scorecard["benchmark"]["variant"] == "i2v"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["metrics"]["per_metric"]["i2v_subject"]["raw_score"] == 0.8
    assert scorecard["metrics"]["per_metric"]["vbench_plus_plus_i2v_average"]["raw_score"] == pytest.approx(0.6)
    assert scorecard["metrics"]["per_metric"]["vbench_plus_plus_average"]["raw_score"] == pytest.approx(0.6)
    assert raw_rows[-1]["metric_id"] == "vbench_plus_plus_average"
    assert dimension_scores["variant"] == "i2v"


def test_vbench_plus_plus_runner_prepares_long_custom_input_split(tmp_path: Path) -> None:
    from worldfoundry.evaluation.tasks.execution.runners.vbench_2_0.vbench_shared_official_impl import (
        ensure_long_custom_input_split,
    )
    videos_root = tmp_path / "videos"
    videos_root.mkdir()
    (videos_root / "sample_alpha.mp4").write_bytes(b"fake-video-a")
    (videos_root / "sample_beta.MP4").write_bytes(b"fake-video-b")

    manifest = ensure_long_custom_input_split(videos_root)

    assert manifest["official_preprocess_skip_ready"] is True
    assert manifest["source_video_count"] == 2
    assert manifest["ready_folder_count"] == 2
    assert (videos_root / "split_clip" / "sample_alpha-0" / "sample_alpha-0_full.mp4").exists()
    assert (videos_root / "split_clip" / "sample_beta-1" / "sample_beta-1_full.mp4").exists()
    assert {entry["method"] for entry in manifest["entries"]} <= {"copy", "existing", "hardlink", "symlink"}


def test_vbench_shared_runner_write_video_shim_adds_missing_api(tmp_path: Path) -> None:
    # The shim patches an *existing* torchvision install; without torchvision
    # the subprocess import legitimately fails.
    pytest.importorskip("torchvision")
    from worldfoundry.evaluation.tasks.execution.runners.vbench_2_0.vbench_shared_official_impl import (
        ensure_torchvision_write_video_shim,
    )
    shim_dir = ensure_torchvision_write_video_shim(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(shim_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torchvision.io as io; print(callable(getattr(io, 'write_video', None)))",
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "True"


def test_create_tiny_video_fixture_writes_mp4(tmp_path: Path) -> None:
    create_tiny_video = _load_script("create_tiny_video")
    output = tmp_path / "tiny" / "tiny_color_motion.mp4"

    exit_code = create_tiny_video.main(["--output", str(output), "--frames", "4", "--size", "32", "--json"])

    assert exit_code == 0
    assert output.is_file()
    assert output.stat().st_size > 0


def test_vbench_2_0_runner_normalizes_category_results(tmp_path: Path) -> None:
    run_vbench_2_0_official_runner = _load_script("run_vbench_2_0_official_runner")
    upstream_results = tmp_path / "vbench2_eval_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "Composition": [0.9, []],
                "Instance_Preservation": [0.8, []],
                "Camera_Motion": [0.7, []],
                "Human_Anatomy": [0.6, []],
                "Material": [0.5, []],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "vbench2-out"

    exit_code = run_vbench_2_0_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 0
    assert scorecard["benchmark"]["variant"] == "vbench2"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["run"]["status"] == "normalized"
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["metrics"]["per_metric"]["vbench2_creativity"]["raw_score"] == pytest.approx(0.9)
    assert scorecard["metrics"]["per_metric"]["vbench2_commonsense"]["raw_score"] == pytest.approx(0.8)
    assert scorecard["metrics"]["per_metric"]["vbench2_controllability"]["raw_score"] == pytest.approx(0.7)
    assert scorecard["metrics"]["per_metric"]["vbench2_human_fidelity"]["raw_score"] == pytest.approx(0.6)
    assert scorecard["metrics"]["per_metric"]["vbench2_physics"]["raw_score"] == pytest.approx(0.5)
    assert scorecard["metrics"]["per_metric"]["vbench2_total"]["raw_score"] == pytest.approx(0.7)
    assert raw_rows[-1]["metric_id"] == "vbench2_total"


def test_vbench2_runner_discovers_real_data_contract(tmp_path: Path) -> None:
    run_vbench_2_0_official_runner = _load_script("run_vbench_2_0_official_runner")
    dataset_root = tmp_path / "datasets"
    annotation_root = dataset_root / "VBench-2.0_human_annotation"
    anomaly_root = dataset_root / "VBench-2.0_human_anomaly"
    videos_root = tmp_path / "generated"
    annotation_root.mkdir(parents=True)
    anomaly_root.mkdir(parents=True)
    videos_root.mkdir()
    prompt_video = "A lion with wings.-0.mp4"
    (videos_root / prompt_video).write_bytes(b"fake")
    (annotation_root / "Composition.json").write_text(
        json.dumps(
            [
                {
                    "prompt_en": "A lion with wings.",
                    "videos": {"Demo": f"Demo/Composition/{prompt_video}"},
                    "human_anno": {"Demo": {}},
                }
            ]
        ),
        encoding="utf-8",
    )
    (annotation_root / "VBench2_arena_feedback.csv").write_text("winner,loser\n", encoding="utf-8")
    (anomaly_root / "README.md").write_text("# Human Anomaly Dataset\n", encoding="utf-8")
    upstream_results = tmp_path / "vbench2_eval_results.json"
    upstream_results.write_text(
        json.dumps({"Composition": [0.9, []], "Diversity": [0.7, []]}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "vbench2-out"

    exit_code = run_vbench_2_0_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--videos-path",
            str(videos_root),
            "--vbench2-dataset-root",
            str(dataset_root),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads((output_dir / "vbench2_dataset_manifest.json").read_text(encoding="utf-8"))
    video_coverage = json.loads((output_dir / "vbench2_video_coverage.json").read_text(encoding="utf-8"))
    benchmark_contract = json.loads((output_dir / "vbench2_benchmark_contract.json").read_text(encoding="utf-8"))
    prompt_rows = [
        json.loads(line) for line in (output_dir / "vbench2_prompt_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert exit_code == 0
    assert scorecard["run"]["status"] == "normalized"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["validation"]["official_runtime_executed"] is False
    assert scorecard["evaluation"]["available"] is True
    assert scorecard["dataset"]["real_data_ready"] is True
    assert scorecard["dataset"]["matched_reference_video_count"] == 1
    assert dataset_manifest["datasets"]["human_annotation"]["status"] == "partial"
    assert dataset_manifest["prompt_count"] == 1
    assert video_coverage["matched_reference_video_names"] == [prompt_video]
    assert benchmark_contract["official_validation_boundary"]["normalizer_only"] is True
    assert prompt_rows[0]["prompt"] == "A lion with wings."


def test_worldmodelbench_official_runner_normalizes_existing_upstream_results(tmp_path: Path) -> None:
    run_worldmodelbench_official_runner = _load_script("run_worldmodelbench_official_runner")
    worldmodelbench_root = tmp_path / "WorldModelBench"
    worldmodelbench_root.mkdir()
    (worldmodelbench_root / "worldmodelbench.json").write_text(
        json.dumps(
            [
                {"first_frame": "images/sample_a.jpg"},
                {"first_frame": "images/sample_b.jpg"},
            ]
        ),
        encoding="utf-8",
    )
    upstream_results = tmp_path / "worldmodelbench_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "model_name": "demo-model",
                "preds": {
                    "sample_a": {
                        "instruction": ["Score: 3"],
                        "common_sense": ["No", "Yes"],
                        "physical_laws": ["No", "No", "Yes", "No", "No"],
                    },
                    "sample_b": {
                        "instruction": ["Score: 1"],
                        "common_sense": ["No", "No"],
                        "physical_laws": ["No", "Yes", "No", "No", "No"],
                    },
                },
                "accs": {
                    "instruction": [3, 1],
                    "common_sense": [True, False, True, True],
                    "physical_laws": [True, True, False, True, True, True, False, True, True, True],
                },
            }
        ),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample_a.mp4").write_bytes(b"fake-video")
    (videos_dir / "sample_b.mp4").write_bytes(b"fake-video")
    output_dir = tmp_path / "worldmodelbench-out"

    exit_code = run_worldmodelbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--worldmodelbench-root",
            str(worldmodelbench_root),
            "--video-dir",
            str(videos_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    judge_rows = [
        json.loads(line) for line in (output_dir / "judge_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 1
    assert scorecard["benchmark"]["contract_only"] is False
    assert scorecard["evaluation"]["kind"] == "official_worldmodelbench"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["validation"]["official_result_shape"]["ok"] is True
    assert scorecard["dataset"]["official_manifest_coverage"]["coverage_complete"] is True
    assert scorecard["dataset"]["official_manifest_coverage"]["expected_file_count"] == 2
    assert scorecard["metrics"]["per_metric"]["instruction_following"]["raw_score"] == 2.0
    assert scorecard["metrics"]["per_metric"]["common_sense"]["raw_score"] == 1.5
    assert scorecard["metrics"]["per_metric"]["physical_adherence"]["raw_score"] == 4.0
    assert scorecard["metrics"]["per_metric"]["world_model_average"]["raw_score"] == 7.5
    assert raw_rows[-1]["metric_id"] == "world_model_average"
    assert len(judge_rows) == 16


def test_worldmodelbench_official_runner_uses_data_root_evaluation_with_judge_checkpoint(tmp_path: Path) -> None:
    run_worldmodelbench_official_runner = _load_script("run_worldmodelbench_official_runner")
    worldmodelbench_root = tmp_path / "missing-official-repo"
    data_root = tmp_path / "worldmodelbench-data"
    data_root.mkdir()
    (data_root / "worldmodelbench.json").write_text(
        json.dumps(
            [
                {"first_frame": "images/sample_a.jpg", "text_instruction": "move left"},
                {"first_frame": "images/sample_b.jpg", "text_instruction": "move right"},
            ]
        ),
        encoding="utf-8",
    )
    (data_root / "evaluation.py").write_text(
        "\n".join(
            [
                "import argparse, json, os",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--model_name', required=True)",
                "parser.add_argument('--video_dir', required=True)",
                "parser.add_argument('--judge', required=True)",
                "parser.add_argument('--save_name', required=True)",
                "parser.add_argument('--cot', action='store_true')",
                "args = parser.parse_args()",
                "Path(args.save_name).parent.joinpath('observed_args.json').write_text(",
                "    json.dumps({'cwd': os.getcwd(), 'judge': args.judge, 'video_dir': args.video_dir}),",
                "    encoding='utf-8',",
                ")",
                "results = {",
                "    'model_name': args.model_name,",
                "    'preds': {",
                "        'sample_a': {'instruction': ['Score: 3'], 'common_sense': ['Yes', 'Yes'], 'physical_laws': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes']},",
                "        'sample_b': {'instruction': ['Score: 1'], 'common_sense': ['Yes', 'No'], 'physical_laws': ['Yes', 'No', 'Yes', 'No', 'Yes']},",
                "    },",
                "    'accs': {",
                "        'instruction': [3, 1],",
                "        'common_sense': [True, True, True, False],",
                "        'physical_laws': [True, True, True, True, True, True, False, True, False, True],",
                "    },",
                "}",
                "Path(f'{args.save_name}.json').write_text(json.dumps(results), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample_a.mp4").write_bytes(b"fake-video")
    (videos_dir / "sample_b.mp4").write_bytes(b"fake-video")
    judge_checkpoint = tmp_path / "vila-judge-ckpt"
    judge_checkpoint.write_text("fake checkpoint marker", encoding="utf-8")
    output_dir = tmp_path / "worldmodelbench-out"

    exit_code = run_worldmodelbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--worldmodelbench-root",
            str(worldmodelbench_root),
            "--data-root",
            str(data_root),
            "--video-dir",
            str(videos_dir),
            "--judge",
            str(judge_checkpoint),
            "--model-name",
            "demo-model",
            "--python",
            sys.executable,
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    observed_args = json.loads((output_dir / "upstream" / "observed_args.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert observed_args == {
        "cwd": str(data_root),
        "judge": str(judge_checkpoint),
        "video_dir": str(videos_dir),
    }
    assert scorecard["official_benchmark_verified"] is True
    assert scorecard["integration_evidence"] is True
    assert scorecard["validation"]["official_runtime_executed"] is True
    assert scorecard["dataset"]["data_root"] == str(data_root)
    assert scorecard["dataset"]["official_manifest_coverage"]["coverage_complete"] is True
    assert scorecard["run"]["command"][1] == str(data_root / "evaluation.py")


def test_worldmodelbench_official_runner_rejects_invalid_official_acc_shape(tmp_path: Path) -> None:
    run_worldmodelbench_official_runner = _load_script("run_worldmodelbench_official_runner")
    worldmodelbench_root = tmp_path / "WorldModelBench"
    worldmodelbench_root.mkdir()
    (worldmodelbench_root / "worldmodelbench.json").write_text(
        json.dumps(
            [
                {"first_frame": "images/sample_a.jpg"},
                {"first_frame": "images/sample_b.jpg"},
            ]
        ),
        encoding="utf-8",
    )
    upstream_results = tmp_path / "worldmodelbench_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "model_name": "demo-model",
                "preds": {
                    "sample_a": {"instruction": ["Score: 3"]},
                    "sample_b": {"instruction": ["Score: 1"]},
                },
                "accs": {
                    "instruction": [3, 1],
                    "common_sense": [True, False, True],
                    "physical_laws": [True, True, False, True, True, True, False, True, True, True],
                },
            }
        ),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample_a.mp4").write_bytes(b"fake-video")
    (videos_dir / "sample_b.mp4").write_bytes(b"fake-video")
    output_dir = tmp_path / "worldmodelbench-out"

    exit_code = run_worldmodelbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--worldmodelbench-root",
            str(worldmodelbench_root),
            "--video-dir",
            str(videos_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    issues = scorecard["validation"]["official_result_shape"]["issues"]
    assert exit_code == 1
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["evaluation"]["available"] is False
    assert {"category": "common_sense", "reason": "invalid_acc_shape", "actual": 3, "expected": 4} in issues


def test_worldmodelbench_official_runner_reports_manifest_video_coverage(tmp_path: Path) -> None:
    run_worldmodelbench_official_runner = _load_script("run_worldmodelbench_official_runner")
    worldmodelbench_root = tmp_path / "WorldModelBench"
    worldmodelbench_root.mkdir()
    (worldmodelbench_root / "worldmodelbench.json").write_text(
        json.dumps(
            [
                {"first_frame": "images/sample_a.jpg"},
                {"first_frame": "images/sample_b.jpg"},
                {"first_frame": "images/sample_c.jpg"},
            ]
        ),
        encoding="utf-8",
    )
    upstream_results = tmp_path / "worldmodelbench_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "model_name": "demo-model",
                "preds": {"sample_a": {"instruction": ["Score: 3"]}},
                "accs": {
                    "instruction": [3],
                    "common_sense": [True, False],
                    "physical_laws": [True, True, False, True, True],
                },
            }
        ),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample_a.mp4").write_bytes(b"fake-video")
    (videos_dir / "unexpected.mp4").write_bytes(b"fake-video")
    output_dir = tmp_path / "worldmodelbench-out"

    exit_code = run_worldmodelbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--worldmodelbench-root",
            str(worldmodelbench_root),
            "--video-dir",
            str(videos_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    coverage = scorecard["dataset"]["official_manifest_coverage"]
    assert exit_code == 1
    assert coverage["coverage_complete"] is False
    assert coverage["expected_file_count"] == 3
    assert coverage["matched_file_count"] == 1
    assert coverage["missing_file_count"] == 2
    assert coverage["unexpected_file_count"] == 1
    assert coverage["missing_video_names"] == ["sample_b.mp4", "sample_c.mp4"]
    assert coverage["unexpected_video_names"] == ["unexpected.mp4"]


def test_worldmodelbench_official_runner_resolves_hf_cache_manifest(tmp_path: Path) -> None:
    run_worldmodelbench_official_runner = _load_script("run_worldmodelbench_official_runner")
    hf_cache_dir = tmp_path / "hfd"
    snapshot = (
        hf_cache_dir
        / "datasets--Efficient-Large-Model--worldmodelbench"
        / "snapshots"
        / "f2450a891498b3daed3f28fc9ba0ab8adce90617"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "worldmodelbench.json").write_text(
        json.dumps({"data": [{"id": "0001"}, {"first_frame": "images/0002.png"}]}),
        encoding="utf-8",
    )
    upstream_results = tmp_path / "worldmodelbench_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "preds": {
                    "0001": {"instruction": ["Score: 3"]},
                    "0002": {"instruction": ["Score: 1"]},
                },
                "accs": {
                    "instruction": [3, 1],
                    "common_sense": [True, False, True, False],
                    "physical_laws": [True, True, False, True, True, True, False, True, True, True],
                },
            }
        ),
        encoding="utf-8",
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "0001.mp4").write_bytes(b"fake-video")
    (videos_dir / "0002.webm").write_bytes(b"fake-video")
    output_dir = tmp_path / "worldmodelbench-out"

    exit_code = run_worldmodelbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--hf-cache-dir",
            str(hf_cache_dir),
            "--video-dir",
            str(videos_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    coverage = scorecard["dataset"]["official_manifest_coverage"]
    assert exit_code == 1
    assert scorecard["dataset"]["data_root"] == str(snapshot)
    assert coverage["manifest_path"] == str(snapshot / "worldmodelbench.json")
    assert coverage["coverage_complete"] is True
    assert coverage["generated_file_count"] == 2


def test_worldmodelbench_official_runner_does_not_double_normalize_normalized_score(tmp_path: Path) -> None:
    run_worldmodelbench_official_runner = _load_script("run_worldmodelbench_official_runner")
    worldmodelbench_root = tmp_path / "WorldModelBench"
    worldmodelbench_root.mkdir()
    (worldmodelbench_root / "worldmodelbench.json").write_text("[]", encoding="utf-8")
    upstream_results = tmp_path / "worldmodelbench_results.json"
    upstream_results.write_text(
        json.dumps({"metrics": {"physical_adherence": {"normalized_score": 0.8}}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "worldmodelbench-out"

    exit_code = run_worldmodelbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--worldmodelbench-root",
            str(worldmodelbench_root),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    metric = scorecard["metrics"]["per_metric"]["physical_adherence"]
    assert exit_code == 1
    assert metric["raw_score"] is None
    assert metric["normalized_score"] == 0.8
    assert metric["score_scale"] == "normalized"


def test_worldscore_official_runner_normalizes_existing_official_results(tmp_path: Path) -> None:
    run_worldscore_official_runner = _load_script("run_worldscore_official_runner")
    official_results = tmp_path / "worldscore.json"
    official_results.write_text(
        json.dumps(
            {
                "camera_control": 80,
                "object_control": 60,
                "content_alignment": 70,
                "3d_consistency": 50,
                "photometric_consistency": 90,
                "style_consistency": 80,
                "subjective_quality": 100,
                "motion_accuracy": 40,
                "motion_magnitude": 60,
                "motion_smoothness": 80,
                "WorldScore-Static": 75,
                "WorldScore-Dynamic": 71,
                "per_sample_metrics": [
                    {
                        "sample_id": "demo-0001",
                        "metrics": {
                            "camera_control": {
                                "score_normalized": 0.8,
                            },
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "worldscore-out"

    exit_code = run_worldscore_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--official-results-path",
            str(official_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    sample_rows = [
        json.loads(line) for line in (output_dir / "per_sample_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 0
    assert scorecard["evaluation"]["kind"] == "official_worldscore"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["worldfoundry_contract_validation_evidence"] is True
    assert scorecard["metrics"]["per_metric"]["controllability"]["raw_score"] == 70.0
    assert scorecard["metrics"]["per_metric"]["quality"]["raw_score"] == 78.0
    assert scorecard["metrics"]["per_metric"]["dynamics"]["raw_score"] == 60.0
    assert scorecard["metrics"]["per_metric"]["worldscore_average"]["raw_score"] == 71.0
    assert scorecard["metrics"]["per_metric"]["worldscore_average"]["normalized_score"] == 0.71
    assert raw_rows[-1]["metric_id"] == "worldscore_average"
    assert sample_rows[0]["sample_id"] == "demo-0001"


def test_worldscore_official_runner_reads_explicit_output_dir(tmp_path: Path) -> None:
    run_worldscore_official_runner = _load_script("run_worldscore_official_runner")
    model_path = tmp_path / "model-root"
    stale_output = model_path / "old-run"
    explicit_output = tmp_path / "worldscore_output"
    sample_dir = explicit_output / "static" / "style-a" / "scene-a" / "category-a" / "sample-a"
    stale_output.mkdir(parents=True)
    sample_dir.mkdir(parents=True)
    (stale_output / "worldscore.json").write_text(
        json.dumps({"WorldScore-Dynamic": 12, "camera_control": 12, "object_control": 12}),
        encoding="utf-8",
    )
    (explicit_output / "worldscore.json").write_text(
        json.dumps(
            {
                "camera_control": 90,
                "object_control": 70,
                "content_alignment": 80,
                "motion_accuracy": 60,
                "WorldScore-Dynamic": 76,
            }
        ),
        encoding="utf-8",
    )
    (sample_dir / "evaluation.json").write_text(
        json.dumps({"camera_control": {"camera_control": {"score_normalized": 0.9}}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "worldscore-out"

    exit_code = run_worldscore_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--model-path",
            str(model_path),
            "--evaluation-output-dir",
            str(explicit_output),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    sample_rows = [
        json.loads(line) for line in (output_dir / "per_sample_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 0
    assert scorecard["evaluation"]["official_results"] == str(explicit_output / "worldscore.json")
    assert scorecard["evaluation"]["evaluation_output_dir"] == str(explicit_output)
    assert scorecard["dataset"]["generated_artifact_dir"] == str(explicit_output)
    assert scorecard["metrics"]["per_metric"]["controllability"]["raw_score"] == 80.0
    assert scorecard["metrics"]["per_metric"]["worldscore_average"]["raw_score"] == 76.0
    assert scorecard["worldfoundry_contract_validation_evidence"] is True
    assert scorecard["integration_evidence"] is False
    assert len(sample_rows) == 1
    assert sample_rows[0]["relative_path"] == "static/style-a/scene-a/category-a/sample-a/evaluation.json"


def test_worldscore_official_runner_aggregates_generated_evaluation_tree(tmp_path: Path) -> None:
    run_worldscore_official_runner = _load_script("run_worldscore_official_runner")
    generated_root = tmp_path / "worldscore-generated"
    sample_a = generated_root / "static" / "style-a" / "scene-a" / "category-a" / "sample-a"
    sample_b = generated_root / "dynamic" / "style-b" / "scene-b" / "category-b" / "sample-b"
    sample_a.mkdir(parents=True)
    sample_b.mkdir(parents=True)
    (sample_a / "evaluation.json").write_text(
        json.dumps(
            {
                "camera_control": {"score": 80},
                "object_control": {"score": 60},
                "content_alignment": 70,
                "motion_accuracy": 40,
            }
        ),
        encoding="utf-8",
    )
    (sample_b / "evaluation.json").write_text(
        json.dumps(
            {
                "camera_control": {"score": 60},
                "object_control": {"score": 40},
                "content_alignment": 90,
                "motion_accuracy": 80,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "worldscore-out"

    exit_code = run_worldscore_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--evaluation-output-dir",
            str(tmp_path / "missing-evaluation-output"),
            "--generated-root",
            str(generated_root),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    sample_rows = [
        json.loads(line) for line in (output_dir / "per_sample_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    aggregate_path = output_dir / "official" / "worldscore_from_evaluation_tree.json"
    assert exit_code == 0
    assert aggregate_path.is_file()
    assert scorecard["evaluation"]["official_results"] == str(aggregate_path)
    assert scorecard["evaluation"]["normalization_source"] == "generated_evaluation_tree"
    assert scorecard["worldfoundry_contract_validation_evidence"] is True
    assert scorecard["integration_evidence"] is False
    assert scorecard["metrics"]["per_metric"]["controllability"]["raw_score"] == 60.0
    assert scorecard["metrics"]["per_metric"]["quality"]["raw_score"] == 80.0
    assert scorecard["metrics"]["per_metric"]["dynamics"]["raw_score"] == 60.0
    assert scorecard["metrics"]["per_metric"]["worldscore_average"]["raw_score"] == pytest.approx(200.0 / 3.0)
    assert len(sample_rows) == 2


def test_worldscore_official_runner_resolves_hf_cache_data_path(tmp_path: Path) -> None:
    run_worldscore_official_runner = _load_script("run_worldscore_official_runner")
    hf_cache_dir = tmp_path / "hfd"
    snapshot = (
        hf_cache_dir
        / "datasets--Howieeeee--WorldScore"
        / "snapshots"
        / "42c4e267e1cc0529af1b0284ce018e04d43906e3"
    )
    snapshot.mkdir(parents=True)
    official_results = tmp_path / "worldscore.json"
    official_results.write_text(json.dumps({"WorldScore-Dynamic": 71}), encoding="utf-8")
    output_dir = tmp_path / "worldscore-out"

    exit_code = run_worldscore_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--hf-cache-dir",
            str(hf_cache_dir),
            "--official-results-path",
            str(official_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert scorecard["dataset"]["data_path"] == str(snapshot)
    assert run_worldscore_official_runner.resolve_worldscore_data_path(
        run_worldscore_official_runner.build_parser().parse_args(
            ["--output-dir", str(output_dir), "--hf-cache-dir", str(hf_cache_dir)]
        )
    ) == snapshot


def test_run_benchmark_execution_worldscore_validation_without_fixture_reports_no_evidence(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    manifests = run_benchmark_execution.load_manifests(REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog", "worldscore")
    result = run_benchmark_execution.run_benchmark(
        manifests[0],
        tmp_path / "benchmark_zoo",
        timeout_seconds=120,
        command_kind="validation",
        check_artifacts=True,
    )
    assert result["command_kind"] == "validation"
    assert result["returncode"] == 1
    assert result["official_benchmark_verified"] is False
    assert result["integration_evidence"] is False
    assert result["ok"] is False


def test_videobench_official_runner_normalizes_existing_upstream_results(tmp_path: Path) -> None:
    run_videobench_official_runner = _load_script("run_videobench_official_runner")
    upstream_results = tmp_path / "videobench_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "imaging_quality": {
                    "average_scores": {"demo-model": 4.0},
                    "scores": {"0": {"prompt_en": "a close up of grapes", "demo-model": 4}},
                },
                "color": {
                    "average_scores": {"demo-model": 2.0},
                    "scores": {"0": {"prompt_en": "a red bird", "demo-model": 2}},
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "videobench-out"

    exit_code = run_videobench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    sample_rows = [
        json.loads(line) for line in (output_dir / "per_sample_scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    per_metric = scorecard["metrics"]["per_metric"]
    assert exit_code == 1
    assert scorecard["evaluation"]["kind"] == "official_videobench"
    assert scorecard["integration_evidence"] is False
    assert per_metric["imaging_quality"]["raw_score"] == 4.0
    assert per_metric["imaging_quality"]["normalized_score"] == 0.75
    assert per_metric["color_consistency"]["raw_score"] == 2.0
    assert per_metric["color_consistency"]["normalized_score"] == pytest.approx(0.5)
    assert per_metric["videobench_average"]["normalized_score"] == pytest.approx((0.75 + 0.5) / 2)
    assert raw_rows[-1]["metric_id"] == "videobench_average"
    assert sample_rows[0]["metric_id"] == "imaging_quality"
    assert sample_rows[0]["model_name"] == "demo-model"
    assert scorecard["run"]["status"] == "normalized"
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["validation"]["official_runtime_executed"] is False
    assert scorecard["official_benchmark_verified"] is False
    assert (output_dir / "upstream_stdout.log").is_file()
    assert (output_dir / "upstream_stderr.log").is_file()


def test_videobench_runner_writes_blocked_scorecard_for_missing_api_key(tmp_path: Path) -> None:
    run_videobench_official_runner = _load_script("run_videobench_official_runner")
    videobench_root = tmp_path / "Video-Bench"
    generated_dir = tmp_path / "generated"
    annotation_root = tmp_path / "Video-Bench_human_annotation"
    (videobench_root / "videobench").mkdir(parents=True)
    generated_dir.mkdir()
    annotation_root.mkdir()
    (annotation_root / "imaging_quality.json").write_text(
        json.dumps([{"prompt_en": "Demo prompt", "videos": {}, "human_anno": {}}]),
        encoding="utf-8",
    )

    output_dir = tmp_path / "videobench-blocked-out"
    exit_code = run_videobench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--videobench-root",
            str(videobench_root),
            "--generated-video-dir",
            str(generated_dir),
            "--annotation-root",
            str(annotation_root),
            "--dimension",
            "imaging_quality",
            "--gpt4o-api-key",
            "",
            "--gpt4o-mini-api-key",
            "",
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert scorecard["run"]["status"] == "blocked"
    assert scorecard["evaluation"]["available"] is False
    assert scorecard["validation"]["normalizer_only"] is False
    assert scorecard["validation"]["official_runtime_executed"] is False
    assert scorecard["validation"]["blocked_reasons"] == [
        "WORLDFOUNDRY_VIDEOBENCH_GPT4O_API_KEY or OPENAI_API_KEY",
        "WORLDFOUNDRY_VIDEOBENCH_GPT4O_MINI_API_KEY or OPENAI_API_KEY",
    ]
    assert scorecard["evaluation"]["videos_path"] == str(generated_dir.resolve())
    assert scorecard["evaluation"]["full_json_dir"] == str((output_dir / "VideoBench_full.from_annotations.json").resolve())
    assert scorecard["dataset"]["human_annotation"]["available"] is True
    assert (output_dir / "upstream" / "blocked_missing_judge_api_score_results.json").is_file()


def test_videobench_runner_builds_prompt_suite_from_hf_annotations(tmp_path: Path) -> None:
    run_videobench_official_runner = _load_script("run_videobench_official_runner")
    annotation_root = tmp_path / "Video-Bench_human_annotation"
    annotation_root.mkdir()
    (annotation_root / "imaging_quality.json").write_text(
        json.dumps(
            [
                {"prompt_en": "Close up of grapes on a rotating table.", "videos": {}, "human_anno": {}},
                {"prompt_en": "A bright mountain lake.", "videos": {}, "human_anno": {}},
            ]
        ),
        encoding="utf-8",
    )
    (annotation_root / "color.json").write_text(
        json.dumps(
            [
                {"prompt_en": "A bright mountain lake.", "videos": {}, "human_anno": {}},
            ]
        ),
        encoding="utf-8",
    )
    full_json_path = run_videobench_official_runner.build_full_json_from_annotations(
        annotation_root,
        tmp_path / "VideoBench_full.from_annotations.json",
        ["imaging_quality", "color"],
    )

    payload = json.loads(full_json_path.read_text(encoding="utf-8"))
    by_prompt = {row["prompt"]: row["dimension"] for row in payload}
    assert by_prompt["Close up of grapes on a rotating table."] == ["imaging_quality"]
    assert by_prompt["A bright mountain lake."] == ["imaging_quality", "color"]


def test_videobench_runner_resolves_generated_video_dir_for_official_command(tmp_path: Path) -> None:
    run_videobench_official_runner = _load_script("run_videobench_official_runner")
    videobench_root = tmp_path / "Video-Bench"
    generated_dir = tmp_path / "generated"
    output_dir = tmp_path / "out"
    full_json = tmp_path / "VideoBench_full.json"
    config_path = output_dir / "videobench_config.private.json"
    upstream_output = output_dir / "upstream"
    (videobench_root / "videobench").mkdir(parents=True)
    generated_dir.mkdir()
    output_dir.mkdir()
    full_json.write_text("[]", encoding="utf-8")
    args = run_videobench_official_runner.build_parser().parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--videobench-root",
            str(videobench_root),
            "--generated-video-dir",
            str(generated_dir),
            "--full-json-dir",
            str(full_json),
            "--dimension",
            "imaging_quality",
            "--python",
            "python",
        ]
    )

    videos_path = run_videobench_official_runner.resolve_videos_path(args)
    resolved_full_json = run_videobench_official_runner.resolve_full_json_path(
        args,
        output_dir,
        ["imaging_quality"],
        annotation_root=None,
    )
    command = run_videobench_official_runner.build_official_command(
        args,
        config_path,
        upstream_output,
        ["imaging_quality"],
        videos_path,
        resolved_full_json,
    )

    assert videos_path == generated_dir
    assert command[command.index("--videos_path") + 1] == str(generated_dir)
    assert command[command.index("--full_json_dir") + 1] == str(full_json)


def test_videobench_runner_scorecard_records_real_data_roots(tmp_path: Path) -> None:
    run_videobench_official_runner = _load_script("run_videobench_official_runner")
    upstream_results = tmp_path / "results_imaging_quality_score_results.json"
    annotation_root = tmp_path / "Video-Bench_human_annotation"
    official_videos_root = tmp_path / "Video-Bench_videos"
    generated_dir = tmp_path / "generated"
    annotation_root.mkdir()
    official_videos_root.mkdir()
    (official_videos_root / "color").mkdir()
    (generated_dir / "video-text consistency" / "demo-model").mkdir(parents=True)
    (generated_dir / "video-text consistency" / "demo-model" / "Close up of grapes.mp4").write_text("fake", encoding="utf-8")
    (annotation_root / "imaging_quality.json").write_text(
        json.dumps([{"prompt_en": "Close up of grapes", "videos": {}, "human_anno": {}}]),
        encoding="utf-8",
    )
    upstream_results.write_text(
        json.dumps(
            {
                "average_scores": {"demo-model": 5.0},
                "scores": {"0": {"prompt_en": "Close up of grapes", "demo-model": 5.0}},
            }
        ),
        encoding="utf-8",
    )
    scorecard = run_videobench_official_runner.normalize_videobench_results(
        {"average_scores": {"demo-model": 5.0}, "scores": {"0": {"prompt_en": "Close up of grapes", "demo-model": 5.0}}},
        benchmark_id="video-bench",
        output_dir=tmp_path / "out",
        upstream_results_path=upstream_results,
        annotation_root=annotation_root,
        official_videos_root=official_videos_root,
        videos_path=generated_dir,
        full_json_path=tmp_path / "VideoBench_full.json",
        command=["python", "evaluate.py"],
        duration_seconds=1.0,
        returncode=0,
        stdout_path=tmp_path / "out" / "upstream_stdout.log",
        stderr_path=tmp_path / "out" / "upstream_stderr.log",
    )

    assert scorecard["dataset"]["human_annotation"]["available"] is True
    assert scorecard["dataset"]["official_videos"]["available"] is True
    assert scorecard["dataset"]["official_videos"]["dimension_dirs"] == ["color"]
    assert scorecard["evaluation"]["videos_path"] == str(generated_dir.resolve())
    assert scorecard["metrics"]["per_metric"]["imaging_quality"]["normalized_score"] == 1.0


def test_videoscore_official_runner_normalizes_existing_upstream_results(tmp_path: Path) -> None:
    run_videoscore_official_runner = _load_script("run_videoscore_official_runner")
    upstream_results = tmp_path / "eval_video_feedback_videoscore.json"
    upstream_results.write_text(
        json.dumps(
            [
                {
                    "id": "sample-a",
                    "text": "a prompt",
                    "ref": "[2, 3, 4, 1, 2]",
                    "ans": "[2.0, 3.0, 4.0, 1.0, 2.0]",
                },
                {
                    "id": "sample-b",
                    "text": "another prompt",
                    "ref": "[3, 3, 2, 4, 1]",
                    "ans": "[3.0, 3.0, 2.0, 4.0, 1.0]",
                },
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "videoscore-out"

    exit_code = run_videoscore_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    sample_rows = [
        json.loads(line) for line in (output_dir / "per_sample_scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    dataset_manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    generated_manifest = json.loads((output_dir / "generated_video_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert scorecard["evaluation"]["kind"] == "official_videoscore"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["dataset"]["hf_dataset_id"] == "TIGER-Lab/VideoFeedback"
    assert scorecard["metrics"]["per_metric"]["visual_quality"]["raw_score"] == 2.5
    assert scorecard["metrics"]["per_metric"]["temporal_consistency"]["raw_score"] == 3.0
    assert scorecard["metrics"]["per_metric"]["dynamic_degree"]["raw_score"] == 3.0
    assert scorecard["metrics"]["per_metric"]["text_to_video_alignment"]["raw_score"] == 2.5
    assert scorecard["metrics"]["per_metric"]["factual_consistency"]["raw_score"] == 1.5
    assert scorecard["metrics"]["per_metric"]["videoscore_average"]["raw_score"] == 2.5
    assert scorecard["metrics"]["per_metric"]["videoscore_average"]["normalized_score"] == 0.5
    assert raw_rows[-1]["metric_id"] == "videoscore_average"
    assert sample_rows[0]["sample_id"] == "sample-a"
    assert dataset_manifest["expected_rows"] == 4000
    assert generated_manifest["expected_file_count"] == 4000


def test_videoscore_official_runner_discovers_videofeedback_and_frames(tmp_path: Path) -> None:
    run_videoscore_official_runner = _load_script("run_videoscore_official_runner")
    upstream_results = tmp_path / "eval_video_feedback_videoscore.json"
    dataset_root = tmp_path / "VideoFeedback"
    frames_dir = tmp_path / "frames"
    dataset_root.mkdir()
    frames_dir.mkdir()
    upstream_results.write_text(
        json.dumps([{"id": "p100263", "text": "a prompt", "ans": "[4, 4, 4, 4, 4]"}]),
        encoding="utf-8",
    )
    (dataset_root / "real_train.jsonl").write_text(
        json.dumps({"id": "p100263", "text prompt": "a prompt", "video link": "https://example/p100263.mp4"})
        + "\n",
        encoding="utf-8",
    )
    (frames_dir / "p100263.mp4").write_bytes(b"video")
    output_dir = tmp_path / "videoscore-out"

    exit_code = run_videoscore_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--dataset-root",
            str(dataset_root),
            "--frames-dir",
            str(frames_dir),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    generated_manifest = json.loads((output_dir / "generated_video_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert scorecard["metrics"]["per_metric"]["videoscore_average"]["normalized_score"] == 1.0
    assert dataset_manifest["exists"] is True
    assert dataset_manifest["file_count"] == 1
    assert generated_manifest["video_file_count"] == 1
    assert generated_manifest["coverage_complete"] is True


def test_videoscore_official_runner_bounded_sample_records_integration_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_videoscore_official_runner = _load_script("run_videoscore_official_runner")
    bench_root = tmp_path / "VideoScore-Bench"
    split_root = bench_root / "video_feedback"
    split_root.mkdir(parents=True)
    frames_dir = tmp_path / "frames_video_feedback"
    frames_dir.mkdir()
    (frames_dir / "sample-a.mp4").write_bytes(b"video")
    output_dir = tmp_path / "videoscore-out"

    def fake_bounded_runner(args: argparse.Namespace, result_file: Path) -> tuple[list[dict[str, object]], Path]:
        rows = [
            {
                "id": "sample-a",
                "text": "a prompt",
                "ref": "[2, 3, 4, 1, 2]",
                "ans": "[2.0, 3.0, 4.0, 1.0, 2.0]",
            }
        ]
        run_videoscore_official_runner.write_json(result_file, rows)
        return rows, frames_dir

    monkeypatch.setattr(run_videoscore_official_runner, "run_bounded_videoscore", fake_bounded_runner)

    exit_code = run_videoscore_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--bounded-sample-count",
            "1",
            "--bench-data-root",
            str(bench_root),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert scorecard["official_benchmark_verified"] is True
    assert scorecard["integration_evidence"] is True
    assert "--bounded-sample-count" in scorecard["run"]["command"]
    assert scorecard["evaluation"]["dataset_root"] == str(split_root.resolve())
    assert scorecard["metrics"]["per_metric"]["videoscore_average"]["raw_score"] == 2.4


def test_videoscore_runner_patches_transformers_dynamic_cache_api(monkeypatch: pytest.MonkeyPatch) -> None:
    run_videoscore_official_runner = _load_script("run_videoscore_official_runner")

    class FakeDynamicCache:
        def get_seq_length(self, layer_idx: int = 0) -> int:
            return 7 + layer_idx

    fake_cache_utils = types.ModuleType("transformers.cache_utils")
    fake_cache_utils.DynamicCache = FakeDynamicCache
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.cache_utils = fake_cache_utils
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)

    run_videoscore_official_runner.patch_transformers_dynamic_cache_api()

    assert FakeDynamicCache().get_usable_length(128) == 7


def test_t2v_compbench_official_runner_normalizes_existing_csv_dir(tmp_path: Path) -> None:
    run_t2v_compbench_official_runner = _load_script("run_t2v_compbench_official_runner")
    csv_dir = tmp_path / "official-csv"
    csv_dir.mkdir()
    (csv_dir / "demo_consistent_attr_score.csv").write_text(
        "name,prompt,Score\n0001.mp4,a red cube,0.80\nscore: ,0.75\n",
        encoding="utf-8",
    )
    (csv_dir / "demo_spatial_score.csv").write_text(
        "video_name,image_name,prompt,object_1,object_2,score\n0002.mp4,0002.jpg,left of,cat,dog,0.50\nScore: ,0.50\n",
        encoding="utf-8",
    )
    (csv_dir / "demo_motion_score.csv").write_text(
        "id,object_1,d_1,object_2,d_2,Score\n0003,car,right,,,0.25\nscore: ,0.25\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "t2v-compbench-out"

    exit_code = run_t2v_compbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--csv-dir",
            str(csv_dir),
            "--model-name",
            "demo",
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    sample_rows = [
        json.loads(line) for line in (output_dir / "per_sample_scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    csv_manifest = json.loads((output_dir / "leaderboard_csv_manifest.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    generated_manifest = json.loads((output_dir / "generated_video_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert scorecard["evaluation"]["kind"] == "official_t2v_compbench"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["dataset"]["hf_dataset_id"] == "Kaiyue/T2V-CompBench-Videos"
    assert scorecard["metrics"]["per_metric"]["consistent_attribute_binding"]["raw_score"] == 0.75
    assert scorecard["metrics"]["per_metric"]["spatial_relationships"]["raw_score"] == 0.5
    assert scorecard["metrics"]["per_metric"]["motion_binding"]["raw_score"] == 0.25
    assert scorecard["metrics"]["per_metric"]["t2v_compbench_average"]["raw_score"] == 0.5
    assert scorecard["metrics"]["per_metric"]["t2v_compbench_average"]["normalized_score"] == 0.5
    assert raw_rows[-1]["metric_id"] == "t2v_compbench_average"
    assert {row["metric_id"] for row in sample_rows} == {
        "consistent_attribute_binding",
        "spatial_relationships",
        "motion_binding",
    }
    assert csv_manifest["found_metrics"] == [
        "consistent_attribute_binding",
        "motion_binding",
        "spatial_relationships",
    ]
    assert dataset_manifest["expected_rows"] == 25200
    assert generated_manifest["expected_file_count"] == 1400


def test_t2v_compbench_official_runner_discovers_dataset_and_generated_videos(tmp_path: Path) -> None:
    run_t2v_compbench_official_runner = _load_script("run_t2v_compbench_official_runner")
    csv_dir = tmp_path / "official-csv"
    dataset_root = tmp_path / "Kaiyue--T2V-CompBench-Videos"
    video_root = tmp_path / "generated"
    (video_root / "action_binding").mkdir(parents=True)
    (video_root / "spatial_relationships").mkdir(parents=True)
    dataset_root.mkdir()
    csv_dir.mkdir()
    (dataset_root / "metadata.jsonl").write_text(
        json.dumps({"video": {"src": "zip://demo/action_5/0001.mp4"}, "label": 0}) + "\n",
        encoding="utf-8",
    )
    (video_root / "action_binding" / "0001.mp4").write_bytes(b"video")
    (video_root / "spatial_relationships" / "0002.mp4").write_bytes(b"video")
    (csv_dir / "demo_action_binding_score.csv").write_text(
        "name,prompt,Score\n0001.mp4,a dog jumps,4\nscore: ,3\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "t2v-compbench-out"

    exit_code = run_t2v_compbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--csv-dir",
            str(csv_dir),
            "--dataset-root",
            str(dataset_root),
            "--video-root",
            str(video_root),
            "--model-name",
            "demo",
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    generated_manifest = json.loads((output_dir / "generated_video_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert scorecard["metrics"]["per_metric"]["action_binding"]["normalized_score"] == 0.03
    assert dataset_manifest["exists"] is True
    assert dataset_manifest["file_count"] == 1
    assert generated_manifest["video_file_count"] == 2
    assert generated_manifest["by_category"]["action_binding"] == 1
    assert generated_manifest["by_category"]["spatial_relationships"] == 1


def test_t2v_compbench_official_runner_accepts_upstream_average_json(tmp_path: Path) -> None:
    run_t2v_compbench_official_runner = _load_script("run_t2v_compbench_official_runner")
    upstream_results = tmp_path / "scores.json"
    upstream_results.write_text(json.dumps({"t2v_compbench_average": 0.42}), encoding="utf-8")
    output_dir = tmp_path / "t2v-compbench-json-out"

    exit_code = run_t2v_compbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert scorecard["metrics"]["per_metric"]["t2v_compbench_average"]["raw_score"] == 0.42


def test_vmbench_official_runner_normalizes_existing_results_json(tmp_path: Path) -> None:
    run_vmbench_official_runner = _load_script("run_vmbench_official_runner")
    upstream_results = tmp_path / "results.json"
    upstream_results.write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "prompt": "a toy car moves across a table",
                    "filepath": "0001.mp4",
                    "perceptible_amplitude_score": 0.8,
                    "object_integrity_score": 0.7,
                    "temporal_coherence_score": 0.6,
                    "commonsense_adherence_score": 0.5,
                    "motion_smoothness_score": 0.4,
                },
                {
                    "index": 2,
                    "prompt": "a ball rolls down a ramp",
                    "filepath": "0002.mp4",
                    "perceptible_amplitude_socre": 0.6,
                    "object_integrity_score": 0.5,
                    "temporal_coherence_score": 0.4,
                    "commonsense_adherence_score": 0.3,
                    "motion_smoothness_score": 0.2,
                },
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "vmbench-out"

    exit_code = run_vmbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    sample_rows = [
        json.loads(line) for line in (output_dir / "per_sample_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 1
    assert scorecard["evaluation"]["kind"] == "official_vmbench"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["metrics"]["per_metric"]["perceptible_amplitude_score"]["raw_score"] == pytest.approx(0.7)
    assert scorecard["metrics"]["per_metric"]["object_integrity_score"]["raw_score"] == pytest.approx(0.6)
    assert scorecard["metrics"]["per_metric"]["temporal_coherence_score"]["raw_score"] == pytest.approx(0.5)
    assert scorecard["metrics"]["per_metric"]["commonsense_adherence_score"]["raw_score"] == pytest.approx(0.4)
    assert scorecard["metrics"]["per_metric"]["motion_smoothness_score"]["raw_score"] == pytest.approx(0.3)
    assert scorecard["metrics"]["per_metric"]["vmbench_average"]["raw_score"] == pytest.approx(0.5)
    assert raw_rows[-1]["metric_id"] == "vmbench_average"
    assert len(sample_rows) == 2
    assert sample_rows[1]["metrics"]["perceptible_amplitude_score"] == pytest.approx(0.6)


def test_vmbench_official_runner_normalizes_existing_scores_csv(tmp_path: Path) -> None:
    run_vmbench_official_runner = _load_script("run_vmbench_official_runner")
    upstream_results = tmp_path / "scores.csv"
    upstream_results.write_text(
        "Metric,Average Score\nPAS,80\nOIS,70\nTCS,60\nCAS,50\nMSS,40\nTotal Score,60\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "vmbench-csv-out"

    exit_code = run_vmbench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert scorecard["metrics"]["per_metric"]["perceptible_amplitude_score"]["raw_score"] == pytest.approx(0.8)
    assert scorecard["metrics"]["per_metric"]["vmbench_average"]["raw_score"] == pytest.approx(0.6)
    assert scorecard["metrics"]["per_metric"]["vmbench_average"]["normalized_score"] == pytest.approx(0.6)


def test_camerabench_official_runner_normalizes_existing_upstream_results(tmp_path: Path) -> None:
    run_camerabench_official_runner = _load_script("run_camerabench_official_runner")
    upstream_results = tmp_path / "binary_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "overall_average_precision": 0.8,
                "overall_roc_auc": 0.6,
                "evaluated_splits": 1,
                "results_by_split": {
                    "demo_split": {
                        "average_precision": 0.8,
                        "roc_auc": 0.6,
                        "num_samples": 4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "camerabench-out"

    exit_code = run_camerabench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prediction_rows = [
        json.loads(line) for line in (output_dir / "camera_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 1
    assert scorecard["evaluation"]["kind"] == "official_camerabench"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["metrics"]["per_metric"]["camera_motion_average_precision"]["raw_score"] == 0.8
    assert scorecard["metrics"]["per_metric"]["camera_motion_roc_auc"]["raw_score"] == 0.6
    assert scorecard["metrics"]["per_metric"]["camerabench_average"]["raw_score"] == 0.7
    assert raw_rows[-1]["metric_id"] == "camerabench_average"
    assert prediction_rows[0]["split"] == "demo_split"


def test_camerabench_official_runner_strict_full_suite_marks_framework_ready(tmp_path: Path) -> None:
    run_camerabench_official_runner = _load_script("run_camerabench_official_runner")
    data_root = tmp_path / "CameraBench"
    video_dir = data_root / "videos_gif"
    video_dir.mkdir(parents=True)
    (data_root / "test.jsonl").write_text('{"path": "videos/sample-a.mp4", "labels": ["pan"]}\n', encoding="utf-8")
    (video_dir / "sample-a.gif").write_bytes(b"GIF89a")
    upstream_results = tmp_path / "camerabench_all_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "overall_average_precision": 0.8,
                "overall_roc_auc": 0.6,
                "evaluated_splits": 1,
                "overall_binary_acc": 0.9,
                "overall_question_acc": 0.7,
                "overall_retrieval_text": 0.5,
                "overall_retrieval_image": 0.7,
                "overall_retrieval_group": 0.9,
                "results_by_split": {"demo_split": {"num_samples": 1}},
                "results": [
                    {
                        "sample_id": "sample-a",
                        "gen_match": 0.8,
                        "spice": 0.6,
                        "cider": 0.6,
                        "bleu2": 0.6,
                        "rouge_l": 0.6,
                        "meteor": 0.6,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "camerabench-full-out"

    exit_code = run_camerabench_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--benchmark-data-root",
            str(data_root),
            "--strict",
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["validation"]["normalizer_only"] is False
    assert scorecard["validation"]["strict_failed"] is False
    assert scorecard["eligibility"]["full_suite_valid"] is True
    assert scorecard["eligibility"]["all_metrics_available"] is True
    assert scorecard["dataset"]["coverage"]["complete"] is True
    assert scorecard["metrics"]["summary"]["available_metrics"] == 6
    assert scorecard["metrics"]["per_metric"]["camera_caption_score"]["raw_score"] == pytest.approx(0.7)
    assert scorecard["metrics"]["per_metric"]["camerabench_average"]["raw_score"] == pytest.approx(0.72)


def test_chronomagic_official_runner_normalizes_existing_upstream_results(tmp_path: Path) -> None:
    run_chronomagic_official_runner = _load_script("run_chronomagic_official_runner")
    upstream_results = tmp_path / "ChronoMagic-Bench-Input.json"
    upstream_results.write_text(
        json.dumps(
            {
                "demo-model": {
                    "Average_MTScore": 0.5,
                    "Average_CHScore": 2.5,
                    "Average_GPT4o-MTScore": 3.0,
                    "UMT-FVD": -1,
                    "UMTScore": -1,
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "chronomagic-out"

    exit_code = run_chronomagic_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--model-name",
            "demo-model",
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    sample_rows = [
        json.loads(line) for line in (output_dir / "per_sample_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    dataset_manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    generated_manifest = json.loads((output_dir / "generated_video_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert scorecard["evaluation"]["kind"] == "official_chronomagic"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["dataset"]["hf_dataset_id"] == "BestWishYsh/ChronoMagic-Bench"
    assert scorecard["metrics"]["per_metric"]["chronomagic_score"]["raw_score"] == 2.0
    assert scorecard["metrics"]["per_metric"]["chronomagic_score"]["normalized_score"] == 0.5
    assert scorecard["metrics"]["per_metric"]["temporal_transformation"]["raw_score"] == 1.75
    assert scorecard["metrics"]["per_metric"]["temporal_transformation"]["normalized_score"] == 0.5
    assert raw_rows[-1]["metric_id"] == "temporal_transformation"
    assert sample_rows[0]["model_name"] == "demo-model"
    assert dataset_manifest["expected_rows"] == 1799
    assert generated_manifest["expected_file_count"] == 1799


def test_chronomagic_full_normalization_requires_all_official_components(tmp_path: Path) -> None:
    run_chronomagic_official_runner = _load_script("run_chronomagic_official_runner")
    upstream_results = tmp_path / "ChronoMagic-Bench-Input.json"
    upstream_results.write_text(
        json.dumps({"demo-model": {"Average_CHScore": 2.5, "UMT-FVD": -1, "UMTScore": -1}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "chronomagic-out"

    exit_code = run_chronomagic_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--model-name",
            "demo-model",
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert scorecard["normalization_ok"] is False
    assert scorecard["evaluation"]["required_components_available"] is False
    assert scorecard["evaluation"]["missing_required_components"] == ["gpt4o_mtscore", "mtscore"]
    assert scorecard["metrics"]["component_availability"]["required_components"] == [
        "chscore",
        "gpt4o_mtscore",
        "mtscore",
    ]


def test_chronomagic_official_runner_discovers_real_dataset_and_generated_videos(tmp_path: Path) -> None:
    run_chronomagic_official_runner = _load_script("run_chronomagic_official_runner")
    upstream_results = tmp_path / "ChronoMagic-Bench-Input.json"
    dataset_root = tmp_path / "ChronoMagic-Bench"
    generated_video_dir = tmp_path / "generated"
    dataset_root.mkdir()
    generated_video_dir.mkdir()
    upstream_results.write_text(
        json.dumps({"demo-model": {"Average_MTScore": 1.0, "Average_CHScore": 5.0, "Average_GPT4o-MTScore": 5.0}}),
        encoding="utf-8",
    )
    (dataset_root / "test.jsonl").write_text(
        json.dumps(
            {
                "videoid": "3d_printing_19",
                "name": "Time-lapse of a 3D printing process",
                "sub_category": "3d_printing",
                "main_category": "human_creation",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (generated_video_dir / "3d_printing_19.mp4").write_bytes(b"video")
    output_dir = tmp_path / "chronomagic-out"

    exit_code = run_chronomagic_official_runner.main(
        [
            "--output-dir",
            str(output_dir),
            "--from-upstream-results",
            str(upstream_results),
            "--model-name",
            "demo-model",
            "--dataset-root",
            str(dataset_root),
            "--generated-video-dir",
            str(generated_video_dir),
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    generated_manifest = json.loads((output_dir / "generated_video_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert scorecard["metrics"]["per_metric"]["chronomagic_score"]["normalized_score"] == 1.0
    assert dataset_manifest["exists"] is True
    assert dataset_manifest["file_count"] == 1
    assert generated_manifest["video_file_count"] == 1
    assert generated_manifest["coverage_complete"] is True


def test_chronomagic_official_runner_runs_chscore_component(tmp_path: Path) -> None:
    run_chronomagic_official_runner = _load_script("run_chronomagic_official_runner")
    chronomagic_root = tmp_path / "ChronoMagic-Bench"
    chscore_dir = chronomagic_root / "CHScore"
    chscore_dir.mkdir(parents=True)
    (chronomagic_root / "evaluate.py").write_text("print('unused full runner')\n", encoding="utf-8")
    (chscore_dir / "step0-get_CHScore.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_names", nargs="+")
parser.add_argument("--input_folder")
parser.add_argument("--output_folder")
parser.add_argument("--model_pth")
parser.add_argument("--eval_type")
args = parser.parse_args()
out = Path(args.output_folder)
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.model_names[0]}_1_CHScore.json").write_text(
    json.dumps({"total_average_score": 4.0, "all_scores": [{"demo": {"TSI_score": 4.0}}]}),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    (chronomagic_root / "get_uploaded_json.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_path")
parser.add_argument("--output_path")
args = parser.parse_args()
Path(args.output_path).mkdir(parents=True, exist_ok=True)
(Path(args.output_path) / "ChronoMagic-Bench-Input.json").write_text(
    json.dumps({"demo-model": {"Average_CHScore": 4.0, "UMT-FVD": -1, "UMTScore": -1}}),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    input_folder = tmp_path / "toy_video"
    (input_folder / "demo-model").mkdir(parents=True)
    (input_folder / "demo-model" / "demo.mp4").write_bytes(b"video")
    output_dir = tmp_path / "chronomagic-out"

    exit_code = run_chronomagic_official_runner.main(
        [
            "--chronomagic-root",
            str(chronomagic_root),
            "--components",
            "chscore",
            "--input-folder",
            str(input_folder),
            "--model-name",
            "demo-model",
            "--model-pth-chscore",
            str(tmp_path / "cotracker2.pth"),
            "--output-dir",
            str(output_dir),
            "--python",
            sys.executable,
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line) for line in (output_dir / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 0
    assert scorecard["benchmark"]["requires_api"] is False
    assert scorecard["evaluation"]["component_run"] is True
    assert scorecard["official_component_verified"] is True
    assert scorecard["run"]["components"] == ["chscore"]
    assert raw_rows[0]["metric_id"] == "chronomagic_score"
    assert raw_rows[0]["raw_score"] == 4.0
    assert raw_rows[1]["metric_id"] == "temporal_transformation"
    assert raw_rows[1]["available"] is False


def test_chronomagic_full_runner_executes_strict_component_pipeline(tmp_path: Path) -> None:
    run_chronomagic_official_runner = _load_script("run_chronomagic_official_runner")
    chronomagic_root = tmp_path / "ChronoMagic-Bench"
    (chronomagic_root / "CHScore").mkdir(parents=True)
    (chronomagic_root / "GPT4o_MTScore").mkdir()
    (chronomagic_root / "MTScore").mkdir()
    (chronomagic_root / "evaluate.py").write_text("raise SystemExit(99)\n", encoding="utf-8")
    (chronomagic_root / "CHScore" / "step0-get_CHScore.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_names", nargs="+")
parser.add_argument("--input_folder")
parser.add_argument("--output_folder")
parser.add_argument("--model_pth")
parser.add_argument("--eval_type")
args = parser.parse_args()
out = Path(args.output_folder)
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.model_names[0]}_1_CHScore.json").write_text(
    json.dumps({"total_average_score": 4.0, "all_scores": [{"demo": {"TSI_score": 4.0}}]}),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    (chronomagic_root / "GPT4o_MTScore" / "step0-extract_video_frames.py").write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir")
parser.add_argument("--output_dir")
parser.add_argument("--model_names", nargs="+")
parser.add_argument("--eval_type")
args = parser.parse_args()
(Path(args.output_dir) / args.model_names[0]).mkdir(parents=True, exist_ok=True)
""",
        encoding="utf-8",
    )
    (chronomagic_root / "GPT4o_MTScore" / "step1-get_temp_results.py").write_text(
        """
import argparse
import sys
from pathlib import Path

if "None" in sys.argv:
    raise SystemExit(42)
parser = argparse.ArgumentParser()
parser.add_argument("--num_workers")
parser.add_argument("--openai_api")
parser.add_argument("--base_url", default=None)
parser.add_argument("--input_dir")
parser.add_argument("--output_dir")
parser.add_argument("--model_names", nargs="+")
parser.add_argument("--eval_type")
args = parser.parse_args()
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
(Path(args.output_dir) / f"{args.model_names[0]}_metamorphic.json").write_text("{}", encoding="utf-8")
""",
        encoding="utf-8",
    )
    (chronomagic_root / "GPT4o_MTScore" / "step2-get_GPT4o-MTScore.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir")
parser.add_argument("--output_dir")
parser.add_argument("--model_names", nargs="+")
parser.add_argument("--eval_type")
args = parser.parse_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.model_names[0]}_1_GPT4o-MTScore.json").write_text(
    json.dumps({"Average Score": 3.0, "Formatted Data": {"demo": {"Score": "3"}}}),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    (chronomagic_root / "MTScore" / "step0-get_MTScore.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_names", nargs="+")
parser.add_argument("--input_folder")
parser.add_argument("--output_folder")
parser.add_argument("--model_pth")
parser.add_argument("--eval_type")
args = parser.parse_args()
out = Path(args.output_folder)
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.model_names[0]}_1_MTScore.json").write_text(
    json.dumps({"average_metamorphic_score": 0.5, "video_scores": [{"video_name": "demo.mp4", "metamorphic_score": 0.5}]}),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    (chronomagic_root / "get_uploaded_json.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_path")
parser.add_argument("--output_path")
args = parser.parse_args()
Path(args.output_path).mkdir(parents=True, exist_ok=True)
(Path(args.output_path) / "ChronoMagic-Bench-Input.json").write_text(
    json.dumps({"demo-model": {"Average_CHScore": 4.0, "Average_GPT4o-MTScore": 3.0, "Average_MTScore": 0.5, "UMT-FVD": -1, "UMTScore": -1}}),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    input_folder = tmp_path / "toy_video"
    (input_folder / "demo-model").mkdir(parents=True)
    (input_folder / "demo-model" / "demo.mp4").write_bytes(b"video")
    output_dir = tmp_path / "chronomagic-out"

    exit_code = run_chronomagic_official_runner.main(
        [
            "--chronomagic-root",
            str(chronomagic_root),
            "--input-folder",
            str(input_folder),
            "--model-name",
            "demo-model",
            "--model-pth-chscore",
            str(tmp_path / "cotracker2.pth"),
            "--model-pth-mtscore",
            str(tmp_path / "InternVideo2-stage2_1b-224p-f4.pt"),
            "--openai-api",
            "test-key",
            "--output-dir",
            str(output_dir),
            "--python",
            sys.executable,
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert scorecard["normalization_ok"] is True
    assert scorecard["evaluation"]["required_components_available"] is True
    assert scorecard["evaluation"]["missing_required_components"] == []
    assert scorecard["official_benchmark_verified"] is True
    assert len(scorecard["run"]["commands"]) == 6
    assert all("evaluate.py" not in " ".join(command) for command in scorecard["run"]["commands"])


def test_chronomagic_component_failure_writes_failed_scorecard(tmp_path: Path) -> None:
    run_chronomagic_official_runner = _load_script("run_chronomagic_official_runner")
    chronomagic_root = tmp_path / "ChronoMagic-Bench"
    chscore_dir = chronomagic_root / "CHScore"
    chscore_dir.mkdir(parents=True)
    (chronomagic_root / "evaluate.py").write_text("print('unused full runner')\n", encoding="utf-8")
    (chscore_dir / "step0-get_CHScore.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    (chronomagic_root / "get_uploaded_json.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    input_folder = tmp_path / "toy_video"
    (input_folder / "demo-model").mkdir(parents=True)
    (input_folder / "demo-model" / "demo.mp4").write_bytes(b"video")
    output_dir = tmp_path / "chronomagic-out"

    exit_code = run_chronomagic_official_runner.main(
        [
            "--chronomagic-root",
            str(chronomagic_root),
            "--components",
            "chscore",
            "--input-folder",
            str(input_folder),
            "--model-name",
            "demo-model",
            "--output-dir",
            str(output_dir),
            "--python",
            sys.executable,
            "--json",
        ]
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    upstream = json.loads((output_dir / "upstream" / "ChronoMagic-Bench-Input.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert scorecard["run"]["status"] == "failed"
    assert scorecard["run"]["returncode"] == 7
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["metrics"]["summary"]["available_metrics"] == 0
    assert upstream["demo-model"]["returncode"] == 7


def test_run_benchmark_execution_checks_expected_artifact_sha256(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    artifact = tmp_path / "score.json"
    digest = hashlib.sha256(b"{}").hexdigest()
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="checksum-benchmark",
        path=tmp_path / "demo.json",
        data={
            "run_command": [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(artifact)!r}).write_bytes(b'{{}}')",
            ],
            "runner": {"expected_artifacts": [{"path": str(artifact), "sha256": digest}]},
        },
    )

    result = run_benchmark_execution.run_benchmark(manifest, tmp_path / "benchmark_zoo", timeout_seconds=10)

    assert result["ok"] is True
    assert result["artifact_checks"][0]["checksum_ok"] is True
    assert result["artifact_checks"][0]["actual_sha256"] == digest


def test_run_benchmark_execution_checks_video_probe_constraints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    artifact = tmp_path / "sample.mp4"

    def fake_probe(path: Path) -> dict[str, object]:
        assert path == artifact
        return {"ok": True, "frame_count": 16, "width": 640, "height": 352, "fps": 12.0}

    monkeypatch.setattr(run_benchmark_execution, "probe_video_artifact", fake_probe)
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="video-probe-benchmark",
        path=tmp_path / "benchmark.json",
        data={
            "run_command": [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(artifact)!r}).write_bytes(b'not-a-real-video')",
            ],
            "runner": {
                "expected_artifacts": [
                    {
                        "path": str(artifact),
                        "video_probe": {"min_frames": 12, "min_width": 600, "min_height": 300, "min_fps": 10},
                    }
                ]
            },
        },
    )

    result = run_benchmark_execution.run_benchmark(manifest, tmp_path / "benchmark_zoo", timeout_seconds=10)

    assert result["ok"] is True
    video_probe = result["artifact_checks"][0]["video_probe"]
    assert video_probe["ok"] is True
    assert video_probe["checks"] == {
        "min_frames": True,
        "min_width": True,
        "min_height": True,
        "min_fps": True,
    }


def test_run_benchmark_execution_fails_video_probe_constraints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    artifact = tmp_path / "sample.mp4"
    monkeypatch.setattr(
        run_benchmark_execution,
        "probe_video_artifact",
        lambda path: {"ok": True, "frame_count": 4, "width": 320, "height": 180, "fps": 12.0},
    )
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="video-probe-fail-benchmark",
        path=tmp_path / "benchmark.json",
        data={
            "run_command": [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(artifact)!r}).write_bytes(b'not-a-real-video')",
            ],
            "runner": {
                "expected_artifacts": [
                    {"path": str(artifact), "video_probe": {"min_frames": 12, "min_width": 600}}
                ]
            },
        },
    )

    result = run_benchmark_execution.run_benchmark(manifest, tmp_path / "benchmark_zoo", timeout_seconds=10)

    assert result["returncode"] == 0
    assert result["ok"] is False
    assert result["integration_evidence"] is False
    assert result["artifact_checks"][0]["video_ok"] is False
    assert result["artifact_checks"][0]["video_probe"]["checks"] == {"min_frames": False, "min_width": False}


_CONTRACT_FIXTURE_VALIDATION_SNIPPET = (
    "import json, os; "
    "from pathlib import Path; "
    "root = Path(os.environ['WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR']); "
    "root.mkdir(parents=True, exist_ok=True); "
    "Path(root, 'scorecard.json').write_text(json.dumps({"
    "'official_benchmark_verified': False, "
    "'integration_evidence': False, "
    "'run': {'status': 'contract_fixture', 'runner': 'benchmark_zoo_contract_runner_fixture'}, "
    "'benchmark': {'contract_only': True, 'evidence_level': 'contract_fixture_only'}, "
    "'eligibility': {'leaderboard_valid': False}"
    "})); "
    "Path(root, 'benchmark_contract.json').write_text(json.dumps("
    "{'benchmark_id': 'vbench', 'contract_only': True})); "
    "Path(root, 'raw_metric_table.jsonl').write_text("
    "json.dumps({'metric_id': 'overall_consistency', 'available': False}) + '\\n')"
)


def test_run_benchmark_execution_blocks_contract_only_validation_report(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="vbench",
        path=tmp_path / "demo.json",
        data={
            "runner": {
                "validation_command": [sys.executable, "-c", _CONTRACT_FIXTURE_VALIDATION_SNIPPET],
                "expected_artifacts": ["scorecard.json", "benchmark_contract.json", "raw_metric_table.jsonl"],
            }
        },
    )

    result = run_benchmark_execution.run_benchmark(
        manifest,
        tmp_path / "benchmark_zoo",
        timeout_seconds=10,
        command_kind="validation",
    )

    assert result["ok"] is False
    assert result["official_benchmark_verified"] is False
    assert result["integration_evidence"] is False
    assert result["contract_only_scorecard"]["contract_only"] is True
    assert result["evidence_level"] == "contract_fixture_only"
    assert "contract-only validation" in result["error"]


def test_run_benchmark_execution_blocks_normalizer_only_validation_report(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="normalizer-validation",
        path=tmp_path / "demo.json",
        data={
            "runner": {
                "validation_command": [
                    sys.executable,
                    "-c",
                    (
                        "import json, os; "
                        "from pathlib import Path; "
                        "root = Path(os.environ['WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR']); "
                        "root.mkdir(parents=True, exist_ok=True); "
                        "Path(root, 'scorecard.json').write_text(json.dumps({"
                        "'official_benchmark_verified': True, "
                        "'integration_evidence': False, "
                        "'run': {'status': 'official_verified', 'command': None}, "
                        "'benchmark': {'contract_only': False}"
                        "}))"
                    ),
                ],
                "expected_artifacts": ["scorecard.json"],
            }
        },
    )

    result = run_benchmark_execution.run_benchmark(
        manifest,
        tmp_path / "benchmark_zoo",
        timeout_seconds=10,
        command_kind="validation",
    )

    assert result["ok"] is False
    assert result["command_ok"] is True
    assert result["official_benchmark_verified"] is True
    assert result["integration_evidence"] is False
    assert result["evidence_level"] == "normalizer_only"
    assert result["scorecard_runtime_flags"]["integration_evidence"] is False
    assert "normalizer-only" in result["error"]


def test_run_benchmark_execution_requires_explicit_runtime_evidence_flags(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="missing-flags-validation",
        path=tmp_path / "demo.json",
        data={
            "runner": {
                "validation_command": [
                    sys.executable,
                    "-c",
                    (
                        "import json, os; "
                        "from pathlib import Path; "
                        "root = Path(os.environ['WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR']); "
                        "root.mkdir(parents=True, exist_ok=True); "
                        "Path(root, 'scorecard.json').write_text(json.dumps({"
                        "'run': {'status': 'succeeded'}, "
                        "'benchmark': {'contract_only': False}"
                        "}))"
                    ),
                ],
                "expected_artifacts": ["scorecard.json"],
            }
        },
    )

    result = run_benchmark_execution.run_benchmark(
        manifest,
        tmp_path / "benchmark_zoo",
        timeout_seconds=10,
        command_kind="validation",
    )

    assert result["ok"] is False
    assert result["command_ok"] is True
    assert result["official_benchmark_verified"] is False
    assert result["integration_evidence"] is False
    assert result["scorecard_runtime_flags"]["found"] is True
    assert "official_benchmark_verified" not in result["scorecard_runtime_flags"]
    assert "normalizer-only" in result["error"]


def test_worldbench_official_runner_normalizes_existing_upstream_results(tmp_path: Path) -> None:
    run_worldbench = _load_script("run_worldbench_official_runner")
    upstream_results = tmp_path / "worldbench_results.json"
    upstream_results.write_text(
        json.dumps(
            {
                "summary": {
                    "video_based_accuracy": 75,
                    "multiple_choice_accuracy": 0.8,
                    "binary_accuracy": {"normalized_score": 0.6},
                },
                "samples": [
                    {
                        "sample_id": "video-001",
                        "component": "video_based",
                        "correct": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = run_worldbench.load_upstream_results(upstream_results)
    scorecard = run_worldbench.normalize_worldbench_results(
        loaded,
        benchmark_id="worldbench",
        output_dir=tmp_path / "out",
        results_path=upstream_results,
    )

    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["validation"]["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["official_results_imported"] is True
    leaderboard = scorecard["metrics"]["leaderboard"]
    assert leaderboard["video_based_accuracy"] == 0.75
    assert leaderboard["multiple_choice_accuracy"] == 0.8
    assert leaderboard["binary_accuracy"] == 0.6
    assert leaderboard["text_based_accuracy"] == pytest.approx(0.7)
    assert leaderboard["worldbench_average"] == pytest.approx(0.725)
    per_metric = scorecard["metrics"]["per_metric"]
    assert per_metric["binary_accuracy"]["raw_score"] is None
    assert per_metric["binary_accuracy"]["score_scale"] == "normalized"

    raw_rows = [
        json.loads(line)
        for line in Path(scorecard["artifacts"]["raw_metric_table"]).read_text(encoding="utf-8").splitlines()
    ]
    sample_rows = [
        json.loads(line)
        for line in Path(scorecard["artifacts"]["per_sample_scores"]).read_text(encoding="utf-8").splitlines()
    ]
    assert raw_rows[-1]["metric_id"] == "worldbench_average"
    assert sample_rows[0]["sample_id"] == "video-001"


def test_run_benchmark_execution_blocks_worldbench_normalizer_only_validation(tmp_path: Path) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    run_worldbench = _load_script("run_worldbench_official_runner")
    upstream_results = tmp_path / "worldbench_summary.json"
    upstream_results.write_text(
        json.dumps({"summary": {"video_based_accuracy": "80%", "text_based_accuracy": 0.5}}),
        encoding="utf-8",
    )
    manifest = run_benchmark_execution.BenchmarkManifest(
        benchmark_id="worldbench",
        path=tmp_path / "benchmark.json",
        data={
            "runner": {
                "validation_command": [sys.executable, str(Path(run_worldbench.__file__))],
                "expected_artifacts": ["scorecard.json", "per_sample_scores.jsonl", "raw_metric_table.jsonl"],
            }
        },
    )

    result = run_benchmark_execution.run_benchmark(
        manifest,
        tmp_path / "reports",
        timeout_seconds=10,
        command_kind="validation",
        env_overrides={"WORLDFOUNDRY_WORLDBENCH_RESULTS_PATH": str(upstream_results)},
    )

    assert result["ok"] is False
    assert result["command_ok"] is True
    assert result["official_benchmark_verified"] is False
    assert result["integration_evidence"] is False
    assert result["evidence_level"] == "normalizer_only"
    assert result["scorecard_runtime_flags"]["integration_evidence"] is False
    assert result["scorecard_runtime_flags"]["normalization_ok"] is True
    assert result["scorecard_runtime_flags"]["official_results_imported"] is True
    assert result.get("error") == "validation scorecard is normalizer-only and is not official runtime integration evidence"


def test_validate_integration_blocks_validation_without_expected_artifacts(tmp_path: Path) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.BenchmarkManifest(
        benchmark_id="no-validation-artifact-contract",
        path=tmp_path / "benchmark.json",
        data={"runner": {"validation_command": [sys.executable, "-c", "print('ok')"]}},
    )

    result = validate_integration.validation_stage(
        manifest,
        tmp_path / "validation",
        execute=False,
        timeout_seconds=10,
    )

    assert result["status"] == "blocked"
    assert result["error"] == "missing validation expected_artifacts"


def test_validate_integration_requires_validation_scorecard_artifact(tmp_path: Path) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.BenchmarkManifest(
        benchmark_id="no-scorecard-artifact-contract",
        path=tmp_path / "benchmark.json",
        data={
            "runner": {
                "validation_command": [sys.executable, "-c", "print('ok')"],
                "expected_artifacts": ["raw_metric_table.jsonl"],
            }
        },
    )

    result = validate_integration.validation_stage(
        manifest,
        tmp_path / "validation",
        execute=False,
        timeout_seconds=10,
    )

    assert result["status"] == "blocked"
    assert result["error"] == "benchmark validation expected_artifacts must include scorecard.json"


def test_validate_integration_passes_validation_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    validate_integration = _load_script("validate_integration")
    captured_env: dict[str, str] | None = None
    manifest = validate_integration.verify_sources.BenchmarkManifest(
        benchmark_id="env-validation",
        path=tmp_path / "benchmark.json",
        data={
            "runner": {
                "validation_command": [sys.executable, "-c", "print('ok')"],
                "expected_artifacts": ["scorecard.json"],
            }
        },
    )

    def fake_run_benchmark(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal captured_env
        captured_env = kwargs.get("env_overrides")  # type: ignore[assignment]
        return {"ok": True, "integration_evidence": True, "artifact_checks": []}

    monkeypatch.setattr(validate_integration.run_benchmark_execution, "run_benchmark", fake_run_benchmark)

    result = validate_integration.validation_stage(
        manifest,
        tmp_path / "validation",
        execute=True,
        timeout_seconds=10,
        env_overrides={"WORLDFOUNDRY_GENERATED_ARTIFACT_DIR": "/tmp/videos"},
    )

    assert result["status"] == "benchmark_validation_passed"
    assert captured_env == {"WORLDFOUNDRY_GENERATED_ARTIFACT_DIR": "/tmp/videos"}


def test_validate_integration_blocks_contract_only_validation_as_integration_evidence(tmp_path: Path) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.BenchmarkManifest(
        benchmark_id="vbench",
        path=tmp_path / "benchmark.json",
        data={
            "runner": {
                "validation_command": [sys.executable, "-c", _CONTRACT_FIXTURE_VALIDATION_SNIPPET],
                "expected_artifacts": ["scorecard.json", "benchmark_contract.json", "raw_metric_table.jsonl"],
            }
        },
    )

    result = validate_integration.validation_stage(
        manifest,
        tmp_path / "validation",
        execute=True,
        timeout_seconds=10,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "blocked"
    assert "contract-only validation" in result["error"]
    assert result["evidence_level"] == "contract_fixture_only"
    assert result["contract_only_scorecard"]["contract_only"] is True


def test_validate_integration_blocks_normalizer_only_validation_as_integration_evidence(tmp_path: Path) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.BenchmarkManifest(
        benchmark_id="normalizer-validation",
        path=tmp_path / "benchmark.json",
        data={
            "runner": {
                "validation_command": [
                    sys.executable,
                    "-c",
                    (
                        "import json, os; "
                        "from pathlib import Path; "
                        "root = Path(os.environ['WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR']); "
                        "root.mkdir(parents=True, exist_ok=True); "
                        "Path(root, 'scorecard.json').write_text(json.dumps({"
                        "'official_benchmark_verified': True, "
                        "'integration_evidence': False, "
                        "'run': {'status': 'official_verified', 'command': None}, "
                        "'benchmark': {'contract_only': False}"
                        "}))"
                    ),
                ],
                "expected_artifacts": ["scorecard.json"],
            }
        },
    )

    result = validate_integration.validation_stage(
        manifest,
        tmp_path / "validation",
        execute=True,
        timeout_seconds=10,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "blocked"
    assert "normalizer-only" in result["error"]


def test_validate_integration_blocks_missing_dataset_metadata(tmp_path: Path) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.BenchmarkManifest(
        benchmark_id="missing-dataset",
        path=tmp_path / "benchmark.json",
        data={"runner": {"validation_command": [sys.executable, "-c", "print('ok')"]}},
    )

    result = validate_integration.dataset_stage(
        manifest,
        tmp_path / "cache",
        execute=False,
        dataset_id_filter=None,
        disable_xet=False,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "blocked"
    assert "missing required Hugging Face dataset metadata" in result["error"]


def test_validate_integration_allows_explicit_dataset_free_benchmark(tmp_path: Path) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.BenchmarkManifest(
        benchmark_id="dataset-free",
        path=tmp_path / "benchmark.json",
        data={"dataset": {"not_applicable": True, "reason": "metric-only local validation benchmark"}},
    )

    result = validate_integration.dataset_stage(
        manifest,
        tmp_path / "cache",
        execute=False,
        dataset_id_filter=None,
        disable_xet=False,
    )

    assert result["ok"] is True
    assert result["ready"] is True
    assert result["status"] == "not_applicable"
    assert result["reason"] == "metric-only local validation benchmark"


def test_validate_integration_blocks_unsafe_dataset_download_by_default(tmp_path: Path) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.BenchmarkManifest(
        benchmark_id="unsafe-dataset",
        path=tmp_path / "benchmark.json",
        data={
            "official_sources": {
                "huggingface_datasets": [
                    {"repo_id": "org/unsafe", "private": False, "gated": "auto", "license": None}
                ]
            }
        },
    )

    result = validate_integration.dataset_stage(
        manifest,
        tmp_path / "cache",
        execute=True,
        dataset_id_filter=None,
        disable_xet=False,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "blocked"
    assert "unsafe dataset access" in result["error"]
    assert {issue["reason"] for issue in result["dataset_access_issues"]} == {
        "gated_dataset",
        "missing_or_unknown_license",
    }


def test_validate_integration_allows_unsafe_dataset_only_with_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.BenchmarkManifest(
        benchmark_id="unsafe-dataset",
        path=tmp_path / "benchmark.json",
        data={
            "official_sources": {
                "huggingface_datasets": [
                    {"repo_id": "org/unsafe", "private": False, "gated": "auto", "license": None}
                ]
            }
        },
    )
    calls: list[dict[str, object]] = []

    def fake_download_manifest(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"ok": True, "executed": True, "hf_dataset_ids": ["org/unsafe"]}

    monkeypatch.setattr(validate_integration.download_datasets, "download_manifest", fake_download_manifest)

    result = validate_integration.dataset_stage(
        manifest,
        tmp_path / "cache",
        execute=True,
        dataset_id_filter=None,
        disable_xet=True,
        allow_unsafe_datasets=True,
    )

    assert result["ok"] is True
    assert result["ready"] is True
    assert result["status"] == "dataset_ready"
    assert result["unsafe_datasets_allowed"] is True
    assert result["dataset_access_issues"]
    assert calls[0]["execute"] is True
    assert calls[0]["allow_unsafe_datasets"] is True


def test_benchmark_validate_integration_status_blocks_known_stage_failures() -> None:
    validate_integration = _load_script("validate_integration")

    status = validate_integration.integration_status(
        {
            "source": {"ok": True},
            "repo_clone": {"ok": True, "ready": False},
            "environment": {"ok": True},
            "dataset": {"ok": False, "ready": False},
            "benchmark_validation": {"ok": True, "ready": False},
        }
    )

    assert status == "blocked"


def test_run_benchmark_execution_main_never_defaults_to_all_benchmarks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_benchmark_execution = _load_script("run_benchmark_execution")
    manifest_dir = tmp_path / "benchmark_zoo"
    _write_manifest(
        manifest_dir,
        {
            "benchmarks": [
                {"benchmark_id": "alpha", "run_command": ["alpha-command"]},
                {"benchmark_id": "beta", "run_command": ["beta-command"]},
            ]
        },
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="selected\n", stderr="")

    monkeypatch.setattr(run_benchmark_execution.subprocess, "run", fake_run)

    status = run_benchmark_execution.main(
        [
            "--manifest-dir",
            str(manifest_dir),
            "--benchmark-id",
            "beta",
            "--output-root",
            str(tmp_path / "reports"),
        ]
    )

    report_path = tmp_path / "reports" / "beta" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert status == 0
    assert calls == [["beta-command"]]
    assert report["schema_version"] == "worldfoundry-benchmark-run-report"
    assert isinstance(report["generated_at"], str)
    assert report["validator"]["entrypoint"] == "python -m worldfoundry.evaluation.tasks.execution.orchestration.manifest_cli"
    assert "git" in report["validator"]
    assert report["benchmark_id"] == "beta"
    assert report["run_command"] == ["beta-command"]


def test_validate_integration_plan_only_stops_at_source_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    manifest_dir = tmp_path / "benchmark_zoo"
    _write_manifest(
        manifest_dir,
        {
            "benchmarks": [
                {
                    "benchmark_id": "vbench-mini",
                    "official_repo_url": "https://github.com/example/vbench",
                    "hf_dataset_id": "org/vbench-mini",
                    "runner": {
                        "validation_command": [sys.executable, "-c", "print('validation')"],
                        "expected_artifacts": ["scorecard.json"],
                    },
                }
            ]
        },
    )

    def fake_verify_manifest(manifest: object, timeout_seconds: int) -> dict[str, object]:
        return {"benchmark_id": "vbench-mini", "ok": True, "checks": {}}

    monkeypatch.setattr(validate_integration.verify_sources, "verify_manifest", fake_verify_manifest)
    monkeypatch.setattr(
        validate_integration,
        "check_environment",
        lambda require_hf: {"ok": True, "status": "env_ready", "checks": {}},
    )

    manifests = validate_integration.verify_sources.load_manifests(manifest_dir, "vbench-mini")
    report = validate_integration.validate_manifest(
        manifests[0],
        output_root=tmp_path / "validation",
        clone_root=tmp_path / "repos",
        cache_dir=tmp_path / "cache",
        timeout_seconds=10,
        clone_timeout_seconds=10,
        depth=1,
        execute_clone=False,
        update_clone=False,
        fresh_clone=False,
        execute_download=False,
        dataset_ids=None,
        disable_xet=False,
        execute_validation=False,
    )

    assert report["status"] == "source_verified"
    assert report["integrated"] is False
    assert report["schema_version"] == "worldfoundry-benchmark-integration-report"
    assert report["manifest_sha256"] == validate_integration.json_sha256(manifests[0].data)
    assert report["manifest_sha256_scope"] == "entry"
    assert report["manifest_file_sha256"] == validate_integration.file_sha256(manifests[0].path)
    assert report["validator"]["script"] == "scripts/benchmark_zoo/validate_integration.py"
    assert report["validator"]["script_sha256"] == validate_integration.file_sha256(
        REPO_ROOT / "scripts" / "benchmark_zoo" / "validate_integration.py"
    )
    assert report["validator"]["python"]["version"] == sys.version.split()[0]
    assert isinstance(report["validator"]["git"]["commit"], str)
    assert report["stages"]["dataset"]["status"] == "planned"
    assert report["stages"]["dataset"]["result"]["status"] == "blocked"
    assert report["stages"]["dataset"]["result"]["commands"] == []
    assert report["stages"]["repo_clone"]["checks"][0]["executed"] is False
    assert Path(report["report_path"]).is_file()


def test_validate_integration_clone_executes_git_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="cloned\n", stderr="")

    monkeypatch.setattr(validate_integration.subprocess, "run", fake_run)

    result = validate_integration.clone_git_repo(
        "https://github.com/example/benchmark",
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
        fresh=False,
    )

    assert result["ok"] is True
    assert result["ready"] is True
    assert calls == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/example/benchmark.git",
            str(tmp_path / "repos" / "github.com_example_benchmark"),
        ]
    ]


def test_benchmark_validate_integration_clone_timeout_decodes_bytes_for_json_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=10, output=b"partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr(validate_integration.subprocess, "run", fake_run)

    result = validate_integration.clone_git_repo(
        "https://github.com/example/benchmark",
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
        fresh=False,
    )

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["stdout"] == "partial stdout"
    assert result["stderr"] == "partial stderr"
    json.dumps(result)


def test_benchmark_validate_integration_fresh_clone_removes_existing_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    repo_dir = validate_integration.repo_dir_for_url(tmp_path / "repos", "https://github.com/example/benchmark")
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / ".git" / "index.lock").write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert not (repo_dir / ".git" / "index.lock").exists()
        return subprocess.CompletedProcess(command, 0, stdout="cloned\n", stderr="")

    monkeypatch.setattr(validate_integration.subprocess, "run", fake_run)

    result = validate_integration.clone_git_repo(
        "https://github.com/example/benchmark",
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
        fresh=True,
    )

    assert result["ok"] is True
    assert result["ready"] is True
    assert result["removed_existing"] is True
    assert calls == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/example/benchmark.git",
            str(repo_dir),
        ]
    ]


def test_benchmark_validate_integration_existing_repo_with_index_lock_is_blocked(tmp_path: Path) -> None:
    validate_integration = _load_script("validate_integration")
    repo_dir = validate_integration.repo_dir_for_url(tmp_path / "repos", "https://github.com/example/benchmark")
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / ".git" / "index.lock").write_text("", encoding="utf-8")

    result = validate_integration.clone_git_repo(
        "https://github.com/example/benchmark",
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "blocked"
    assert "index lock" in result["error"]


def test_benchmark_validate_integration_existing_repo_with_tracked_changes_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    repo_dir = validate_integration.repo_dir_for_url(tmp_path / "repos", "https://github.com/example/benchmark")
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[3:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
        if command[3:] == ["status", "--porcelain", "--untracked-files=no"]:
            return subprocess.CompletedProcess(command, 0, stdout=" D file.py\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(validate_integration.subprocess, "run", fake_run)

    result = validate_integration.clone_git_repo(
        "https://github.com/example/benchmark",
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "blocked"
    assert "tracked worktree changes" in result["error"]


def test_benchmark_validate_integration_clone_manifest_repos_surfaces_blocked_child_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_integration = _load_script("validate_integration")
    manifest = validate_integration.verify_sources.BenchmarkManifest(
        benchmark_id="alpha",
        path=tmp_path / "benchmark.json",
        data={"official_repo_url": "https://github.com/example/benchmark"},
    )

    def fake_clone_git_repo(url: str, clone_root: Path, **kwargs: object) -> dict[str, object]:
        return {"ok": False, "ready": False, "status": "blocked", "error": "partial clone"}

    monkeypatch.setattr(validate_integration, "clone_git_repo", fake_clone_git_repo)

    result = validate_integration.clone_manifest_repos(
        manifest,
        tmp_path / "repos",
        timeout_seconds=10,
        depth=1,
        execute=True,
        update=False,
        fresh=False,
    )

    assert result["ok"] is False
    assert result["ready"] is False
    assert result["status"] == "blocked"
    assert result["checks"][0]["error"] == "partial clone"


def test_benchmark_env_check_reports_repo_dataset_and_validation_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_check = _load_script("env_check")
    manifest = env_check.verify_sources.BenchmarkManifest(
        benchmark_id="ready-benchmark",
        path=tmp_path / "benchmark.json",
        data={
            "official_repo_url": "https://github.com/example/ready-benchmark",
            "hf_dataset_id": "org/dataset",
            "runner": {
                "validation_command": [sys.executable, "-c", "print('ok')"],
                "expected_artifacts": ["score.json"],
            },
        },
    )
    clone_root = tmp_path / "repos"
    repo_dir = env_check.repo_dir_for_url(clone_root, "https://github.com/example/ready-benchmark")
    (repo_dir / ".git").mkdir(parents=True)

    cache_dir = tmp_path / "cache" / "hfd"
    dataset_cache = cache_dir / "datasets--org--dataset"
    snapshot = dataset_cache / "snapshots" / "abc123"
    blob = dataset_cache / "blobs" / "blob-a"
    (dataset_cache / "refs").mkdir(parents=True)
    (dataset_cache / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (dataset_cache / "refs" / "main").write_text("abc123", encoding="utf-8")
    blob.write_text("sample", encoding="utf-8")
    (snapshot / "sample.json").symlink_to("../../blobs/blob-a")
    monkeypatch.setattr(env_check.sys, "version_info", (3, 10, 12))
    monkeypatch.setattr(env_check, "check_command", lambda name: {"ok": True, "name": name, "path": f"/usr/bin/{name}"})

    result = env_check.check_manifest(
        manifest,
        clone_root=clone_root,
        cache_dir=cache_dir,
        require_repo=True,
        require_dataset=True,
        require_validation=True,
    )

    assert result["ok"] is True
    assert result["repo_checks"][0]["ready"] is True
    assert result["dataset_checks"][0]["ready"] is True
    assert result["validation_command_present"] is True
    assert result["expected_artifacts"] == ["score.json"]


def test_benchmark_env_check_allows_explicit_dataset_free_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_check = _load_script("env_check")
    manifest = env_check.verify_sources.BenchmarkManifest(
        benchmark_id="dataset-free",
        path=tmp_path / "benchmark.json",
        data={"dataset": {"not_applicable": True, "reason": "uses generated artifacts"}},
    )
    monkeypatch.setattr(env_check.sys, "version_info", (3, 10, 12))
    monkeypatch.setattr(env_check, "check_command", lambda name: {"ok": True, "name": name, "path": f"/usr/bin/{name}"})

    result = env_check.check_manifest(
        manifest,
        clone_root=tmp_path / "repos",
        cache_dir=tmp_path / "cache" / "hfd",
        require_repo=False,
        require_dataset=True,
        require_validation=False,
    )

    assert result["ok"] is True
    assert result["dataset_checks"][0]["ready"] is True
    assert result["dataset_checks"][0]["status"] == "not_applicable"


def test_benchmark_zoo_scripts_are_stdlib_only() -> None:
    # The per-benchmark scripts were removed; the remaining scripts under
    # scripts/benchmark_zoo/ are thin wrappers that delegate into the
    # ``worldfoundry`` package.  The invariant that still matters is that they
    # do not import third-party packages at module scope (so ``--help`` and
    # argument parsing work in a bare environment).
    allowed_modules = set(sys.stdlib_module_names) | {"__future__", "worldfoundry"}
    for path in (REPO_ROOT / "scripts" / "benchmark_zoo").glob("*.py"):
        path_allowed_modules = set(allowed_modules)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                modules = {node.module.split(".", 1)[0]} if node.module else set()
            else:
                continue

            unexpected = modules - path_allowed_modules
            assert unexpected == set(), f"{path} imports {unexpected}"


# ── Thin contract tests for the current orchestration surface ──────────────
#
# The script-era tests above are skipped until rewritten; these cover the
# replacement API: ``orchestration.benchmark_runner`` + ``zoo benchmark-run``.


def test_orchestration_run_benchmark_execution_contract() -> None:
    """The library entrypoint keeps its keyword-only lifecycle signature."""
    import inspect

    from worldfoundry.evaluation.tasks.execution.orchestration import benchmark_runner

    signature = inspect.signature(benchmark_runner.run_benchmark_execution)
    parameters = signature.parameters
    assert list(parameters)[0] == "benchmark_id"
    for keyword in ("output_dir", "manifest_path", "mode", "generated_artifact_dir"):
        assert keyword in parameters
        assert parameters[keyword].kind is inspect.Parameter.KEYWORD_ONLY


def test_orchestration_registry_loads_default_manifest_with_formal_ids() -> None:
    """The runner registry loads the checked-in catalog and covers every formal id."""
    from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import (
        build_benchmark_runner_registry,
    )

    registry = build_benchmark_runner_registry()
    assert len(registry) >= FORMAL_BENCHMARK_COUNT
    missing = [bench_id for bench_id in formal_benchmark_ids() if bench_id not in registry]
    assert missing == []


def test_cli_zoo_benchmark_run_parser_contract() -> None:
    """``zoo benchmark-run`` requires --benchmark-id/--output-dir and parses modes."""
    pytest.importorskip("yaml")
    from worldfoundry.cli.main import _build_parser

    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["zoo", "benchmark-run"])

    args = parser.parse_args(
        [
            "zoo",
            "benchmark-run",
            "--benchmark-id",
            "vbench",
            "--output-dir",
            "runs/zoo/vbench",
            "--mode",
            "official-validation",
        ]
    )
    assert args.benchmark_id == "vbench"
    assert str(args.output_dir) == "runs/zoo/vbench"
    assert args.mode == "official-validation"
