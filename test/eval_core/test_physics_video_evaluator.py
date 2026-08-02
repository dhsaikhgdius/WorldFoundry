from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import ManifestBenchmarkRunner
from worldfoundry.evaluation.tasks.official.physics_video import write_physics_video_evaluation
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
    DEFAULT_VIDEO_CATALOG_DIR,
    load_benchmark_catalog_shard_entries,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

def _write_dummy_video(root: Path, sample_id: str) -> Path:
    video_path = root / f"{sample_id}.mp4"
    video_path.write_bytes(b"dummy video bytes")
    video_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "readable": True,
                "width": 64,
                "height": 48,
                "fps": 12.0,
                "duration_seconds": 1.5,
                "frame_count": 18,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return video_path


def test_physics_video_evaluator_scores_dummy_artifact_and_blocks_official_metrics(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    _write_dummy_video(generated_dir, "sample-1")
    prompt_manifest = tmp_path / "prompts.json"
    prompt_manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "prompt_id": "sample-1",
                        "prompt": "A ball rolls down a ramp.",
                        "answer": "A",
                        "judge_response": "A",
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    write_physics_video_evaluation(
        benchmark_id="videophy2",
        display_name="VideoPhy2",
        official_metric_ids=("semantic_adherence", "videophy2_average"),
        output_dir=tmp_path / "eval",
        generated_artifact_dir=generated_dir,
        prompt_manifest=prompt_manifest,
        expected_width=64,
        expected_height=48,
        expected_fps=12.0,
        expected_duration_seconds=1.5,
    )
    scorecard = json.loads((tmp_path / "eval" / "scorecard.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (tmp_path / "eval" / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_metric = {row["metric_id"]: row for row in rows}

    assert scorecard["evaluation"]["kind"] == "in_tree_physics_video_contract_evaluator"
    assert scorecard["leaderboard_valid"] is False
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["metrics"]["local"]["generated_video_exists"] == 1.0
    assert scorecard["metrics"]["local"]["generated_video_readable"] == 1.0
    assert scorecard["metrics"]["local"]["mcqa_accuracy"] == 1.0
    assert by_metric["semantic_adherence"]["status"] == "blocked"
    assert by_metric["semantic_adherence"]["score"] is None
    assert "does not fabricate" in by_metric["videophy2_average"]["blocked_reason"]


def test_physics_video_manifest_runner_writes_local_checks_for_planned_contract(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    _write_dummy_video(generated_dir, "sample-1")
    prompt_manifest = tmp_path / "prompts.jsonl"
    prompt_manifest.write_text(
        json.dumps({"prompt_id": "sample-1", "prompt": "A block slides on ice."}) + "\n",
        encoding="utf-8",
    )
    entry = next(item for item in load_benchmark_catalog_shard_entries("video") if item.benchmark_id == "phygenbench")
    result = ManifestBenchmarkRunner(entry, manifest_path=DEFAULT_VIDEO_CATALOG_DIR / "phygenbench.yaml").evaluate(
        output_dir=tmp_path / "phygenbench",
        mode="contract",
        generated_artifact_dir=generated_dir,
        physics_prompt_manifest=prompt_manifest,
    )

    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in result.raw_results_path.read_text(encoding="utf-8").splitlines()]
    by_metric = {row["metric_id"]: row for row in raw_rows}

    assert result.ok is False
    assert result.metadata["evaluator"] == "in_tree_physics_video_contract_evaluator"
    assert scorecard["benchmark"]["benchmark_id"] == "phygenbench"
    assert scorecard["run"]["status"] == "in_tree_local_checks"
    assert scorecard["metrics"]["local"]["artifact_manifest_coverage"] == 1.0
    assert scorecard["metrics"]["per_metric"]["physical_commonsense"]["status"] == "blocked"
    assert by_metric["generated_video_duration_check"]["status"] in {"passed", "failed"}
    assert by_metric["physical_law_adherence"]["score"] is None


def test_physics_video_evaluator_blocks_when_no_artifact_or_manifest(tmp_path: Path) -> None:
    write_physics_video_evaluation(
        benchmark_id="phygenbench",
        display_name="PhyGenBench",
        official_metric_ids=("physical_law_adherence", "physics_iq_average"),
        output_dir=tmp_path / "blocked",
    )
    scorecard = json.loads((tmp_path / "blocked" / "scorecard.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line)
        for line in (tmp_path / "blocked" / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_metric = {row["metric_id"]: row for row in raw_rows}

    assert scorecard["dataset"]["generated_file_count"] == 0
    assert scorecard["evaluation"]["skip_count"] >= 2
    assert by_metric["artifact_manifest_coverage"]["status"] == "not_available"
    assert by_metric["generated_video_exists"]["status"] == "not_available"
    assert by_metric["physical_law_adherence"]["status"] == "blocked"


def test_physics_video_evaluator_imports_videophy2_official_labels(tmp_path: Path) -> None:
    official_results = tmp_path / "videophy2_results.json"
    official_results.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "sample-1",
                        "sa": 5,
                        "pc": 4,
                        "physics_rules_followed": ["gravity", "momentum"],
                        "physics_rules_unfollowed": ["mass"],
                    },
                    {
                        "id": "sample-2",
                        "sa": 2,
                        "pc": 5,
                        "physics_rules_followed": ["reflection"],
                        "physics_rules_unfollowed": [],
                    },
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    write_physics_video_evaluation(
        benchmark_id="videophy2",
        display_name="VideoPhy2",
        official_metric_ids=(
            "semantic_adherence",
            "physical_commonsense",
            "joint_score",
            "rule_classification_accuracy",
            "videophy2_average",
        ),
        output_dir=tmp_path / "videophy2",
        official_results_path=official_results,
    )
    scorecard = json.loads((tmp_path / "videophy2" / "scorecard.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (tmp_path / "videophy2" / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_metric = {row["metric_id"]: row for row in rows}

    assert by_metric["semantic_adherence"]["score"] == 0.7
    assert by_metric["physical_commonsense"]["score"] == 0.9
    assert by_metric["joint_score"]["score"] == 0.5
    assert by_metric["rule_classification_accuracy"]["score"] == 0.75
    assert scorecard["metrics"]["leaderboard"]["videophy2_average"] == 0.5
    assert scorecard["leaderboard_valid"] is False


def test_physics_video_evaluator_imports_phyground_scores(tmp_path: Path) -> None:
    official_results = tmp_path / "phyground_scores.json"
    official_results.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "video": "sample-1",
                        "SA": 4,
                        "PTV": 5,
                        "persistence": 3,
                        "physical": {
                            "laws": {
                                "gravity": {"score": 5, "status": "scored"},
                                "flow_dynamics": {"score": 4, "status": "scored"},
                                "reflection": {"score": 3, "status": "scored"},
                            }
                        },
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    write_physics_video_evaluation(
        benchmark_id="phyground",
        display_name="PhyGround",
        official_metric_ids=(
            "semantic_adherence",
            "physical_temporal_validity",
            "persistence",
            "solid_body_score",
            "fluid_dynamics_score",
            "optics_score",
            "phyground_overall",
        ),
        output_dir=tmp_path / "phyground",
        official_results_path=official_results,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "phyground" / "raw_metric_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_metric = {row["metric_id"]: row for row in rows}

    assert by_metric["semantic_adherence"]["score"] == 0.8
    assert by_metric["physical_temporal_validity"]["score"] == 1.0
    assert by_metric["solid_body_score"]["score"] == 1.0
    assert by_metric["fluid_dynamics_score"]["score"] == 0.8
    assert by_metric["optics_score"]["score"] == 0.6
    assert by_metric["phyground_overall"]["status"] == "passed"
