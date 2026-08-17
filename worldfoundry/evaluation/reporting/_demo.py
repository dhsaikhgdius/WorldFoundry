#!/usr/bin/env python
"""End-to-end demo of worldfoundry.evaluation.reporting.

Builds three comparable scorecards (same benchmark/protocol/dataset, three
different models) from scratch, then exercises every reporting artifact:

  * scorecard.json            (canonical run document)
  * summary.json + report.md  (compact summary + human-readable report)
  * index.json / index.jsonl / index.html   (aggregate + browser)
  * comparison.json + comparison.md         (side-by-side with baseline deltas)

Run:  python worldfoundry/evaluation/reporting/_demo.py
Open: the demo_out/index.html in a browser.
"""
from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.reporting import (
    build_run_index,
    build_scorecard,
    build_markdown_run_index,
    write_run_browser,
    write_run_comparison,
    write_run_index,
    write_run_report_artifacts,
    write_scorecard,
)

# Created in main(), not at import time: importing a module must not write
# into the package directory.
DEMO_DIR = Path(__file__).parent / "_demo_out"

# A single shared evaluation protocol so all three runs are *comparable*
# (identical comparison_identity strict fields -> same comparison_key).
BENCHMARK = {
    "benchmark_id": "vbench",
    "benchmark_name": "VBench",
    "benchmark_revision": "v2-3.0.0",
    "task_type": "text_to_video",
    "suite": "world-foundry",
    "evaluation_protocol": "vbench-official",
    "protocol_id": "vbench-official",
    "protocol_revision": "3.0.0",
    "protocol_config_hash": "sha256:vb-official-cfg-001",
    "protocol_fidelity": "official",
}
DATASET = {
    "dataset_id": "vbench-test-1600",
    "name": "VBench test split",
    "split": "test",
    "sample_count": 1600,
    "dataset_revision": "r1",
    "dataset_hash": "sha256:vbench-1600-v1",
    "data_fidelity": "full",
}
PROVENANCE = {
    "claim": {"leaderboard_candidate": True},
    "fidelity": {"evaluation": "official", "data": "full", "generation": "pinned"},
}

# Three models, three different score profiles. All on the SAME protocol so
# comparison_identity.compare_identities() returns no hard errors.
RUNS = [
    {
        "model": {"model_id": "cascade-xl", "model_name": "Cascade-XL", "model_type": "diffusion"},
        "run": {
            "run_id": "run-cascade-xl-20260727",
            "status": "succeeded",
            "started_at": "2026-07-27T09:01:00Z",
            "finished_at": "2026-07-27T11:42:00Z",
        },
        "metrics_summary": {
            "sample_count": 1600,
            "successful_samples": 1600,
            "failed_samples": 0,
            "leaderboard": {
                "overall_score": 83.41,
                "subject_consistency": 86.2,
                "temporal_flicker": 0.812,   # lower is better
                "spatial_quality": 81.7,
            },
            "per_metric": {
                "overall_score": {"higher_is_better": True},
                "subject_consistency": {"higher_is_better": True},
                "temporal_flicker": {"higher_is_better": False},
                "spatial_quality": {"higher_is_better": True},
            },
        },
        "leaderboard_evidence": {"official_full_suite": True, "passed": True},
    },
    {
        "model": {"model_id": "aurora-7b", "model_name": "Aurora-7B", "model_type": "diffusion"},
        "run": {
            "run_id": "run-aurora-7b-20260727",
            "status": "succeeded",
            "started_at": "2026-07-27T12:10:00Z",
            "finished_at": "2026-07-27T14:48:00Z",
        },
        "metrics_summary": {
            "sample_count": 1600,
            "successful_samples": 1588,
            "failed_samples": 12,
            "leaderboard": {
                "overall_score": 79.05,
                "subject_consistency": 82.4,
                "temporal_flicker": 0.776,
                "spatial_quality": 78.9,
            },
            "per_metric": {
                "overall_score": {"higher_is_better": True},
                "subject_consistency": {"higher_is_better": True},
                "temporal_flicker": {"higher_is_better": False},
                "spatial_quality": {"higher_is_better": True},
            },
        },
        "leaderboard_evidence": {"official_full_suite": True, "passed": True},
    },
    {
        # A baseline / older model — used as the comparison baseline.
        "model": {"model_id": "aurora-1b", "model_name": "Aurora-1B", "model_type": "diffusion"},
        "run": {
            "run_id": "run-aurora-1b-20260727",
            "status": "succeeded",
            "started_at": "2026-07-27T15:05:00Z",
            "finished_at": "2026-07-27T17:20:00Z",
        },
        "metrics_summary": {
            "sample_count": 1600,
            "successful_samples": 1600,
            "failed_samples": 0,
            "leaderboard": {
                "overall_score": 71.20,
                "subject_consistency": 74.8,
                "temporal_flicker": 0.741,
                "spatial_quality": 69.4,
            },
            "per_metric": {
                "overall_score": {"higher_is_better": True},
                "subject_consistency": {"higher_is_better": True},
                "temporal_flicker": {"higher_is_better": False},
                "spatial_quality": {"higher_is_better": True},
            },
        },
        "leaderboard_evidence": {"official_full_suite": True, "passed": True},
    },
]


def build_one(run_spec: dict, index: int) -> Path:
    """Build a scorecard in its own run directory and emit summary + report."""
    run_dir = DEMO_DIR / f"run-{index:02d}-{run_spec['model']['model_id']}"
    run_dir.mkdir(exist_ok=True)
    scorecard_path = run_dir / "scorecard.json"
    artifacts = {
        "samples": str(run_dir / "per_sample.jsonl"),
        "predictions": str(run_dir / "predictions"),
    }
    write_scorecard(
        scorecard_path,
        run=run_spec["run"],
        benchmark=BENCHMARK,
        model=run_spec["model"],
        dataset=DATASET,
        generation={"num_requests": 1600, "successful": run_spec["metrics_summary"]["successful_samples"]},
        metrics_summary=run_spec["metrics_summary"],
        artifacts=artifacts,
        leaderboard_evidence=run_spec["leaderboard_evidence"],
        provenance=PROVENANCE,
        evaluation_kind="new_model_evaluation",
    )
    # summary.json + report.md derived from the scorecard.
    write_run_report_artifacts(output_dir=run_dir, scorecard_path=scorecard_path)
    return run_dir


def main() -> None:
    print(f"=== reporting demo → {DEMO_DIR} ===\n")
    DEMO_DIR.mkdir(exist_ok=True)
    run_dirs = [build_one(spec, i) for i, spec in enumerate(RUNS)]

    # 1) Single-run Markdown report (show one inline).
    report_md = (run_dirs[0] / "report.md").read_text()
    print("########## report.md  (run-00, cascade-xl) ##########")
    print(report_md)

    # 2) Run index over all three runs → json + jsonl + HTML browser.
    index = write_run_index(
        run_dirs,
        output_dir=DEMO_DIR,
    )
    print("########## index (markdown view) ##########")
    print(build_markdown_run_index(index))

    # 3) Side-by-side comparison with aurora-1b as baseline → deltas.
    summary_paths = [d / "summary.json" for d in run_dirs]
    comparison = write_run_comparison(
        summary_paths,
        labels=["Cascade-XL", "Aurora-7B", "Aurora-1B (baseline)"],
        baseline="Aurora-1B (baseline)",
        output_json=DEMO_DIR / "comparison.json",
        output_md=DEMO_DIR / "comparison.md",
    )
    print("########## comparison.md ##########")
    print((DEMO_DIR / "comparison.md").read_text())

    # 4) Also write a standalone browser from the in-memory index payload.
    write_run_browser(index, DEMO_DIR / "index.html")

    print("=== artifacts written ===")
    for p in sorted(DEMO_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(DEMO_DIR)}  ({p.stat().st_size} bytes)")
    print(f"\nOpen in a browser:  file://{DEMO_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
