#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import project_root

REPO_ROOT = project_root(__file__)
SRC_ROOT = REPO_ROOT
MODE_CONFIGS = {
    "universal": "inference_universal.yaml",
    "gta_drive": "inference_gta_drive.yaml",
    "templerun": "inference_templerun.yaml",
}


def _ensure_src_on_path() -> None:
    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)


def _write_summary(output_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def _run_integrated(case: dict[str, Any], output_root: Path, device: str) -> dict[str, Any]:
    _ensure_src_on_path()
    from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline

    video_path = output_root / "worldfoundry_matrix_game_2.mp4"
    pipeline = MatrixGame2Pipeline.from_pretrained(
        model_path=case["pretrained_model_path"],
        mode=case["mode"],
        device=device,
        checkpoint_path=case["checkpoint_path"],
    )
    result = pipeline(
        images=case["image_path"],
        interactions=None,
        num_frames=int(case["num_output_frames"]),
        fps=int(case["fps"]),
        seed=int(case["seed"]),
        output_path=video_path,
        visualize_ops=False,
        official_bench_actions=bool(case.get("official_bench_actions", False)),
        return_dict=True,
    )
    summary = {
        "variant": "integrated",
        "case_id": case.get("id"),
        "video": {
            "video_path": str(video_path),
            "num_output_frames": result.get("num_output_frames"),
            "fps": result.get("fps"),
        },
        "result": {key: value for key, value in result.items() if key != "video"},
    }
    return _write_summary(output_root, summary)


def _run_official(case: dict[str, Any], output_root: Path, device: str) -> dict[str, Any]:
    del device
    repo_root = Path(case["repo_root"])
    mode = str(case["mode"])
    config_name = MODE_CONFIGS[mode]
    checkpoint_path = Path(case["pretrained_model_path"]) / case["checkpoint_path"]
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(repo_root / "inference.py"),
        "--config_path",
        str(repo_root / "configs" / "inference_yaml" / config_name),
        "--checkpoint_path",
        str(checkpoint_path),
        "--img_path",
        str(case["image_path"]),
        "--output_folder",
        str(output_root),
        "--num_output_frames",
        str(int(case["num_output_frames"])),
        "--seed",
        str(int(case["seed"])),
        "--pretrained_model_path",
        str(case["pretrained_model_path"]),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, cwd=repo_root, env=env, check=True)
    video_path = output_root / "demo.mp4"
    summary = {
        "variant": "official",
        "case_id": case.get("id"),
        "command": command,
        "video": {
            "video_path": str(video_path),
            "fps": int(case["fps"]),
            "num_output_frames": int(case["num_output_frames"]),
        },
    }
    return _write_summary(output_root, summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Matrix-Game-2 official/integrated parity worker.")
    parser.add_argument("--model", choices=["mg2"], required=True)
    parser.add_argument("--variant", choices=["official", "integrated"], required=True)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    case = json.loads(args.case_json.read_text(encoding="utf-8"))
    if args.variant == "official":
        _run_official(case, args.output_root, args.device)
    else:
        _run_integrated(case, args.output_root, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
