from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT
WORKER_SCRIPT = REPO_ROOT / "scripts" / "model_zoo" / "matrix_game_parity_worker.py"
DEFAULT_REPO_ROOT = REPO_ROOT / "tmp" / "model_zoo" / "repos" / "github.com_SkyworkAI_Matrix-Game" / "Matrix-Game-2"
DEFAULT_CKPT_ROOT = (
    REPO_ROOT
    / "cache"
    / "hfd"
    / "models--Skywork--Matrix-Game-2.0"
    / "snapshots"
    / "f1729d99a80e0f07993a77d7dad4a3190e23c2c8"
)
DEFAULT_REF_IMAGE = REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "matrix-game-2" / "universal" / "0000.png"
MODE_ACTIONS = {
    "universal": ["forward", "camera_r", "forward_right", "camera_l"],
    "gta_drive": ["forward", "camera_l", "back", "camera_r"],
    "templerun": ["jump", "slide", "leftside", "turnright", "nomove"],
}
MODE_CHECKPOINTS = {
    "universal": "base_distilled_model/base_distill.safetensors",
    "gta_drive": "gta_distilled_model/gta_keyboard2dim.safetensors",
    "templerun": "templerun_distilled_model/templerun_7dim_onlykey.safetensors",
}
VARIANT_LABELS = {
    "official": "official",
    "integrated": "worldfoundry",
}


def default_output_dir() -> Path:
    return Path(
        os.environ.get(
            "WORLDFOUNDRY_MODEL_PARITY_OUTPUT_DIR",
            str(REPO_ROOT / "tmp" / "model_zoo" / "parity" / "matrix-game-2"),
        )
    )


def artifact_name(variant: str, mode: str, seed: int, frames: int) -> str:
    label = VARIANT_LABELS[variant]
    return f"matrix_game2_{label}_{mode}_seed{seed}_{frames}f.mp4"


def metadata_name(variant: str, mode: str, seed: int, frames: int) -> str:
    return artifact_name(variant, mode, seed, frames).removesuffix(".mp4") + ".json"


def build_parser(variant: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run Matrix-Game-2 {variant} demo for model-zoo parity.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("MATRIX_GAME2_REPO_ROOT", DEFAULT_REPO_ROOT)),
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path(os.environ.get("MATRIX_GAME2_CKPT_DIR", DEFAULT_CKPT_ROOT)),
    )
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--save-file", type=Path, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--mode", choices=sorted(MODE_ACTIONS), default="universal")
    parser.add_argument("--num-output-frames", type=int, default=15)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=os.environ.get("MATRIX_GAME2_DEVICE", "cuda:0"))
    return parser


def ensure_src_on_path() -> None:
    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)


def default_image_path(mode: str, repo_root: Path) -> Path:
    if mode == "gta_drive":
        return repo_root / "demo_images" / "gta_drive" / "0001.png"
    if mode == "templerun":
        return repo_root / "demo_images" / "temple_run" / "0000.png"
    return DEFAULT_REF_IMAGE


def build_case(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    ensure_src_on_path()
    operator_module = importlib.import_module("worldfoundry.operators.matrix_game_2_operator")
    matrix_game_2_operator = operator_module.MatrixGame2Operator

    image_path = args.image_path or default_image_path(args.mode, args.repo_root)
    actions = MODE_ACTIONS[args.mode]
    operator = matrix_game_2_operator(mode=args.mode)
    condition_frames = (int(args.num_output_frames) - 1) * 4 + 1
    conditions = operator.process_official_bench_actions(num_frames=condition_frames)
    return {
        "id": f"matrix_game2_{VARIANT_LABELS[variant]}_{args.mode}_seed{args.seed}_{args.num_output_frames}f",
        "model": "mg2",
        "mode": args.mode,
        "image_path": str(image_path.expanduser().resolve()),
        "actions": actions,
        "official_bench_actions": True,
        "num_output_frames": int(args.num_output_frames),
        "fps": int(args.fps),
        "seed": int(args.seed),
        "repo_root": str(args.repo_root.expanduser().resolve()),
        "pretrained_model_path": str(args.ckpt_dir.expanduser().resolve()),
        "checkpoint_path": MODE_CHECKPOINTS[args.mode],
        "keyboard_condition": conditions["keyboard_condition"].tolist(),
        "mouse_condition": conditions.get("mouse_condition").tolist() if "mouse_condition" in conditions else None,
    }


def check_prerequisites(args: argparse.Namespace, variant: str) -> int:
    image_path = args.image_path or default_image_path(args.mode, args.repo_root)
    if variant == "official" and not (args.repo_root / "inference.py").is_file():
        print(f"missing official Matrix-Game-2 repo: {args.repo_root}", file=sys.stderr)
        return 3
    if not args.ckpt_dir.is_dir():
        print(f"missing Matrix-Game-2 checkpoint directory: {args.ckpt_dir}", file=sys.stderr)
        print(
            "run: python scripts/model_zoo/download_checkpoints.py --model-id matrix-game-2 "
            "--repo-id Skywork/Matrix-Game-2.0 --execute",
            file=sys.stderr,
        )
        return 4
    checkpoint_file = args.ckpt_dir / MODE_CHECKPOINTS[args.mode]
    if not checkpoint_file.is_file():
        print(f"missing Matrix-Game-2 mode checkpoint: {checkpoint_file}", file=sys.stderr)
        return 4
    if not image_path.is_file():
        print(f"missing Matrix-Game-2 input image: {image_path}", file=sys.stderr)
        return 5
    return 0


def run_worker(case_json: Path, output_root: Path, variant: str, device: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--model",
        "mg2",
        "--variant",
        "official" if variant == "official" else "integrated",
        "--case-json",
        str(case_json),
        "--output-root",
        str(output_root),
        "--device",
        device,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return json.loads((output_root / "summary.json").read_text(encoding="utf-8"))


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    variant: str,
    case: dict[str, Any],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    metadata = {
        "schema_version": "worldfoundry-matrix-game2-demo",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "variant": variant,
        "case": case,
        "worker_summary": summary,
        "output": str(output_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(argv: list[str] | None, *, variant: str) -> int:
    args = build_parser(variant).parse_args(argv)
    prerequisite_status = check_prerequisites(args, variant)
    if prerequisite_status != 0:
        return prerequisite_status

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case = build_case(args, variant)
    case_json = args.output_dir / f"{case['id']}_case.json"
    case_json.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    worker_root = args.output_dir / f"{case['id']}_worker"
    summary = run_worker(case_json, worker_root, variant, args.device)

    output_path = args.save_file or (
        args.output_dir / artifact_name(variant, args.mode, int(args.seed), int(args.num_output_frames))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary["video"]["video_path"], output_path)
    metadata_path = args.metadata_json or args.output_dir / metadata_name(
        variant,
        args.mode,
        int(args.seed),
        int(args.num_output_frames),
    )
    write_metadata(metadata_path, args=args, variant=variant, case=case, summary=summary, output_path=output_path)
    return 0
