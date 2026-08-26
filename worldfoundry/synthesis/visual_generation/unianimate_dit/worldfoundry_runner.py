from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

from worldfoundry.core.io.paths import package_module_root as package_root
from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.core.process import run_logged_subprocess, synthesis_timeout_seconds


def _demo_asset_root() -> Path:
    override = os.getenv("WORLDFOUNDRY_UNIANIMATE_DIT_DEMO_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return resolve_data_path("test_cases", "unianimate_dit").resolve()


DEFAULT_CASE = '[1, "data/images/WOMEN-Blouses_Shirts-id_00004955-01_4_full.jpg", "data/saved_pose/WOMEN-Blouses_Shirts-id_00004955-01_4_full"]'
DEFAULT_REF_IMAGE = "data/images/WOMEN-Blouses_Shirts-id_00004955-01_4_full.jpg"
DEFAULT_SOURCE_VIDEO = "data/videos/source_video.mp4"
DEFAULT_POSE_DIR = "data/saved_pose/WOMEN-Blouses_Shirts-id_00004955-01_4_full"


def _ensure_link(link_path: Path, target_path: Path) -> None:
    target_path = target_path.expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"required UniAnimate asset path does not exist: {target_path}")
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == target_path:
            return
        raise FileExistsError(f"refusing to replace existing path: {link_path}")
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


def _as_bool(value: str | bool | int | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _rewrite_demo_script(
    source: str,
    *,
    max_frames: int,
    steps: int,
    height: int,
    width: int,
    seed: int,
    use_usp: bool,
) -> str:
    text = source
    text = re.sub(r"^height\s*=\s*\d+", f"height = {height}", text, flags=re.MULTILINE)
    text = re.sub(r"^width\s*=\s*\d+", f"width = {width}", text, flags=re.MULTILINE)
    text = re.sub(r"^seed\s*=\s*\d+", f"seed = {seed}", text, flags=re.MULTILINE)
    text = re.sub(r"^max_frames\s*=\s*\d+", f"max_frames = {max_frames}", text, flags=re.MULTILINE)
    text = re.sub(
        r"test_list_path\s*=\s*\[.*?\]\s*\n\nmisc_size",
        f"test_list_path = [\n    {DEFAULT_CASE},\n]\n\nmisc_size",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"num_inference_steps\s*=\s*50", f"num_inference_steps={steps}", text)
    text = re.sub(
        r"start_frame\s*=\s*random\.randint\(0,\s*_total_frame_num-cover_frame_num-1\)",
        "start_frame = 0 # random.randint(0, _total_frame_num-cover_frame_num-1)",
        text,
    )
    del use_usp
    return text


def _newest_mp4(root: Path, *, since: float) -> Path | None:
    candidates = [
        path
        for path in root.rglob("*.mp4")
        if path.is_file() and path.stat().st_mtime >= since
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_generated_panel(source_path: Path, output_path: Path) -> bool:
    """Crop UniAnimate's official three-panel comparison video to the generated panel."""
    try:
        import cv2
    except Exception:
        return False

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    if width < 3 or height <= 0:
        cap.release()
        return False
    panel_width = width // 3
    if panel_width <= 0:
        cap.release()
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_width, height),
    )
    if not writer.isOpened():
        cap.release()
        return False
    wrote = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame[:, width - panel_width : width])
            wrote = True
    finally:
        cap.release()
        writer.release()
    if not wrote:
        tmp_path.unlink(missing_ok=True)
        return False
    tmp_path.replace(output_path)
    return True


def _ensure_demo_assets(repo_root: Path) -> None:
    _ensure_link(repo_root / "data", _demo_asset_root())


def _ensure_demo_pose(workspace_root: Path, source_repo_root: Path) -> None:
    pose_dir = workspace_root / DEFAULT_POSE_DIR
    if (pose_dir / "pose.jpg").is_file() and any(path.suffix.lower() in {".jpg", ".png"} for path in pose_dir.iterdir()):
        return
    align_script = source_repo_root / "run_align_pose.py"
    if not align_script.is_file():
        raise FileNotFoundError(f"UniAnimate pose alignment script not found: {align_script}")
    pose_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = workspace_root / "unianimate_pose_align.stdout.log"
    stderr_path = workspace_root / "unianimate_pose_align.stderr.log"
    completed = run_logged_subprocess(
        [
            sys.executable,
            str(align_script),
            "--ref_name",
            DEFAULT_REF_IMAGE,
            "--source_video_paths",
            DEFAULT_SOURCE_VIDEO,
            "--saved_pose_dir",
            DEFAULT_POSE_DIR,
        ],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        cwd=str(workspace_root),
        timeout=synthesis_timeout_seconds(),
    )
    if completed.returncode != 0:
        try:
            tail_out = stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            tail_err = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            tail_out, tail_err = f"(unable to read {stdout_path})", f"(unable to read {stderr_path})"
        raise RuntimeError(
            "UniAnimate pose alignment failed before inference:\n" + tail_out + "\n" + tail_err
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldFoundry UniAnimate-DiT official-demo runner")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--wan-root", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-usp", default="false")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    source_script = repo_root / "examples" / "unianimate_wan" / "inference_unianimate_wan_480p.py"
    if not source_script.is_file():
        raise FileNotFoundError(f"UniAnimate official script not found: {source_script}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_root = output_path.parent / "unianimate_dit_workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    _ensure_link(workspace_root / "Wan2.1-I2V-14B-720P", Path(args.wan_root))
    _ensure_link(workspace_root / "checkpoints", Path(args.checkpoint_dir))
    _ensure_demo_assets(workspace_root)
    _ensure_demo_pose(workspace_root, repo_root)

    run_script = output_path.parent / "unianimate_dit_official_demo.py"
    run_script.write_text(
        _rewrite_demo_script(
            source_script.read_text(encoding="utf-8"),
            max_frames=args.max_frames,
            steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            seed=args.seed,
            use_usp=_as_bool(args.use_usp),
        ),
        encoding="utf-8",
    )
    before = time.time()
    env = os.environ.copy()
    worldfoundry_parent = package_root("worldfoundry").parent
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(worldfoundry_parent), str(repo_root), str(workspace_root), env.get("PYTHONPATH", "")) if item
    )
    env["WORLDFOUNDRY_UNIANIMATE_USE_USP"] = "1" if _as_bool(args.use_usp) else "0"
    if _as_bool(args.use_usp):
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={max(int(args.nproc_per_node), 1)}",
            str(run_script),
        ]
    else:
        command = [sys.executable, str(run_script)]
    log_path = output_path.with_suffix(output_path.suffix + ".runner.log")
    stderr_path = output_path.with_suffix(output_path.suffix + ".runner.stderr.log")
    completed = run_logged_subprocess(
        command,
        stdout_path=log_path,
        stderr_path=stderr_path,
        cwd=str(workspace_root),
        env=env,
        timeout=synthesis_timeout_seconds(),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"UniAnimate official demo failed with code {completed.returncode}; see {log_path}")

    produced = _newest_mp4(workspace_root / "outputs", since=before)
    if produced is None:
        produced = _newest_mp4(workspace_root, since=before)
    if produced is None:
        raise RuntimeError("UniAnimate official demo finished but no mp4 artifact was produced.")
    composite_path = output_path.with_name(f"{output_path.stem}_official_composite{output_path.suffix}")
    shutil.copy2(produced, composite_path)
    if not _extract_generated_panel(produced, output_path):
        shutil.copy2(produced, output_path)


if __name__ == "__main__":
    main()
