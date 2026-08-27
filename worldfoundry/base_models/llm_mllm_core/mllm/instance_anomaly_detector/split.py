import json
import os
from pathlib import Path

from worldfoundry.core.process import run_logged_subprocess
from worldfoundry.runtime.jobs import run_bounded_command


def get_video_info(filepath):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,duration",
        "-of",
        "json",
        filepath,
    ]
    # Bounded metadata probe: a hung ffprobe on a corrupt input must not stall
    # the split pipeline, so cap the read and fail loudly instead of handing an
    # empty stdout to the JSON parser.
    result = run_bounded_command(cmd, timeout=10)
    if result["timed_out"] or result["returncode"] != 0:
        raise RuntimeError(
            f"ffprobe failed for {filepath} with code {result['returncode']}: "
            f"{str(result['stderr']).strip()}"
        )
    info = json.loads(result["stdout"])
    duration = float(info["streams"][0]["duration"])
    fr_str = info["streams"][0]["avg_frame_rate"]
    if "/" in fr_str:
        num, den = map(int, fr_str.split("/"))
        fps = num / den if den != 0 else 25
    else:
        fps = float(fr_str)
    return duration, fps


def split_video(filepath, output_folder):
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    duration, fps = get_video_info(filepath)
    total_segments = int(duration) - 1
    log_dir = Path(output_folder) / "_logs" / name

    for i in range(total_segments):
        out_path = os.path.join(output_folder, f"{name}_clip_{i:04d}.mp4")
        if os.path.exists(out_path):
            continue
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            filepath,
            "-ss",
            str(i),
            "-t",
            "2",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            out_path,
        ]
        run_logged_subprocess(
            cmd,
            stdout_path=log_dir / f"clip_{i:04d}.stdout.log",
            stderr_path=log_dir / f"clip_{i:04d}.stderr.log",
        )


def split(input_folder, model_name):
    output_folder = f"./model_clip/{model_name}"
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            split_video(os.path.join(input_folder, file), output_folder)
