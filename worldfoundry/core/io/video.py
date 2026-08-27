"""Video read/write primitives shared by inference and evaluation code."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

from worldfoundry.core.process import run_logged_subprocess

from .media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .scratch import make_scratch_dir
from .storage import local_path_for_uri, parse_uri_scheme, uri_to_local_path, write_binary_uri


def _resolve_ffmpeg_executable(explicit: str | Path | None = None) -> str | None:
    """Resolve an explicit, system, or ImageIO-bundled FFmpeg executable."""

    if explicit is not None:
        return str(explicit)
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError, RuntimeError):
        return None


def extract_frames_from_video_url(video_url: str):
    """Decode a remote video URL and return RGB PIL frames."""

    from PIL import Image

    frames = load_video_frames(video_url)
    return [Image.fromarray(frame[..., :3]) for frame in frames]


def resize_video_tensor_to_resolution(video_tensor, resolution: Iterable[int]):
    """Resize and center-crop a ``[T,C,H,W]`` tensor to ``(target_h, target_w)``."""

    import torchvision

    target_h, target_w = [int(value) for value in resolution]
    orig_h, orig_w = int(video_tensor.shape[2]), int(video_tensor.shape[3])
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    resized = torchvision.transforms.functional.resize(video_tensor, resizing_shape)
    return torchvision.transforms.functional.center_crop(resized, [target_h, target_w])


def read_image_as_video_tensor(
    image_path: str | Path,
    resolution: Iterable[int],
    num_video_frames: int,
    *,
    resize: bool = True,
):
    """Load an image and materialize a ``[1,C,T,H,W]`` uint8 conditioning video tensor."""

    import torch
    import torchvision
    from PIL import Image

    path = Path(image_path)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image extension: {path.suffix}")
    if num_video_frames < 1:
        raise ValueError("num_video_frames must be at least 1")

    image = Image.open(path).convert("RGB")
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    first_frame = image_tensor.unsqueeze(0)
    zero_tail = torch.zeros_like(first_frame).repeat(num_video_frames - 1, 1, 1, 1)
    video = torch.cat([first_frame, zero_tail], dim=0)
    video = (video * 255.0).to(torch.uint8)
    if resize:
        video = resize_video_tensor_to_resolution(video, resolution)
    return video.unsqueeze(0).permute(0, 2, 1, 3, 4)


def video_tensor_to_uint8_frames(video_tensor, *, value_range: str | tuple[float, float] = "auto") -> "object":
    """Convert a normalized torch video tensor to uint8 THWC frames.

    ``value_range="auto"`` treats tensors with negative values as ``[-1, 1]``
    and non-negative floating tensors as ``[0, 1]``.
    """

    import torch

    if not torch.is_tensor(video_tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(video_tensor)}")
    tensor = video_tensor.detach().cpu().float()
    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for 5D video tensor, got shape {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(f"Expected video tensor shape [C,T,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[-1] in {1, 3, 4}:
        video = tensor
    elif tensor.shape[0] in {1, 3, 4}:
        video = tensor.permute(1, 2, 3, 0)
    elif tensor.shape[1] in {1, 3, 4}:
        video = tensor.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Unable to infer channel layout for video tensor shape {tuple(tensor.shape)}")
    if value_range == "auto":
        low, high = (-1.0, 1.0) if float(video.min()) < 0.0 else (0.0, 1.0)
    elif value_range == "-1,1":
        low, high = -1.0, 1.0
    elif value_range == "0,1":
        low, high = 0.0, 1.0
    else:
        low, high = value_range
    if high <= low:
        raise ValueError("value_range high bound must be larger than low bound.")
    video = ((video.clamp(float(low), float(high)) - float(low)) * (255.0 / (float(high) - float(low)))).to(torch.uint8)
    return video.numpy()


def coerce_video_frames(video_input):
    """Normalize common video inputs into a uint8 THWC numpy array."""

    import os

    import numpy as np
    import torch
    from PIL import Image

    if isinstance(video_input, (str, os.PathLike)):
        return load_video_frames(str(video_input))

    if torch.is_tensor(video_input):
        return video_tensor_to_uint8_frames(video_input)

    if isinstance(video_input, np.ndarray):
        video = video_input
    elif isinstance(video_input, (list, tuple)):
        frames = []
        for frame in video_input:
            if torch.is_tensor(frame):
                tensor = frame.detach().cpu()
                if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
                    tensor = tensor.permute(1, 2, 0)
                frame = tensor.numpy()
            elif isinstance(frame, Image.Image):
                frame = np.asarray(frame.convert("RGB"))
            else:
                frame = np.asarray(frame)
            frames.append(frame)
        video = np.stack(frames, axis=0)
    else:
        raise TypeError(f"Unsupported video input type: {type(video_input)}")

    if video.ndim != 4:
        raise ValueError(f"Expected video with 4 dimensions, got shape {tuple(video.shape)}")

    if video.shape[-1] in {1, 3, 4}:
        converted = video
    elif video.shape[1] in {1, 3, 4}:
        converted = np.transpose(video, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unable to infer channel layout for video shape {tuple(video.shape)}")

    if converted.dtype != np.uint8:
        if np.issubdtype(converted.dtype, np.floating):
            if converted.max() <= 1.0:
                converted = np.clip(converted * 255.0, 0, 255)
            else:
                converted = np.clip(converted, 0, 255)
        else:
            converted = np.clip(converted, 0, 255)
        converted = converted.astype(np.uint8)
    return converted


def load_video_frames(video_path: str | Path):
    """Decode a local or remote video into a uint8 THWC numpy array."""

    import imageio
    import numpy as np

    with local_path_for_uri(video_path) as local_path:
        frames = imageio.mimread(str(local_path), memtest=False)
    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")
    arrays = []
    for frame in frames:
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        arrays.append(array)
    return np.stack(arrays, axis=0)


def get_video_details(video_path: str | Path) -> tuple[int, float, float]:
    """Return ``(total_frames, fps, duration_seconds)`` for a local video."""

    from decord import VideoReader, cpu

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video path not found: {path}")
    if path.stat().st_size < 1024:
        raise ValueError(f"Video too short: {path}")
    reader = VideoReader(str(path), num_threads=-1, ctx=cpu(0))
    total_frames = len(reader)
    original_fps = float(reader.get_avg_fps())
    return total_frames, original_fps, total_frames / original_fps


def sample_video_frames(
    video_path: str | Path,
    max_frames: int,
    fps: float = 1.0,
    force_sample: bool = False,
    *,
    num_threads: int = 2,
) -> tuple["object", str, float]:
    """Decode uniformly bounded RGB frames and their correct source timestamps.

    ``fps`` is the requested temporal sampling rate when ``force_sample`` is
    false. Forced sampling returns exactly ``max_frames`` entries (including
    repeated indices for very short clips), matching common video-LLM input
    contracts. The returned tuple is ``(THWC uint8 frames, timestamp text,
    duration_seconds)``.
    """

    import numpy as np
    from decord import VideoReader, cpu

    max_frames = int(max_frames)
    requested_fps = float(fps)
    num_threads = int(num_threads)
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    if not math.isfinite(requested_fps) or requested_fps <= 0:
        raise ValueError("fps must be a positive finite number")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=num_threads)
    total_frames = len(reader)
    source_fps = float(reader.get_avg_fps())
    if total_frames < 1:
        raise ValueError(f"No frames found in video: {video_path}")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid source frame rate {source_fps!r} for video: {video_path}")

    if force_sample:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=np.int64)
    else:
        frame_step = max(int(round(source_fps / requested_fps)), 1)
        indices = np.arange(0, total_frames, frame_step, dtype=np.int64)
        if len(indices) > max_frames:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=np.int64)

    timestamps = [float(index) / source_fps for index in indices]
    timestamp_text = ",".join(f"{timestamp:.2f}s" for timestamp in timestamps)
    frames = reader.get_batch(indices.tolist()).asnumpy()
    return frames, timestamp_text, total_frames / source_fps


def list_numbered_frame_paths(
    frame_dir: str | Path,
    *,
    prefix: str = "frames_",
    suffix: str = ".png",
) -> tuple[Path, ...]:
    """List numerically named frame files without counting unrelated directory entries."""

    root = Path(frame_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"frame directory not found: {root}")
    if not suffix.startswith("."):
        raise ValueError("suffix must be a non-empty extension beginning with '.'")
    normalized_suffix = suffix.lower()
    indexed: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_file() or not path.name.startswith(prefix) or path.suffix.lower() != normalized_suffix:
            continue
        index_text = path.name[len(prefix) : -len(path.suffix)]
        try:
            frame_index = int(index_text)
        except ValueError:
            continue
        indexed.append((frame_index, path))
    return tuple(path for _, path in sorted(indexed, key=lambda item: (item[0], item[1].name)))


def extract_video_frames_to_directory(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    prefix: str = "frames_",
    suffix: str = ".png",
    ffmpeg_path: str | Path | None = None,
    threads: int = 1,
    timeout_seconds: float | None = None,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Extract every source frame with bounded, deterministic ffmpeg settings."""

    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"video path not found: {source}")
    if not prefix:
        raise ValueError("prefix must not be empty")
    if not suffix.startswith("."):
        raise ValueError("suffix must be a non-empty extension beginning with '.'")
    if int(threads) < 1:
        raise ValueError("threads must be positive")
    if timeout_seconds is not None and float(timeout_seconds) <= 0:
        raise ValueError("timeout_seconds must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    existing = list_numbered_frame_paths(root, prefix=prefix, suffix=suffix)
    if existing and not overwrite:
        return existing
    for path in existing:
        path.unlink()
    executable = _resolve_ffmpeg_executable(ffmpeg_path)
    if not executable:
        raise FileNotFoundError("ffmpeg was not found in PATH and no ImageIO-bundled executable is available")

    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        str(int(threads)),
        "-i",
        str(source),
        "-vsync",
        "0",
        "-start_number",
        "1",
        "-y",
        str(root / f"{prefix}%d{suffix}"),
    ]
    stdout_path = root / "ffmpeg_extract.stdout.log"
    stderr_path = root / "ffmpeg_extract.stderr.log"
    try:
        completed = run_logged_subprocess(
            command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=None if timeout_seconds is None else float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        for path in list_numbered_frame_paths(root, prefix=prefix, suffix=suffix):
            path.unlink()
        raise TimeoutError(f"ffmpeg frame extraction timed out after {timeout_seconds} seconds") from exc
    if completed.returncode != 0:
        for path in list_numbered_frame_paths(root, prefix=prefix, suffix=suffix):
            path.unlink()
        err = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
        detail = "\n".join(err.splitlines()[-20:])
        raise RuntimeError(f"ffmpeg frame extraction failed with code {completed.returncode}: {detail}")
    extracted = list_numbered_frame_paths(root, prefix=prefix, suffix=suffix)
    if not extracted:
        raise ValueError(f"ffmpeg produced no frames for video: {source}")
    return extracted


def _ffprobe_number(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ffprobe_rate(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    text = str(value)
    if "/" not in text:
        return _ffprobe_number(text)
    numerator, denominator = text.split("/", 1)
    numerator_value = _ffprobe_number(numerator)
    denominator_value = _ffprobe_number(denominator)
    if numerator_value is None or denominator_value in (None, 0.0):
        return None
    return numerator_value / denominator_value


def probe_video_metadata(
    video_path: str | Path,
    *,
    ffprobe_path: str | Path | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, int | float | None]:
    """Read lightweight video metadata with one bounded ``ffprobe`` process."""

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video path not found: {path}")
    executable = str(ffprobe_path) if ffprobe_path is not None else shutil.which("ffprobe")
    if not executable:
        raise FileNotFoundError("ffprobe was not found in PATH")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    # Lazy import: keeps this core module importable without pulling the
    # runtime layer in at module import time (same pattern as the
    # runtime.compile_cache imports elsewhere in worldfoundry.core).
    from worldfoundry.runtime.jobs import run_bounded_command

    completed = run_bounded_command(command, timeout=float(timeout_seconds))
    if completed["timed_out"]:
        raise TimeoutError(f"ffprobe timed out after {timeout_seconds:g}s for {path}")
    if completed["returncode"] != 0:
        reason = completed["stderr"].strip() or f"ffprobe exited with status {completed['returncode']}"
        raise ValueError(reason)
    try:
        payload = json.loads(completed["stdout"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned invalid JSON for {path}") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise ValueError(f"ffprobe found no video stream in {path}")
    stream = streams[0]
    format_payload = payload.get("format")
    format_metadata = format_payload if isinstance(format_payload, dict) else {}
    fps = _ffprobe_rate(stream.get("avg_frame_rate")) or _ffprobe_rate(stream.get("r_frame_rate"))
    duration = _ffprobe_number(stream.get("duration"))
    if duration is None:
        duration = _ffprobe_number(format_metadata.get("duration"))
    frame_count_value = _ffprobe_number(stream.get("nb_frames"))
    frame_count = None if frame_count_value is None else int(frame_count_value)
    width = _ffprobe_number(stream.get("width"))
    height = _ffprobe_number(stream.get("height"))
    return {
        "width": None if width is None else int(width),
        "height": None if height is None else int(height),
        "fps": fps,
        "duration_seconds": duration,
        "frame_count": frame_count,
    }


def load_frames_from_video(
    video_path: str | Path,
    indices: Iterable[int],
    video_decode_backend: str = "decord",
    eval_: bool = True,
):
    """Load selected RGB frames into a torch tensor using decord or OpenCV."""

    import os

    import cv2
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    path = str(video_path)
    frame_indices = [int(index) for index in indices]
    ext = os.path.splitext(path)[1].lower()
    if ext in {".gif", ".webm"} or video_decode_backend == "opencv":
        capture = cv2.VideoCapture(path)
        frames: dict[int, np.ndarray] = {}
        max_index = max(frame_indices)
        frame_id = 0
        ok = True
        while ok and frame_id <= max_index:
            ok, frame = capture.read()
            if ok and frame_id in frame_indices:
                frames[frame_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_id += 1
        capture.release()
        return torch.tensor(np.stack([frames[index] for index in frame_indices if index in frames]))

    reader = VideoReader(path) if eval_ else VideoReader(path, num_threads=1, ctx=cpu(0))
    batch = reader.get_batch(frame_indices)
    if isinstance(batch, torch.Tensor):
        return batch
    return torch.tensor(batch.asnumpy())


def read_video(video_path: str | Path, *, return_metadata: bool = True):
    """Decode a video into frames and optional metadata."""

    import imageio
    import numpy as np

    with local_path_for_uri(video_path) as local_path:
        reader = imageio.get_reader(str(local_path))
        frames = [np.asarray(frame) for frame in reader]
        metadata = reader.get_meta_data()
    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")
    stacked = np.stack(frames, axis=0)
    return (stacked, metadata) if return_metadata else stacked


def save_video_frames(video_frames, output_path: str | Path, fps: int = 16, **kwargs) -> None:
    """Write a THWC uint8 frame array/list to a video path or URI."""

    write_video(video_frames, output_path, fps=fps, **kwargs)


def save_video_h264(
    video_frames,
    output_path: str | Path,
    *,
    fps: float = 16.0,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    """Write THWC RGB frames as a local H.264/yuv420p MP4 with FFmpeg."""

    frames = coerce_video_frames(video_frames)
    if int(frames.shape[0]) == 0:
        return
    if parse_uri_scheme(output_path) != "file":
        raise ValueError("save_video_h264 currently supports local output paths only")

    target = uri_to_local_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = _resolve_ffmpeg_executable()
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH and no ImageIO-bundled executable is available; "
            "cannot encode H.264 output video"
        )

    frame_count, height, width, _ = frames.shape
    command = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for index in range(frame_count):
            process.stdin.write(frames[index].tobytes())
        process.stdin.close()
        assert process.stderr is not None
        stderr = process.stderr.read()
        process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg H.264 encode failed for {target}: {message}")


def save_image_or_video_tensor(
    tensor,
    save_path,
    *,
    fps: int = 24,
    quality: int | None = None,
    ffmpeg_params: list[str] | None = None,
    value_range: str | tuple[float, float] = "auto",
    image_format: str = "JPEG",
    video_format: str = "mp4",
    **kwargs,
) -> str | None:
    """Save a normalized ``[C,T,H,W]`` or ``[B,C,T,H,W]`` tensor as image/video.

    A single-frame tensor is saved as an image; multi-frame tensors are saved as
    videos. Local paths and URI-like targets supported by ``worldfoundry.core.io``
    storage helpers are both accepted.
    """

    from io import BytesIO

    from PIL import Image

    frames = video_tensor_to_uint8_frames(tensor, value_range=value_range)
    target = save_path
    is_file_obj = hasattr(target, "write")

    if frames.shape[0] == 1:
        image = Image.fromarray(frames[0][..., :3]).convert("RGB")
        if is_file_obj:
            image.save(target, format=image_format, quality=quality or 85, **kwargs)
            return None

        path_text = str(target)
        if not Path(path_text).suffix:
            path_text = f"{path_text}.jpg"
        if parse_uri_scheme(path_text) == "file":
            output_path = uri_to_local_path(path_text)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format=image_format, quality=quality or 85, **kwargs)
        else:
            buffer = BytesIO()
            image.save(buffer, format=image_format, quality=quality or 85, **kwargs)
            write_binary_uri(path_text, buffer)
        return path_text

    if is_file_obj:
        suffix = f".{video_format.lstrip('.')}"
        video_kwargs = dict(kwargs)
        if ffmpeg_params is not None:
            video_kwargs["ffmpeg_params"] = ffmpeg_params
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
            write_video(
                frames,
                handle.name,
                fps=fps,
                quality=quality,
                format=video_format,
                **video_kwargs,
            )
            handle.seek(0)
            target.write(handle.read())
        return None

    path_text = str(target)
    if not Path(path_text).suffix:
        path_text = f"{path_text}.mp4"
    video_kwargs = dict(kwargs)
    if ffmpeg_params is not None:
        video_kwargs["ffmpeg_params"] = ffmpeg_params
    write_video(
        frames,
        path_text,
        fps=fps,
        quality=quality,
        format=video_format,
        **video_kwargs,
    )
    return path_text


def write_video_torchvision(
    filename: str | Path,
    video_array: Any,
    fps: float,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Write RGB video frames with a ``torchvision.io.write_video``-compatible signature."""

    try:
        from torchvision.io import write_video as torchvision_write_video
    except (ImportError, AttributeError, RuntimeError):
        torchvision_write_video = None

    if torchvision_write_video is not None:
        torchvision_write_video(str(filename), video_array, fps, *args, **kwargs)
        return

    import cv2
    import numpy as np

    frames = video_array.detach().cpu().numpy() if hasattr(video_array, "detach") else np.asarray(video_array)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("write_video_torchvision expects frames shaped [T, H, W, C]")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise OSError(f"failed to open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_video(
    video_frames,
    output_path: str | Path,
    *,
    fps: int = 16,
    quality: int | None = None,
    format: str | None = None,
    **kwargs,
) -> None:
    """Write a THWC video array/list to a local path or remote URI."""

    import imageio

    frames = coerce_video_frames(video_frames)
    write_kwargs = {"fps": fps, "macro_block_size": 1, **kwargs}
    if quality is not None:
        write_kwargs["quality"] = quality

    if parse_uri_scheme(output_path) == "file":
        target = uri_to_local_path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(target), frames, format=format, **write_kwargs)
        return

    suffix = Path(str(output_path)).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
        imageio.mimsave(handle.name, frames, format=format, **write_kwargs)
        handle.seek(0)
        write_binary_uri(output_path, handle.read())


def materialize_video_input(
    video_input,
    output_dir: Optional[str] = None,
    filename: str = "input.mp4",
    fps: int = 24,
) -> str:
    """Return a local video path, materializing in-memory inputs when needed.

    When ``output_dir`` is omitted, the video is written to a scratch
    directory under ``${WORLDFOUNDRY_CACHE_DIR}/scratch/`` that is removed
    (best-effort) at interpreter exit. Callers that need the materialized
    file to outlive the process must pass an explicit ``output_dir``.
    """

    import os

    if isinstance(video_input, (str, os.PathLike)):
        candidate = uri_to_local_path(video_input) if parse_uri_scheme(video_input) == "file" else None
        if (
            candidate is not None
            and candidate.exists()
            and candidate.is_file()
            and candidate.suffix.lower() in VIDEO_EXTENSIONS
        ):
            return str(candidate.resolve())

    if output_dir is None:
        output_dir = make_scratch_dir(prefix="worldfoundry_video_")
    output_path = Path(output_dir).expanduser().resolve() / filename
    frames = coerce_video_frames(video_input)
    save_video_frames(frames, str(output_path), fps=fps)
    return str(output_path)


def save_videos_grid(videos, path: str, rescale=False, n_rows=6, fps=8):
    """Save a batch of BCTHW video tensors as a single grid video."""

    import os

    import imageio
    import numpy as np
    import torchvision
    from einops import rearrange

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


__all__ = [
    "VIDEO_EXTENSIONS",
    "coerce_video_frames",
    "extract_video_frames_to_directory",
    "extract_frames_from_video_url",
    "get_video_details",
    "list_numbered_frame_paths",
    "load_frames_from_video",
    "load_video_frames",
    "materialize_video_input",
    "probe_video_metadata",
    "read_image_as_video_tensor",
    "read_video",
    "resize_video_tensor_to_resolution",
    "sample_video_frames",
    "save_image_or_video_tensor",
    "save_video_frames",
    "save_videos_grid",
    "video_tensor_to_uint8_frames",
    "write_video",
    "write_video_torchvision",
]
