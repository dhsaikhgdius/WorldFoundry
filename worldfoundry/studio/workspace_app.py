from __future__ import annotations

import argparse
import contextlib
import json
import os
import select
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from worldfoundry.core.io.paths import project_root
from worldfoundry.core.inference import (
    ASSET_GATED_WORLD_RUNTIME_MODEL_IDS,
    LINGBOT_VARIANT_BASE_ACT_PREVIEW,
    LINGBOT_VARIANT_BASE_CAM,
    LINGBOT_VARIANT_FAST,
    LINGBOT_WORLD_MODEL_ID,
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    generic_model_inference_spec,
    get_model_inference_spec,
    model_inference_spec,
)
from worldfoundry.evaluation.tasks.execution.runners.workspace_registry import (
    run_workspace_benchmark,
    validate_workspace_registry,
    workspace_benchmark_has_input,
    workspace_benchmark_runtime_hint,
    workspace_benchmark_runtime_hints,
    workspace_benchmark_supported,
)
from .catalog import CatalogEntry, find_entry, lingbot_world_fast_load_kwargs
from .conda_dispatch import (
    DISPATCH_ONLY_CALL_KWARGS,
    DISPATCH_ONLY_LOAD_KWARGS,
    dispatch_spec_for_inference,
    run_manager_payload_in_conda,
)
from .execution import RunRecord, StudioManager, _is_gaussian_splat_ply
from .jobs import StudioJob, StudioJobStore, format_elapsed
from .studio_catalog import _studio_catalog, _template_id_hint
from .visualization.backends.frontends import STUDIO_VISUALIZATIONS
from .visualization.backends.viser import npz_has_supported_geometry, viser_orientation_defaults
from .visualization.providers.run_record import first_geometry_point_candidate, first_splat_asset


REPO_ROOT = project_root(__file__)
MANAGER = StudioManager()


def _initial_studio_job_counter(workspace_root: str) -> int:
    max_counter = 0
    runtime_jobs_root = Path(workspace_root) / "runtime_jobs"
    for path in runtime_jobs_root.glob("studio-*"):
        suffix = path.name.removeprefix("studio-")
        if suffix.isdigit():
            max_counter = max(max_counter, int(suffix))
    return max_counter


JOBS = StudioJobStore(
    max_workers=int(os.getenv("WORLDFOUNDRY_WORKSPACE_MAX_JOBS", "8") or "8"),
    initial_counter=_initial_studio_job_counter(MANAGER.workspace_root),
)
OPENENVISION_LOGO_PATH = Path(__file__).with_name("assets") / "openenvision-logo.png"
EVALUATION_VALIDATION_RESULTS_PATH = (
    REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "evaluation" / "existing_results_fixture" / "results.jsonl"
)
SUPPORTED_WORKSPACE_JOB_TYPES = {"inference", "evaluation"}
SETTING_CHOICES = {
    "backend": {"auto", "from_pretrained", "api_init"},
    "attention_backend": {"auto", "torch", "flash_attn_2", "flash_attn_3", "sage", "xformers"},
}
DEFAULT_SETTINGS: dict[str, Any] = {
    "auto_start_job": True,
    "device": os.getenv("WORLDFOUNDRY_STUDIO_DEVICE", "cuda"),
    "backend": "auto",
    "fps": 16,
    "num_frames": 81,
    "height": 720,
    "width": 1280,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": -1,
    "attention_backend": "auto",
    "torch_compile": False,
    "cpu_offload": False,
}
SETTINGS: dict[str, Any] = dict(DEFAULT_SETTINGS)
RUNTIME_OPTION_LABELS = {
    "torch_compile": "Torch Compile",
    "cpu_offload": "CPU Offload",
    "vae_cpu_offload": "VAE Offload",
    "text_encoder_cpu_offload": "Text Encoder Offload",
}
RUNTIME_OPTION_ALIASES = {
    "torch_compile": ("torch_compile", "enable_torch_compile", "use_torch_compile"),
    "cpu_offload": ("cpu_offload", "enable_offloading", "use_cpu_offload", "GPU_memory_mode"),
    "vae_cpu_offload": ("vae_cpu_offload", "offload_vae"),
    "text_encoder_cpu_offload": (
        "text_encoder_cpu_offload",
        "offload_t5",
        "offload_text_encoder_model",
    ),
}
TORCH_COMPILE_ENV_MODELS = {"matrix-game-2"}
MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".m4v": "video/mp4",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".webm": "video/webm",
}
MEDIA_VISUALIZER_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".mp4",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
}
SPARK_VISUALIZER_EXTS = {".spz", ".splat", ".ksplat", ".sog"}
GEOMETRY_VISUALIZER_EXTS = {".ply", ".pcd", ".xyz", ".glb", ".gltf", ".obj"}
WORKSPACE_HIDDEN_VISUALIZER_MODES = {"unified"}
VISUALIZER_LABELS = {
    "points": "Open in Viser",
    "spark": "Open in Spark",
    "rerun": "Open in Rerun",
    "media": "Open Media",
}


class JobCreateRequest(BaseModel):
    job_type: str = "inference"
    workload_type: str = ""
    model_id: str = ""
    variant_id: str = ""
    task_profile_id: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    input_path: str = ""
    model_ref: str = ""
    backend: str = "auto"
    endpoint: str = ""
    api_key: str = ""
    device: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    call_kwargs: dict[str, Any] = Field(default_factory=dict)
    load_kwargs: dict[str, Any] = Field(default_factory=dict)
    output_dir: str = ""
    eval_mode: str = "existing-results"
    benchmark_id: str = ""
    requests_path: str = ""
    results_path: str = ""
    dataset_id: str = ""
    dataset_root: str = ""
    dataset_manifest: str = ""
    model_runner: str = ""
    model_zoo_manifest_dir: str = ""
    model_variant_id: str = ""
    metrics: list[str] = Field(default_factory=lambda: ["artifact_count"])
    required_artifacts: list[str] = Field(default_factory=list)
    generation_cache_dir: str = ""
    generation_cache_mode: str = "off"
    run_plan_path: str = ""
    fail_on_sample_error: bool = False
    write_artifacts_index: bool = True
    materialize_requests: bool = False
    limit: int | None = None


class SettingsUpdateRequest(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


class VisualizerLaunchRequest(BaseModel):
    model_id: str = ""
    asset_path: str = ""
    simulator_url: str = ""
    host: str = "127.0.0.1"
    port: int | None = None
    reuse: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ManagedVisualizer:
    mode: str
    title: str
    url: str
    health_url: str
    host: str
    port: int
    model_id: str
    asset_path: str
    command: list[str]
    log_path: Path | None
    started_at: float
    params: dict[str, Any] = field(default_factory=dict)
    process: subprocess.Popen[str] | None = None
    external: bool = False


DEFAULT_VISUALIZER_MODELS = {
    "world": "matrix-game-2",
    "spark": "vggt-omega",
    "points": "vggt-omega",
    "rerun": "vggt-omega",
    "media": "vggt-omega",
    "embodied": "openvla",
    "unified": "matrix-game-2",
}
POINTS_VISUALIZER_PARAM_ENV = {
    "max_points": "WORLDFOUNDRY_STUDIO_VISER_MAX_POINTS",
    "point_size": "WORLDFOUNDRY_STUDIO_VISER_POINT_SIZE",
    "point_shape": "WORLDFOUNDRY_STUDIO_VISER_POINT_SHAPE",
    "coordinate_preset": "WORLDFOUNDRY_STUDIO_VISER_COORDINATE_PRESET",
    "up_direction": "WORLDFOUNDRY_STUDIO_VISER_UP_DIRECTION",
    "alignment": "WORLDFOUNDRY_STUDIO_VISER_ALIGNMENT",
    "show_cameras": "WORLDFOUNDRY_STUDIO_VISER_SHOW_CAMERAS",
    "camera_size": "WORLDFOUNDRY_STUDIO_VISER_CAMERA_SIZE",
}
POINTS_VISUALIZER_DEFAULT_PARAMS = {
    "coordinate_preset": "asset-native",
    "up_direction": "+z",
    "alignment": "auto",
    "point_size": 0.02,
    "point_shape": "circle",
    "max_points": 400_000,
}
VISUALIZER_ASSET_REQUIRED = {"media", "points"}
VISUALIZER_URL_REQUIRED = {"embodied"}
VISUALIZER_MANAGED: dict[str, ManagedVisualizer] = {}


def _rerun_renderer() -> str:
    """Select the Rerun web renderer, preferring its modern WebGPU path."""

    value = os.getenv("WORLDFOUNDRY_STUDIO_RERUN_RENDERER", "webgpu").strip().lower()
    return value if value in {"webgpu", "webgl"} else "webgpu"


def _coerce_setting_value(key: str, value: Any) -> Any:
    if key not in SETTINGS:
        raise HTTPException(status_code=400, detail=f"unsupported setting: {key}")
    default = SETTINGS[key]
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"setting {key} must be an integer") from exc
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"setting {key} must be a number") from exc
    coerced = str(value)
    choices = SETTING_CHOICES.get(key)
    if choices is not None and coerced not in choices:
        raise HTTPException(status_code=400, detail=f"setting {key} must be one of: {', '.join(sorted(choices))}")
    return coerced


def _settings_file() -> Path | None:
    value = os.getenv("WORLDFOUNDRY_STUDIO_SETTINGS_FILE", "").strip()
    return Path(value).expanduser() if value else None


def _load_settings_from_disk() -> None:
    path = _settings_file()
    if path is None or not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if key in SETTINGS:
            SETTINGS[key] = _coerce_setting_value(key, value)


def _save_settings_to_disk() -> None:
    path = _settings_file()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SETTINGS, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _workspace_visualizer_dir() -> Path:
    path = Path(MANAGER.workspace_root).resolve() / "visualizers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _visualizer_public_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host in {"", "0.0.0.0", "::", "localhost"} else host
    return f"http://{browser_host}:{port}/"


def _visualizer_health_url(mode: str, url: str) -> str:
    if mode in {"world", "media", "spark"}:
        return url.rstrip("/") + "/healthz"
    return url


def _visualizer_process_alive(process: subprocess.Popen[str] | None) -> bool:
    return process is not None and process.poll() is None


def _visualizer_status(record: ManagedVisualizer) -> dict[str, Any]:
    running = record.external or _visualizer_process_alive(record.process)
    return {
        "mode": record.mode,
        "title": record.title,
        "url": record.url,
        "health_url": record.health_url,
        "host": record.host,
        "port": record.port,
        "model_id": record.model_id,
        "asset_path": record.asset_path,
        "params": dict(record.params),
        "command": record.command,
        "log_path": str(record.log_path) if record.log_path else "",
        "started_at": record.started_at,
        "running": running,
        "external": record.external,
        "returncode": record.process.poll() if record.process is not None else None,
    }


def _cleanup_finished_visualizer(mode: str) -> None:
    record = VISUALIZER_MANAGED.get(mode)
    if record is None or record.external or _visualizer_process_alive(record.process):
        return
    VISUALIZER_MANAGED.pop(mode, None)


def _stop_visualizer(mode: str) -> bool:
    record = VISUALIZER_MANAGED.pop(mode, None)
    if record is None or record.external or record.process is None:
        return record is not None
    process = record.process
    if process.poll() is not None:
        return True
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
    except (OSError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=6)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError, ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError, ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=4)
    return True


def _tcp_port_available(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in {"", "0.0.0.0", "::", "localhost"} else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, port))
    except OSError:
        return False
    return True


def _visualizer_port(mode: str, host: str, requested_port: int | None) -> int:
    backend = STUDIO_VISUALIZATIONS.backend_for(mode)
    preferred = int(requested_port or backend.default_port)
    if requested_port is not None:
        if not _tcp_port_available(host, preferred):
            raise HTTPException(status_code=409, detail=f"port {preferred} is already in use")
        return preferred
    for offset in range(64):
        port = preferred + offset
        if _tcp_port_available(host, port):
            return port
    raise HTTPException(status_code=409, detail=f"no free port found near {preferred}")


def _wait_for_visualizer(url: str, *, timeout: float = 35.0) -> bool:
    deadline = time.time() + timeout
    request = urllib.request.Request(url, headers={"User-Agent": "WorldFoundry Workspace"})
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                if 200 <= int(response.status) < 500:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _visualizer_startup_timeout(mode: str) -> float:
    if mode == "world":
        return 120.0
    if mode in {"unified", "rerun", "points"}:
        return 60.0
    return 45.0


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
    except (OSError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=6)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError, ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError, ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=4)


def _visualizer_env_overrides(mode: str, params: Mapping[str, Any]) -> dict[str, str]:
    """Convert supported visualizer params to child-process environment values."""

    if mode != "points":
        return {}
    overrides: dict[str, str] = {}
    for param_key, env_key in POINTS_VISUALIZER_PARAM_ENV.items():
        value = params.get(param_key)
        if value is None:
            continue
        if isinstance(value, bool):
            text = "1" if value else "0"
        else:
            text = str(value).strip()
        if text:
            overrides[env_key] = text
    return overrides


def _resolved_visualizer_params(
    mode: str,
    *,
    model_id: str,
    asset_path: str,
    requested: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve automatic viewer orientation without overriding explicit choices."""

    if mode != "points":
        return dict(requested)
    params = dict(POINTS_VISUALIZER_DEFAULT_PARAMS)
    params.update(viser_orientation_defaults(model_id, asset_path))
    for key, value in requested.items():
        if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
            continue
        params[key] = value
    return params


def _visualizer_reusable(
    existing: ManagedVisualizer | None,
    *,
    model_id: str,
    asset_path: str,
    params: Mapping[str, Any],
) -> bool:
    """Return whether a managed viewer already serves this exact target."""

    return bool(
        existing is not None
        and existing.model_id == model_id
        and existing.asset_path == asset_path
        and existing.params == dict(params)
        and (existing.external or _visualizer_process_alive(existing.process))
    )


def _validate_visualizer_asset(mode: str, asset_path: str) -> str:
    value = (asset_path or "").strip()
    if not value:
        if mode in VISUALIZER_ASSET_REQUIRED:
            raise HTTPException(status_code=400, detail=f"{mode} requires an asset path")
        return ""
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"asset does not exist: {path}")
    return str(path)


def _workspace_child_python() -> str:
    """Return the Python entrypoint child Studio frontends should reuse."""

    return (
        os.getenv("WORLDFOUNDRY_STUDIO_CHILD_PYTHON", "").strip()
        or os.getenv("PYTHON", "").strip()
        or sys.executable
    )


def _visualizer_launch_command(mode: str, payload: VisualizerLaunchRequest, host: str, port: int) -> tuple[list[str], str, str]:
    backend = STUDIO_VISUALIZATIONS.backend_for(mode)
    model_id = (payload.model_id or DEFAULT_VISUALIZER_MODELS.get(mode) or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail=f"{mode} requires a model id")
    try:
        find_entry(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    asset_path = _validate_visualizer_asset(mode, payload.asset_path)
    external_url = (payload.simulator_url or "").strip()
    if mode in VISUALIZER_URL_REQUIRED and not external_url:
        raise HTTPException(status_code=400, detail=f"{mode} requires a simulator URL")
    if mode in {"embodied", "rerun"} and external_url:
        return [], model_id, asset_path
    module = "worldfoundry.studio.cli" if mode == "unified" else "worldfoundry.studio.native_app"
    cmd = [
        _workspace_child_python(),
        "-m",
        module,
        model_id,
        "--frontend",
        mode,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if asset_path:
        cmd.extend(["--asset", asset_path])
    return cmd, model_id, asset_path


def _visualizer_mode_for_artifact(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    suffix = path.suffix.lower()
    if suffix in MEDIA_VISUALIZER_EXTS:
        return "media"
    if suffix == ".rrd":
        return "rerun"
    if suffix in SPARK_VISUALIZER_EXTS:
        return "spark"
    if suffix == ".ply" and _is_gaussian_splat_ply(path):
        return "spark"
    if suffix == ".npz":
        return "points" if npz_has_supported_geometry(path) else ""
    if suffix in GEOMETRY_VISUALIZER_EXTS:
        return "points"
    return ""


def _artifact_visualization_action(
    name: str,
    path_text: str,
    *,
    model_id: str = "",
    output_dir: str = "",
) -> dict[str, Any] | None:
    mode = _visualizer_mode_for_artifact(path_text)
    if not mode:
        return None
    path = str(path_text)
    return {
        "name": name or Path(path).name,
        "path": path,
        "mode": mode,
        "label": VISUALIZER_LABELS.get(mode, f"Open in {mode}"),
        "model_id": model_id or DEFAULT_VISUALIZER_MODELS.get(mode, ""),
        "output_dir": output_dir,
    }


def _launch_visualizer(mode: str, payload: VisualizerLaunchRequest) -> dict[str, Any]:
    if mode not in STUDIO_VISUALIZATIONS.modes:
        raise HTTPException(status_code=404, detail=f"unknown visualizer: {mode}")
    if mode in WORKSPACE_HIDDEN_VISUALIZER_MODES:
        raise HTTPException(status_code=410, detail=f"{mode} is not exposed in the Workspace visualizers.")
    requested_model_id = (payload.model_id or DEFAULT_VISUALIZER_MODELS.get(mode) or "").strip()
    requested_asset_path = (payload.asset_path or "").strip()
    if requested_asset_path:
        requested_asset_path = str(Path(requested_asset_path).expanduser().resolve())
    params = _resolved_visualizer_params(
        mode,
        model_id=requested_model_id,
        asset_path=requested_asset_path,
        requested=dict(payload.params or {}),
    )
    _cleanup_finished_visualizer(mode)
    existing = VISUALIZER_MANAGED.get(mode)
    if (
        payload.reuse
        and _visualizer_reusable(
            existing,
            model_id=requested_model_id,
            asset_path=requested_asset_path,
            params=params,
        )
    ):
        return _visualizer_status(existing)

    if existing is not None:
        _stop_visualizer(mode)

    backend = STUDIO_VISUALIZATIONS.backend_for(mode)
    host = (payload.host or "127.0.0.1").strip() or "127.0.0.1"
    external_url = (payload.simulator_url or "").strip()
    port = int(payload.port or backend.default_port)
    if not (mode in {"embodied", "rerun"} and external_url):
        port = _visualizer_port(mode, host, payload.port)
    command, model_id, asset_path = _visualizer_launch_command(mode, payload, host, port)

    if mode in {"embodied", "rerun"} and external_url:
        record = ManagedVisualizer(
            mode=mode,
            title=backend.title,
            url=external_url,
            health_url=external_url,
            host=host,
            port=port,
            model_id=model_id,
            asset_path=asset_path,
            command=[],
            log_path=None,
            started_at=time.time(),
            params=params,
            process=None,
            external=True,
        )
        VISUALIZER_MANAGED[mode] = record
        return _visualizer_status(record)

    url = _visualizer_public_url(host, port)
    rerun_grpc_port: int | None = None
    rerun_ws_port: int | None = None
    if mode == "rerun":
        for candidate in range(port + 1, port + 65):
            if _tcp_port_available(host, candidate) and _tcp_port_available(host, candidate + 1):
                rerun_ws_port = candidate
                rerun_grpc_port = candidate + 1
                break
        if rerun_ws_port is None or rerun_grpc_port is None:
            raise HTTPException(status_code=409, detail=f"no free Rerun data ports found near {port + 1}")
        browser_host = "127.0.0.1" if host in {"", "0.0.0.0", "::", "localhost"} else host
        source_url = f"ws://{browser_host}:{rerun_ws_port}"
        url = (
            url.rstrip("/")
            + "/?url="
            + urllib.parse.quote(source_url, safe="")
            + f"&renderer={_rerun_renderer()}"
        )
    health_url = _visualizer_health_url(mode, url)
    log_path = _workspace_visualizer_dir() / f"{mode}-{int(time.time())}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if item
    )
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    visualizer_env = _visualizer_env_overrides(mode, params)
    if rerun_grpc_port is not None:
        visualizer_env["WORLDFOUNDRY_STUDIO_RERUN_GRPC_PORT"] = str(rerun_grpc_port)
    if rerun_ws_port is not None:
        visualizer_env["WORLDFOUNDRY_STUDIO_RERUN_WS_PORT"] = str(rerun_ws_port)
    env.update(visualizer_env)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        if visualizer_env:
            log_file.write("# visualizer env " + json.dumps(visualizer_env, sort_keys=True) + "\n")
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    ready = _wait_for_visualizer(health_url, timeout=_visualizer_startup_timeout(mode))
    if not ready:
        returncode = process.poll()
        if returncode is None:
            _terminate_process_group(process)
        details = ""
        with contextlib.suppress(OSError):
            details = log_path.read_text(encoding="utf-8")[-4000:]
        if returncode is None:
            raise HTTPException(
                status_code=504,
                detail=f"{mode} did not become ready at {health_url} within {_visualizer_startup_timeout(mode):.0f}s.\n{details}",
            )
        raise HTTPException(status_code=500, detail=f"{mode} exited before it was ready.\n{details}")

    record = ManagedVisualizer(
        mode=mode,
        title=backend.title,
        url=url,
        health_url=health_url,
        host=host,
        port=port,
        model_id=model_id,
        asset_path=asset_path,
        command=command,
        log_path=log_path,
        started_at=time.time(),
        params=params,
        process=process,
    )
    VISUALIZER_MANAGED[mode] = record
    return _visualizer_status(record)


def _entry_workload(entry: CatalogEntry) -> str:
    if entry.module_path == "worldfoundry.pipelines.sana.pipeline_sana" and not entry.model_id.startswith(
        ("sana-video-", "longsana-video-", "sana-streaming-")
    ):
        return "image"
    task_type = entry.default_task_type.strip().lower().replace("_", "-")
    if task_type in {"t2v", "text-video", "text-to-video", "video-generation"}:
        return "t2v"
    if task_type in {"i2v", "image-video", "image-to-video"}:
        return "i2v"
    if task_type in {"video-to-video", "v2v"}:
        return "v2v"
    if task_type in {"video-to-audio", "v2a"}:
        return "v2a"
    if task_type in {
        "class-conditional-image-generation",
        "class-conditional-generation",
        "image-generation",
        "text-to-image",
        "t2i",
    }:
        return "image"
    template_id = _template_id_hint(entry)
    if template_id == "video-to-video":
        return "v2v"
    if template_id == "conditioned-video":
        return "i2v"
    if template_id == "text-video":
        return "t2v"
    if template_id == "scene-3d":
        return "3d"
    if template_id == "depth-geometry":
        return "geometry"
    if template_id in {"embodied-policy", "visual-action"}:
        return "action"
    if template_id == "hosted-api":
        return "api"
    return "world"


def _entry_extra_variants(entry: CatalogEntry) -> tuple[InferenceVariantSpec, ...]:
    variants: list[InferenceVariantSpec] = []
    for raw_variant in entry.extra_variants:
        variant_id = str(raw_variant.get("variant_id") or "").strip()
        if not variant_id:
            continue
        checkpoints = tuple(
            InferenceCheckpointRef(
                role=str(raw_checkpoint.get("role") or "primary"),
                uri=str(raw_checkpoint.get("uri") or ""),
                required=bool(raw_checkpoint.get("required", True)),
                status=str(raw_checkpoint.get("status") or "unknown"),
            )
            for raw_checkpoint in raw_variant.get("checkpoints", ()) or ()
        )
        variants.append(
            InferenceVariantSpec(
                variant_id=variant_id,
                label=str(raw_variant.get("label") or variant_id),
                checkpoints=checkpoints,
                status=str(raw_variant.get("status") or "configured"),
                load_kwargs=dict(raw_variant.get("load_kwargs") or {}),
                call_kwargs=dict(raw_variant.get("call_kwargs") or {}),
                aliases=tuple(str(item) for item in raw_variant.get("aliases", ()) or ()),
                notes=tuple(str(item) for item in raw_variant.get("notes", ()) or ()),
            )
        )
    return tuple(variants)


def _entry_extra_variant_ids(entry: CatalogEntry) -> set[str]:
    return {str(raw_variant.get("variant_id") or "").strip() for raw_variant in entry.extra_variants}


def _append_task_inputs(
    spec: Any,
    *,
    model_id: str,
    fields: Sequence[InferenceFieldSpec],
):
    """Add model-specific UI fields without duplicating generic inputs."""

    if not fields:
        return spec
    patched_tasks = []
    for task in spec.tasks:
        existing = {(field.target, _param_key(field.field_id)) for field in task.inputs}
        merged = list(task.inputs)
        for field in fields:
            key = (field.target, _param_key(field.field_id))
            if key not in existing:
                merged.append(field)
                existing.add(key)
        patched_tasks.append(replace(task, inputs=tuple(merged)))
    return replace(spec, tasks=tuple(patched_tasks))


def _entry_inference_spec(entry: CatalogEntry):
    curated = get_model_inference_spec(entry.model_id)
    if curated is not None:
        return curated
    if entry.model_id in ASSET_GATED_WORLD_RUNTIME_MODEL_IDS:
        spec = generic_model_inference_spec(
            model_family_id=entry.model_id,
            display_name=entry.display_name,
            default_model_ref=entry.default_model_ref,
            default_load_kwargs=entry.default_load_kwargs,
            default_call_kwargs=entry.default_call_kwargs,
            supports_stream=entry.supports_stream,
            workload_type=_entry_workload(entry),
            supported_call_params=None,
        )
        return replace(
            spec,
            tasks=tuple(
                replace(
                    task,
                    inputs=tuple(
                        replace(field, default=entry.default_interactions or ("forward",))
                        if field.target == "params"
                        and _param_key(field.field_id) in {"interactions", "interaction", "interaction_signal", "action"}
                        else field
                        for field in task.inputs
                    ),
                )
                for task in spec.tasks
            ),
        )
    supported_call_params = entry.input_params or (*entry.call_params, *entry.stream_params)
    if entry.family == "world_model" and not supported_call_params:
        supported_call_params = None
    spec = model_inference_spec(
        model_family_id=entry.model_id,
        display_name=entry.display_name,
        default_model_ref=entry.default_model_ref,
        default_load_kwargs=entry.default_load_kwargs,
        default_call_kwargs=entry.default_call_kwargs,
        supports_stream=entry.supports_stream,
        workload_type=_entry_workload(entry),
        supported_call_params=supported_call_params,
    )
    if entry.model_id in {"lagernvs", "stable-virtual-camera", "wonderjourney"}:
        spec = replace(
            spec,
            tasks=tuple(
                replace(
                    task,
                    label="Video Inference",
                    outputs=(
                        InferenceArtifactSpec("video", "video", required=True, preview=True),
                        InferenceArtifactSpec("manifest", "manifest", required=True),
                    ),
                )
                for task in spec.tasks
            ),
        )
    if entry.model_id in {"dvlt", "lingbot-map"}:
        spec = replace(
            spec,
            tasks=tuple(
                replace(
                    task,
                    label="3D Reconstruction",
                    outputs=(
                        InferenceArtifactSpec("model", "generated_3d_asset", required=True, preview=True),
                        InferenceArtifactSpec("manifest", "manifest", required=True),
                    ),
                )
                for task in spec.tasks
            ),
        )
    task_type = entry.default_task_type.strip().lower().replace("_", "-")
    workload_type = _entry_workload(entry)
    if entry.model_id == "allegro_ti2v":
        spec = _append_task_inputs(
            spec,
            model_id=entry.model_id,
            fields=(
                InferenceFieldSpec(
                    "num_sampling_steps",
                    "Sampling Steps",
                    kind="integer",
                    target="load_kwargs",
                    default=entry.default_load_kwargs.get("num_sampling_steps", 100),
                ),
                InferenceFieldSpec(
                    "guidance_scale",
                    "Guidance",
                    kind="number",
                    target="load_kwargs",
                    default=entry.default_load_kwargs.get("guidance_scale", 8),
                ),
                InferenceFieldSpec(
                    "seed",
                    "Seed",
                    kind="integer",
                    target="load_kwargs",
                    default=entry.default_load_kwargs.get("seed", 1427329220),
                ),
            ),
        )
    if workload_type in {"v2v", "video-to-video"}:
        spec = replace(
            spec,
            tasks=tuple(
                replace(
                    task,
                    label="Video-to-Video Inference",
                    outputs=(
                        InferenceArtifactSpec("video", "video", required=True, preview=True),
                        InferenceArtifactSpec("manifest", "manifest", required=True),
                    ),
                )
                for task in spec.tasks
            ),
        )
    if workload_type in {"t2v", "text-video", "text-to-video"}:
        spec = replace(
            spec,
            tasks=tuple(
                replace(
                    task,
                    label="Video Inference",
                    inputs=tuple(field for field in task.inputs if field.target != "input_path"),
                    outputs=(
                        InferenceArtifactSpec("video", "video", required=True, preview=True),
                        InferenceArtifactSpec("manifest", "manifest", required=True),
                    ),
                )
                for task in spec.tasks
            ),
        )
    if workload_type == "image" or task_type in {
        "class-conditional-image-generation",
        "class-conditional-generation",
        "image-generation",
        "text-to-image",
        "t2i",
    }:
        spec = replace(
            spec,
            default_task_id="image-generation",
            tasks=tuple(
                replace(
                    task,
                    task_id="image-generation",
                    label="Image Inference",
                    inputs=tuple(field for field in task.inputs if field.target != "input_path"),
                    outputs=(
                        InferenceArtifactSpec("image", "generated_image", required=True, preview=True),
                        InferenceArtifactSpec("manifest", "manifest", required=True),
                    ),
                )
                for task in spec.tasks
            ),
        )
    if entry.default_interactions:
        tasks = []
        for task in spec.tasks:
            inputs = []
            changed = False
            for field in task.inputs:
                if (
                    field.target == "params"
                    and _param_key(field.field_id) in {"interactions", "interaction", "interaction_signal", "action"}
                    and (field.default is None or field.default == "")
                ):
                    inputs.append(replace(field, default=entry.default_interactions))
                    changed = True
                else:
                    inputs.append(field)
            tasks.append(replace(task, inputs=tuple(inputs)) if changed else task)
        spec = replace(spec, tasks=tuple(tasks))
    if entry.default_prompt:
        tasks = []
        for task in spec.tasks:
            inputs = []
            changed = False
            for field in task.inputs:
                if field.target == "prompt" and (field.default is None or field.default == ""):
                    inputs.append(replace(field, default=entry.default_prompt))
                    changed = True
                else:
                    inputs.append(field)
            tasks.append(replace(task, inputs=tuple(inputs)) if changed else task)
        spec = replace(spec, tasks=tuple(tasks))
    if entry.default_input_path:
        tasks = []
        for task in spec.tasks:
            inputs = []
            changed = False
            for field in task.inputs:
                if field.target == "input_path":
                    inputs.append(replace(field, default=entry.default_input_path))
                    changed = True
                else:
                    inputs.append(field)
            tasks.append(replace(task, inputs=tuple(inputs)) if changed else task)
        spec = replace(spec, tasks=tuple(tasks))
    extra_variants = _entry_extra_variants(entry)
    if not extra_variants:
        return spec
    existing_ids = {variant.variant_id for variant in spec.variants}
    merged = spec.variants + tuple(variant for variant in extra_variants if variant.variant_id not in existing_ids)
    return replace(spec, variants=merged)


def _entry_runtime_param_names(entry: CatalogEntry) -> set[str]:
    return set(entry.load_params) | set(entry.call_params) | set(entry.stream_params)


def _entry_runtime_options(entry: CatalogEntry) -> dict[str, dict[str, Any]]:
    names = _entry_runtime_param_names(entry)
    options: dict[str, dict[str, Any]] = {}
    for key, aliases in RUNTIME_OPTION_ALIASES.items():
        matched = [alias for alias in aliases if alias in names]
        supported = bool(matched)
        if key == "torch_compile" and entry.model_id in TORCH_COMPILE_ENV_MODELS:
            supported = True
            matched.append("WORLDFOUNDRY_ENABLE_TORCH_COMPILE")
        options[key] = {
            "label": RUNTIME_OPTION_LABELS[key],
            "supported": supported,
            "targets": matched,
        }
    return options


def _variant_model_ref(entry: CatalogEntry, variant: InferenceVariantSpec) -> str:
    if entry.model_id == "cosmos3":
        if variant.variant_id == "cosmos3-super":
            return variant.primary_checkpoint_uri or entry.default_model_ref
        return entry.default_model_ref or variant.primary_checkpoint_uri
    if entry.family == "world_model":
        return entry.default_model_ref
    if entry.model_id == LINGBOT_WORLD_MODEL_ID and variant.variant_id in {
        LINGBOT_VARIANT_BASE_CAM,
        LINGBOT_VARIANT_FAST,
    }:
        return entry.default_model_ref
    return variant.primary_checkpoint_uri or entry.default_model_ref


def _variant_load_kwargs(entry: CatalogEntry, variant: InferenceVariantSpec) -> dict[str, Any]:
    load_kwargs = dict(variant.load_kwargs)
    if entry.model_id == LINGBOT_WORLD_MODEL_ID:
        if variant.variant_id == LINGBOT_VARIANT_FAST:
            load_kwargs.update(lingbot_world_fast_load_kwargs())
            load_kwargs.setdefault("runtime_variant", "fast")
        elif variant.variant_id in {LINGBOT_VARIANT_BASE_CAM, LINGBOT_VARIANT_BASE_ACT_PREVIEW}:
            load_kwargs.update({"runtime_variant": None, "fast_model_path": None})
    return load_kwargs


def _resolve_inference_contract(
    entry: CatalogEntry,
    payload: JobCreateRequest,
) -> tuple[InferenceVariantSpec, InferenceTaskProfile, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = _entry_inference_spec(entry)
    try:
        if payload.variant_id:
            variant = spec.variant(payload.variant_id)
        else:
            try:
                variant = spec.variant(payload.model_id)
            except ValueError:
                variant = spec.variant()
        task = spec.task(payload.task_profile_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model_ref = payload.model_ref or _variant_model_ref(entry, variant)
    load_kwargs = _variant_load_kwargs(entry, variant)
    for field in task.inputs:
        if field.target == "load_kwargs" and field.default is not None:
            # Task profiles may specialize a variant's loading policy.  For
            # example, Cosmos3 action inference intentionally skips the audio
            # tokenizer even though the same Nano checkpoint loads it for
            # sound-generation tasks.  Explicit request load_kwargs are merged
            # later and therefore remain the final override.
            load_kwargs[field.field_id] = field.default
    call_kwargs = {}
    if entry.model_id == "cosmos3":
        # Cosmos3 variants carry a complete T2V fallback, while the selected
        # task owns modality-specific values such as num_frames/output_type and
        # scheduler policy. Let the explicit task contract win for all variants.
        call_kwargs.update(dict(variant.call_kwargs))
        call_kwargs.update(dict(task.default_call_kwargs))
    else:
        if variant.variant_id not in _entry_extra_variant_ids(entry):
            call_kwargs.update(dict(task.default_call_kwargs))
        call_kwargs.update(dict(variant.call_kwargs))
    contract = {
        "model_family_id": entry.model_id,
        "variant_id": variant.variant_id,
        "task_profile_id": task.task_id,
        "variant": variant.to_dict(),
        "task": task.to_dict(),
    }
    return variant, task, model_ref, call_kwargs, load_kwargs, contract


def _catalog_url_value(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("http://", "https://")):
            return text
        return ""
    if isinstance(value, Mapping):
        for key in ("url", "href", "link"):
            text = str(value.get(key) or "").strip()
            if text.startswith(("http://", "https://")):
                return text
    return ""


def _catalog_link_keys(*values: str) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        candidates = {
            text,
            text.replace("_", "-"),
            text.replace("-", "_"),
        }
        for candidate in candidates:
            normalized = candidate.casefold()
            compact = normalized.replace("-", "").replace("_", "")
            for key in (normalized, compact):
                if key and key not in seen:
                    seen.add(key)
                    keys.append(key)
    return tuple(keys)


def _merge_official_links(*link_rows: Mapping[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for row in link_rows:
        for key, value in row.items():
            text = str(value or "").strip()
            if text:
                merged[key] = text
    return merged


def _official_links_from_sources(sources: Mapping[str, Any]) -> dict[str, str]:
    links: dict[str, str] = {}
    github = _catalog_url_value(sources.get("github"))
    if github:
        links["github"] = github
    paper = (
        _catalog_url_value(sources.get("paper"))
        or _catalog_url_value(sources.get("paper_url"))
        or _catalog_url_value(sources.get("paper_plus_plus"))
        or _catalog_url_value(sources.get("arxiv"))
    )
    if paper:
        links["paper"] = paper
    project = (
        _catalog_url_value(sources.get("project_page"))
        or _catalog_url_value(sources.get("project"))
        or _catalog_url_value(sources.get("homepage"))
        or _catalog_url_value(sources.get("worldarena_space"))
    )
    if project:
        links["project"] = project
    return links


def _official_links_from_catalog_entry(entry: Mapping[str, Any]) -> dict[str, str]:
    from worldfoundry.evaluation.models.catalog.schema import _github_url_from_sources

    links: dict[str, str] = {}
    github = _github_url_from_sources(entry)
    if github:
        links["github"] = github

    official_sources = entry.get("official_sources")
    sources = official_sources if isinstance(official_sources, Mapping) else {}
    if not sources:
        raw_sources = entry.get("sources")
        if isinstance(raw_sources, Mapping):
            sources = raw_sources

    source_status = entry.get("source_status")
    if isinstance(source_status, Mapping):
        links = _merge_official_links(links, _official_links_from_sources(source_status))

    links = _merge_official_links(links, _official_links_from_sources(sources))

    paper = _catalog_url_value(entry.get("paper_url")) or _catalog_url_value(entry.get("paper"))
    if paper:
        links.setdefault("paper", paper)
    project = _catalog_url_value(entry.get("project_page")) or _catalog_url_value(entry.get("project"))
    if project:
        links.setdefault("project", project)
    return links


def _catalog_entry_link_keys(entry: Mapping[str, Any]) -> tuple[str, ...]:
    aliases = entry.get("aliases") or ()
    alias_rows = aliases if isinstance(aliases, (list, tuple)) else (aliases,)
    keys = _catalog_link_keys(
        str(entry.get("model_id") or entry.get("id") or ""),
        str(entry.get("pipeline_binding") or ""),
        *(str(alias) for alias in alias_rows),
    )
    variants = entry.get("variants") or ()
    if isinstance(variants, (list, tuple)):
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            keys += _catalog_link_keys(
                str(variant.get("id") or variant.get("variant_id") or ""),
                str(variant.get("pipeline_binding") or ""),
            )
    return keys


def _entry_link_keys(entry: CatalogEntry) -> tuple[str, ...]:
    return _catalog_link_keys(entry.model_id, *entry.aliases)


def _github_url_from_model_ref(model_ref: str) -> str:
    text = str(model_ref or "").strip()
    if "github.com/" in text.casefold():
        return text
    return ""


@lru_cache(maxsize=1)
def _model_catalog_links_index() -> dict[str, dict[str, str]]:
    from worldfoundry.evaluation.models.catalog.manifest import _catalog_paths, _iter_catalog_mappings

    index: dict[str, dict[str, str]] = {}
    for path in _catalog_paths():
        for entry in _iter_catalog_mappings(path):
            links = _official_links_from_catalog_entry(entry)
            if not links:
                continue
            for key in _catalog_entry_link_keys(entry):
                if not key:
                    continue
                existing = index.get(key)
                index[key] = _merge_official_links(existing or {}, links)
    return index


def _entry_official_links(entry: CatalogEntry) -> dict[str, str]:
    index = _model_catalog_links_index()
    links: dict[str, str] = {}
    for key in _entry_link_keys(entry):
        links = _merge_official_links(links, index.get(key, {}))
        if all(links.get(name) for name in ("github", "project", "paper")):
            break
    github_ref = _github_url_from_model_ref(entry.default_model_ref)
    if github_ref:
        links.setdefault("github", github_ref)
    return links


def _model_payload(entry: CatalogEntry) -> dict[str, Any]:
    template_id = _template_id_hint(entry)
    infer_spec = _entry_inference_spec(entry)
    variant_payloads = []
    for variant in infer_spec.variants:
        row = variant.to_dict()
        row["model_ref"] = _variant_model_ref(entry, variant)
        row["load_kwargs"] = _variant_load_kwargs(entry, variant)
        variant_payloads.append(row)
    return {
        "id": entry.model_id,
        "name": entry.display_name,
        "category": entry.category,
        "family": entry.family,
        "summary": entry.summary,
        "tags": list(entry.tags),
        "backend": entry.default_backend,
        "model_ref": entry.default_model_ref,
        "endpoint": entry.default_endpoint,
        "default_prompt": entry.default_prompt,
        "default_input_path": entry.default_input_path,
        "supports_stream": entry.supports_stream,
        "supports_from_pretrained": entry.supports_from_pretrained,
        "supports_api_init": entry.supports_api_init,
        "supports_attention_backend": _supports_attention_backend(entry),
        "template_id": template_id,
        "workload_type": _entry_workload(entry),
        "infer_spec": infer_spec.to_dict(),
        "variants": variant_payloads,
        "tasks": [task.to_dict() for task in infer_spec.tasks],
        "default_variant_id": infer_spec.default_variant_id,
        "default_task_id": infer_spec.default_task_id,
        "runtime_options": _entry_runtime_options(entry),
        "links": _entry_official_links(entry),
    }


@lru_cache(maxsize=1)
def _workspace_models() -> tuple[dict[str, Any], ...]:
    return tuple(_model_payload(entry) for entry in _studio_catalog())


@lru_cache(maxsize=1)
def _workspace_model_ids() -> frozenset[str]:
    return frozenset(entry.model_id for entry in _studio_catalog())


def _workspace_job_output_dir(kind: str, job_id: str | None) -> str:
    identifier = job_id or "pending"
    return str(Path(MANAGER.workspace_root) / kind / identifier)


def _non_empty_list(values: Sequence[str] | None, default: Sequence[str] = ()) -> tuple[str, ...]:
    rows = tuple(str(item).strip() for item in (values or ()) if str(item).strip())
    return rows or tuple(default)


def _optional_path(value: str | Path | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _safe_jsonable(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(item) for item in value]
    return value


def _preview_input_path_from_call_kwargs(call_kwargs: Mapping[str, Any]) -> str:
    for key in (
        "input_path",
        "image",
        "images",
        "image_path",
        "video",
        "video_path",
        "top_cam",
        "agentview_cam",
        "external_cam",
        "left_cam",
        "side_cam",
        "wrist_cam",
        "right_cam",
    ):
        value = call_kwargs.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if isinstance(item, str) and item:
                    return item
        if isinstance(value, Mapping):
            for item in value.values():
                if isinstance(item, str) and item:
                    return item
    operator_kwargs = call_kwargs.get("operator_kwargs")
    if isinstance(operator_kwargs, Mapping):
        return _preview_input_path_from_call_kwargs(operator_kwargs)
    return ""


def _write_workspace_json(path: str | Path, payload: Any) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_safe_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(destination)


@lru_cache(maxsize=1)
def _evaluation_catalog_payload() -> dict[str, Any]:
    try:
        from worldfoundry.evaluation.models.catalog.registry import discover_model_registry
        from worldfoundry.evaluation.tasks.catalog.specs import list_benchmark_zoo_cli_tasks
        from worldfoundry.evaluation.tasks.metrics.registry import list_metric_registry_entries

        benchmarks = list_benchmark_zoo_cli_tasks()
        models = [item.to_dict() for item in discover_model_registry().list()]
        metrics = [item.to_dict() for item in list_metric_registry_entries()]
        return {
            "ok": True,
            "modes": [
                "score-artifacts",
                "model-benchmark",
                "generate-and-score",
                "existing-results",
                "model",
            ],
            "benchmarks": benchmarks,
            "models": models,
            "metrics": metrics,
            "benchmark_runtime_hints": workspace_benchmark_runtime_hints(),
            "benchmark_runtime_issues": validate_workspace_registry(),
            "examples": _evaluation_examples_payload(),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - surfaced as catalog diagnostics in the UI.
        return {
            "ok": False,
            "modes": [
                "score-artifacts",
                "model-benchmark",
                "generate-and-score",
                "existing-results",
                "model",
            ],
            "benchmarks": [],
            "models": [],
            "metrics": [],
            "benchmark_runtime_hints": {},
            "benchmark_runtime_issues": [f"{type(exc).__name__}: {exc}"],
            "examples": _evaluation_examples_payload(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _evaluation_examples_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": "existing-results-validation",
            "label": "Existing Results Validation",
            "eval_mode": "existing-results",
            "benchmark_id": "workspace-existing-results-validation",
            "model_id": "workspace-demo-model",
            "dataset_id": "workspace-validation-fixture",
            "results_path": str(EVALUATION_VALIDATION_RESULTS_PATH),
            "requests_path": "",
            "metrics": ["artifact_count", "required_artifacts_present"],
            "required_artifacts": ["generated_video"],
            "call_kwargs": {},
            "load_kwargs": {},
        }
    ]


def _param_key(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _call_param_names(entry: CatalogEntry) -> set[str]:
    names = set(entry.call_params) | set(entry.stream_params) | set(entry.default_call_kwargs)
    if entry.model_id == LINGBOT_WORLD_MODEL_ID:
        names.add("sampling_steps")
    return names


def _load_param_names(entry: CatalogEntry) -> set[str]:
    return set(entry.load_params) | set(entry.default_load_kwargs)


def _supports_attention_backend(entry: CatalogEntry) -> bool:
    names = _load_param_names(entry) | _call_param_names(entry)
    return "attention_backend" in names


def _supports_backend(entry: CatalogEntry, backend: str) -> bool:
    selected = (backend or "auto").strip()
    if selected == "auto":
        return True
    if selected == "from_pretrained":
        return entry.supports_from_pretrained
    if selected == "api_init":
        return entry.supports_api_init
    return False


def _actual_param_name(names: set[str], *aliases: str) -> str | None:
    lookup = {_param_key(name): name for name in names}
    for alias in aliases:
        match = lookup.get(_param_key(alias))
        if match:
            return match
    return None


def _supported_call_param(entry: CatalogEntry, *aliases: str) -> str | None:
    return _actual_param_name(_call_param_names(entry), *aliases)


def _sync_input_path_to_call_kwargs(
    entry: CatalogEntry,
    *,
    task_type: str,
    input_path: str,
    call_kwargs: dict[str, Any],
) -> None:
    if not input_path:
        return
    workload = _entry_workload(entry).strip().lower().replace("_", "-")
    declared_task_type = str(task_type or entry.default_task_type or "").strip().lower().replace("_", "-")
    image_input_task_types = {
        "i2i",
        "image-to-image",
        "image-editing",
        "r2v",
        "reference-to-video",
    }
    if declared_task_type in image_input_task_types:
        target = _supported_call_param(entry, "image_path", "image", "images")
        if target:
            call_kwargs[target] = [input_path] if target == "images" else input_path
        return
    if workload in {"i2v", "image-video", "image-to-video"} or declared_task_type in {"i2v", "image-video", "image-to-video"}:
        target = _supported_call_param(entry, "image_path", "image", "images")
        if target:
            call_kwargs[target] = input_path if target != "images" else [input_path]
        return
    if workload in {"action", "embodied", "embodied-policy", "robotics", "visual-action"}:
        target = _supported_call_param(entry, "image_path", "image", "images")
        if target:
            call_kwargs[target] = input_path if target != "images" else [input_path]
        return
    video_input_task_types = {"v2v", "video-video", "video-to-video", "v2a", "video-to-audio"}
    if workload in video_input_task_types or declared_task_type in video_input_task_types:
        target = _supported_call_param(entry, "video_path", "video", "videos")
        if target:
            call_kwargs[target] = input_path if target != "videos" else [input_path]
        return
    target = _supported_call_param(entry, "input_path")
    if target:
        call_kwargs[target] = input_path


def _param_key_aliases(key: str) -> tuple[str, ...]:
    normalized = _param_key(key)
    aliases = {
        "num_frames": ("num_frames", "frames", "video_length"),
        "frames": ("num_frames", "frames", "video_length"),
        "height": ("height", "user_height", "output_H", "resize_H", "image_height"),
        "width": ("width", "user_width", "output_W", "resize_W", "image_width"),
        "guidance_scale": ("guidance_scale", "cfg_scale", "scale"),
        "guidance": ("guidance_scale", "cfg_scale", "scale"),
        "seed": ("seed",),
        "fps": ("fps",),
        "num_inference_steps": ("num_inference_steps", "sampling_steps", "infer_steps", "num_steps", "steps"),
        "steps": ("num_inference_steps", "sampling_steps", "infer_steps", "num_steps", "steps"),
        "negative_prompt": ("negative_prompt",),
        "interactions": ("interactions", "interaction_signal", "interaction", "action"),
    }
    return aliases.get(normalized, (key,))


def _task_allowed_param_keys(task: InferenceTaskProfile) -> set[str]:
    allowed: set[str] = set()
    for field in task.inputs:
        if field.target != "params":
            continue
        field_key = _param_key(field.field_id)
        allowed.add(field_key)
        allowed.update(_param_key(alias) for alias in _param_key_aliases(field_key))
    return allowed


def _task_field_default(task: InferenceTaskProfile, *field_ids: str, target: str | None = None) -> Any:
    wanted = {_param_key(field_id) for field_id in field_ids}
    for field in task.inputs:
        if target is not None and field.target != target:
            continue
        if _param_key(field.field_id) in wanted and field.default is not None and field.default != "":
            return field.default
    return None


def _runtime_alias_names_for_supported_options(entry: CatalogEntry) -> set[str]:
    names: set[str] = set()
    for key, option in _entry_runtime_options(entry).items():
        if not option.get("supported"):
            continue
        names.add(key)
        names.update(str(alias) for alias in option.get("targets") or ())
        names.update(RUNTIME_OPTION_ALIASES.get(key, ()))
    return names


def _task_declared_kwargs(task: InferenceTaskProfile, target: str) -> set[str]:
    return {
        _param_key(field.field_id)
        for field in task.inputs
        if field.target == target
    }


def _validate_explicit_kwargs(entry: CatalogEntry, task: InferenceTaskProfile, payload: JobCreateRequest) -> None:
    call_names = _call_param_names(entry)
    load_names = _load_param_names(entry)
    runtime_aliases = {_param_key(name) for name in _runtime_alias_names_for_supported_options(entry)}
    call_lookup = {_param_key(name) for name in call_names} | _task_declared_kwargs(task, "call_kwargs")
    call_lookup.update(_param_key(name) for name in task.default_call_kwargs)
    load_lookup = {_param_key(name) for name in load_names} | _task_declared_kwargs(task, "load_kwargs")
    call_lookup |= {_param_key(name) for name in DISPATCH_ONLY_CALL_KWARGS}
    load_lookup |= {_param_key(name) for name in DISPATCH_ONLY_LOAD_KWARGS}
    unsupported_call = [
        key
        for key in (payload.call_kwargs or {})
        if _param_key(key) not in call_lookup and _param_key(key) not in runtime_aliases
    ]
    unsupported_load = [
        key
        for key in (payload.load_kwargs or {})
        if _param_key(key) not in load_lookup and _param_key(key) not in runtime_aliases
    ]
    if unsupported_call or unsupported_load:
        details = []
        if unsupported_call:
            details.append(f"call_kwargs={', '.join(sorted(unsupported_call))}")
        if unsupported_load:
            details.append(f"load_kwargs={', '.join(sorted(unsupported_load))}")
        raise HTTPException(
            status_code=400,
            detail=f"{entry.model_id} does not declare these inference kwargs: {'; '.join(details)}",
        )


def _field_value_provided(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    return True


def _validate_field_choice(
    entry: CatalogEntry,
    task: InferenceTaskProfile,
    field_id: str,
    choices: Sequence[str],
    value: Any,
) -> None:
    if not choices or not _field_value_provided(value):
        return
    allowed = {str(choice) for choice in choices}
    if str(value) in allowed:
        return
    raise HTTPException(
        status_code=400,
        detail=f"{entry.model_id}/{task.task_id} field {field_id} must be one of: {', '.join(str(choice) for choice in choices)}",
    )


def _task_choice_fields(task: InferenceTaskProfile, target: str) -> dict[str, tuple[str, tuple[str, ...]]]:
    fields: dict[str, tuple[str, tuple[str, ...]]] = {}
    for field in task.inputs:
        if field.target != target or not field.choices:
            continue
        field_key = _param_key(field.field_id)
        for key in (field_key, *_param_key_aliases(field_key)):
            fields[_param_key(key)] = (field.field_id, field.choices)
    return fields


def _validate_task_field_choices(entry: CatalogEntry, task: InferenceTaskProfile, payload: JobCreateRequest) -> None:
    for target, values in (
        ("params", payload.params or {}),
        ("call_kwargs", payload.call_kwargs or {}),
        ("load_kwargs", payload.load_kwargs or {}),
    ):
        fields = _task_choice_fields(task, target)
        if not fields:
            continue
        for key, value in values.items():
            field = fields.get(_param_key(key))
            if field is None:
                continue
            field_id, choices = field
            _validate_field_choice(entry, task, field_id, choices, value)

    direct_values = {
        "prompt": payload.prompt,
        "input_path": payload.input_path,
        "negative_prompt": payload.negative_prompt,
        "model_ref": payload.model_ref,
    }
    for field in task.inputs:
        if field.target not in direct_values:
            continue
        _validate_field_choice(entry, task, field.field_id, field.choices, direct_values[field.target])


def _validate_inference_payload(entry: CatalogEntry, task: InferenceTaskProfile, payload: JobCreateRequest) -> None:
    params = dict(payload.params or {})
    _validate_runtime_options(entry, params)
    _validate_task_field_choices(entry, task, payload)
    backend = payload.backend or entry.default_backend or "auto"
    if not _supports_backend(entry, backend):
        raise HTTPException(status_code=400, detail=f"{entry.model_id} does not support backend={backend}")
    if (payload.endpoint or payload.api_key) and backend != "api_init":
        raise HTTPException(status_code=400, detail="endpoint and api_key are only used with backend=api_init")
    allowed_params = _task_allowed_param_keys(task)
    runtime_param_keys = {_param_key(key) for key in RUNTIME_OPTION_ALIASES}
    model_param_keys = {_param_key(key) for key in (_call_param_names(entry) | _load_param_names(entry))}
    unsupported: list[str] = []
    for key, value in params.items():
        normalized = _param_key(key)
        if normalized in runtime_param_keys:
            continue
        if normalized == "attention_backend":
            if value in {"", None, "auto"} or _supports_attention_backend(entry):
                continue
            unsupported.append(key)
            continue
        if normalized not in allowed_params and normalized not in model_param_keys:
            unsupported.append(key)
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"{entry.model_id}/{task.task_id} does not use these params: {', '.join(sorted(unsupported))}",
        )
    _validate_explicit_kwargs(entry, task, payload)


def _is_missing_param_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _merge_common_params(
    entry: CatalogEntry,
    payload: JobCreateRequest,
    *,
    base_call_kwargs: dict[str, Any] | None = None,
    base_load_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = dict(payload.params or {})
    explicit_call_kwargs = _normalize_explicit_kwargs(
        payload.call_kwargs or {},
        set(entry.call_params) | set(entry.stream_params),
    )
    call_kwargs = dict(base_call_kwargs or {})
    call_kwargs.update(explicit_call_kwargs)
    load_kwargs = dict(base_load_kwargs or {})
    load_kwargs.update(_normalize_explicit_kwargs(payload.load_kwargs or {}, set(entry.load_params)))

    def param_value(param_key: str, *aliases: str) -> Any:
        for key in (param_key, *aliases):
            value = params.get(key)
            if key in params and not _is_missing_param_value(value):
                return value
            normalized = _param_key(key)
            value = params.get(normalized)
            if normalized in params and not _is_missing_param_value(value):
                return value
        return None

    def set_call_from_param(param_key: str, *aliases: str) -> None:
        value = param_value(param_key, *aliases)
        if _is_missing_param_value(value):
            return
        target = _supported_call_param(entry, *aliases)
        if target and target not in explicit_call_kwargs:
            call_kwargs[target] = value

    def set_load_from_param(param_key: str, *aliases: str) -> None:
        value = param_value(param_key, *aliases)
        if _is_missing_param_value(value):
            return
        load_names = {_param_key(name): name for name in entry.load_params}
        for alias in aliases:
            target = load_names.get(_param_key(alias))
            if target and target not in load_kwargs:
                load_kwargs[target] = value
                return

    set_call_from_param("num_frames", "num_frames", "frames", "video_length")
    set_load_from_param("num_frames", "num_frames", "frames", "video_length")
    set_call_from_param("height", "height", "user_height", "output_H", "resize_H", "image_height")
    set_load_from_param("height", "height", "user_height", "output_H", "resize_H", "image_height")
    set_call_from_param("width", "width", "user_width", "output_W", "resize_W", "image_width")
    set_load_from_param("width", "width", "user_width", "output_W", "resize_W", "image_width")
    set_call_from_param("guidance_scale", "guidance_scale", "cfg_scale", "scale")
    set_load_from_param("guidance_scale", "guidance_scale", "cfg_scale", "scale")
    set_call_from_param("seed", "seed")
    set_load_from_param("seed", "seed")
    set_call_from_param("fps", "fps")
    set_load_from_param("fps", "fps", "frame_rate")
    set_call_from_param("num_inference_steps", "num_inference_steps", "steps", "sampling_steps", "infer_steps", "num_steps")
    set_load_from_param("num_inference_steps", "num_inference_steps", "steps", "sampling_steps", "infer_steps", "num_steps")
    if payload.negative_prompt:
        negative_key = _supported_call_param(entry, "negative_prompt")
        if negative_key:
            call_kwargs.setdefault(negative_key, payload.negative_prompt)

    attention_backend = param_value("attention_backend")
    if _supports_attention_backend(entry) and attention_backend not in {"", None, "auto"}:
        load_kwargs.setdefault("attention_backend", attention_backend)
        call_kwargs.setdefault("attention_backend", attention_backend)
    _apply_runtime_options(entry, params=params, load_kwargs=load_kwargs, call_kwargs=call_kwargs)

    call_names = {_param_key(name): name for name in (entry.call_params + entry.stream_params)}
    load_names = {_param_key(name): name for name in entry.load_params}
    for key, value in params.items():
        if _is_missing_param_value(value):
            continue
        normalized = _param_key(key)
        call_target = call_names.get(normalized)
        if call_target and call_target not in explicit_call_kwargs:
            call_kwargs.setdefault(call_target, value)
        load_target = load_names.get(normalized)
        if load_target and load_target not in load_kwargs:
            load_kwargs.setdefault(load_target, value)

    return call_kwargs, load_kwargs


def _normalize_explicit_kwargs(values: Mapping[str, Any], supported_names: set[str]) -> dict[str, Any]:
    lookup = {_param_key(name): name for name in supported_names}
    normalized: dict[str, Any] = {}
    for key, value in dict(values).items():
        normalized[lookup.get(_param_key(key), key)] = value
    return normalized


def _runtime_option_enabled(params: dict[str, Any], key: str) -> bool:
    value = params.get(key)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _should_use_task_default_input_path(
    entry: CatalogEntry,
    payload: JobCreateRequest,
    task_type: str,
) -> bool:
    declared_task_type = str(task_type or entry.default_task_type or "").strip().lower().replace("_", "-")
    if declared_task_type in {"t2v", "text-video", "text-to-video"}:
        return False
    if declared_task_type in {
        "class-conditional-image-generation",
        "class-conditional-generation",
        "image-generation",
        "text-to-image",
        "t2i",
    }:
        return False
    if declared_task_type:
        return True
    workload = str(payload.workload_type or _entry_workload(entry) or "").strip().lower().replace("_", "-")
    if workload in {"t2v", "text-video", "text-to-video"}:
        return False
    return True


def _validate_runtime_options(entry: CatalogEntry, params: dict[str, Any]) -> None:
    options = _entry_runtime_options(entry)
    unsupported = [
        key
        for key in RUNTIME_OPTION_ALIASES
        if _runtime_option_enabled(params, key) and not options.get(key, {}).get("supported")
    ]
    if unsupported:
        labels = ", ".join(RUNTIME_OPTION_LABELS[key] for key in unsupported)
        raise HTTPException(
            status_code=400,
            detail=f"{entry.model_id} does not implement these runtime options: {labels}",
        )


def _set_runtime_option_value(target: dict[str, Any], alias: str, value: bool) -> None:
    if alias == "GPU_memory_mode":
        target.setdefault(alias, "model_cpu_offload" if value else "none")
    else:
        target.setdefault(alias, value)


def _apply_runtime_options(
    entry: CatalogEntry,
    *,
    params: dict[str, Any],
    load_kwargs: dict[str, Any],
    call_kwargs: dict[str, Any],
) -> None:
    _validate_runtime_options(entry, params)
    load_names = set(entry.load_params)
    call_names = set(entry.call_params) | set(entry.stream_params)
    for key, aliases in RUNTIME_OPTION_ALIASES.items():
        if not _runtime_option_enabled(params, key):
            continue
        if key == "torch_compile" and entry.model_id in TORCH_COMPILE_ENV_MODELS:
            load_kwargs.setdefault("torch_compile", True)
            continue
        for alias in aliases:
            if alias in load_names:
                _set_runtime_option_value(load_kwargs, alias, True)
            if alias in call_names:
                _set_runtime_option_value(call_kwargs, alias, True)


def _inference_run_kwargs(payload: JobCreateRequest, *, validate: bool = True) -> tuple[CatalogEntry, dict[str, Any]]:
    entry = find_entry(payload.model_id)
    variant, task, model_ref, variant_call_kwargs, variant_load_kwargs, contract = _resolve_inference_contract(entry, payload)
    if validate:
        _validate_inference_payload(entry, task, payload)
    call_kwargs, load_kwargs = _merge_common_params(
        entry,
        payload,
        base_call_kwargs=variant_call_kwargs,
        base_load_kwargs=variant_load_kwargs,
    )
    if variant.variant_id not in _entry_extra_variant_ids(entry):
        call_kwargs = {**entry.default_call_kwargs, **call_kwargs}
    load_kwargs = {**entry.default_load_kwargs, **load_kwargs}
    params = dict(payload.params or {})
    interactions = params.get("interactions")
    if interactions is None:
        default_interactions = _task_field_default(
            task,
            "interactions",
            "interaction",
            "interaction_signal",
            "action",
            target="params",
        )
        if isinstance(default_interactions, (str, list, tuple)):
            interactions = default_interactions
    task_type = str(params.get("task_type") or call_kwargs.pop("task_type", "") or entry.default_task_type or "")
    fps = int(params.get("fps") or call_kwargs.get("fps") or SETTINGS.get("fps", DEFAULT_SETTINGS["fps"]))
    num_frames = int(
        params.get("num_frames")
        or params.get("frames")
        or call_kwargs.get("num_frames")
        or call_kwargs.get("frame_num")
        or call_kwargs.get("frames")
        or call_kwargs.get("video_length")
        or SETTINGS.get("num_frames", DEFAULT_SETTINGS["num_frames"])
    )
    default_input_path = ""
    if _should_use_task_default_input_path(entry, payload, task_type):
        default_input_path = str(_task_field_default(task, "input_path", "image", "video", target="input_path") or "")
    input_path = payload.input_path or _preview_input_path_from_call_kwargs(call_kwargs) or default_input_path
    _sync_input_path_to_call_kwargs(entry, task_type=task_type, input_path=input_path, call_kwargs=call_kwargs)
    if entry.model_id == "matrix-game-1" and input_path:
        call_kwargs.setdefault("image_path", input_path)
    prompt = payload.prompt or str(_task_field_default(task, "prompt", target="prompt") or entry.default_prompt or "")
    return entry, dict(
        model_id=entry.model_id,
        action="run",
        prompt=prompt,
        input_path=input_path,
        image=None,
        video=None,
        last_frame=None,
        reference_files=None,
        interactions_text=json.dumps(interactions) if interactions is not None else "",
        camera_view_text=json.dumps(params.get("camera_view")) if params.get("camera_view") is not None else "",
        task_type=task_type,
        intrinsics_text=json.dumps(params.get("intrinsics")) if params.get("intrinsics") is not None else "",
        meta_path=str(params.get("meta_path") or ""),
        panorama_path=str(params.get("panorama_path") or ""),
        scene_name=str(params.get("scene_name") or ""),
        fps=fps,
        num_frames=num_frames,
        call_kwargs_text=json.dumps(call_kwargs),
        load_kwargs_text=json.dumps(load_kwargs),
        model_ref=model_ref,
        backend=payload.backend or entry.default_backend,
        endpoint=payload.endpoint or entry.default_endpoint,
        api_key=payload.api_key,
        device=payload.device or str(SETTINGS.get("device") or DEFAULT_SETTINGS["device"]),
        infer_metadata=contract,
    )


def _inference_backend_for_dispatch(entry: CatalogEntry, backend: str) -> str:
    selected = (backend or entry.default_backend or "auto").strip()
    if selected == "auto" and not entry.supports_from_pretrained and entry.supports_api_init:
        return "api_init"
    return selected


def _run_inference(payload: JobCreateRequest, job: StudioJob | None = None):
    entry, run_kwargs = _inference_run_kwargs(payload, validate=False)
    backend = _inference_backend_for_dispatch(entry, str(run_kwargs.get("backend") or "auto"))
    spec = dispatch_spec_for_inference(entry.model_id, backend=backend)
    if spec is not None:
        dispatch_root = Path(MANAGER.workspace_root) / "runtime_jobs" / (job.job_id if job else "direct")
        return run_manager_payload_in_conda(
            model_id=entry.model_id,
            spec=spec,
            workspace_root=MANAGER.workspace_root,
            run_kwargs=run_kwargs,
            dispatch_root=dispatch_root,
            log_callback=job.append_log if job is not None else None,
            cancel_requested=(lambda: bool(job.cancel_requested)) if job is not None else None,
        )
    return MANAGER.run(**run_kwargs, progress_callback=None)


_PREPARED_EVALUATION_MODES = frozenset({"score-artifacts", "model-benchmark", "generate-and-score"})


def _prepared_evaluation_from_payload(payload: JobCreateRequest, output_dir: str | Path):
    """Compile a Workspace payload through the shared evaluation intent service."""

    from worldfoundry.evaluation.tasks.execution.orchestration.service import (
        GenerateAndScoreIntent,
        ModelBenchmarkIntent,
        ScoreArtifactsIntent,
        prepare_evaluation,
    )

    mode = (payload.eval_mode or "").strip().lower().replace("_", "-")
    model_parameters = dict(payload.call_kwargs or {})
    model_runtime = dict(payload.load_kwargs or {})
    if mode == "score-artifacts":
        intent = ScoreArtifactsIntent(
            output_dir=output_dir,
            benchmark_id=payload.benchmark_id,
            artifact_dir=_optional_path(payload.dataset_root) or _optional_path(payload.results_path) or "",
            dataset_id=_optional_path(payload.dataset_id),
            benchmark_env=model_runtime,
            benchmark_parameters=model_parameters,
            leaderboard_candidate=bool(payload.params.get("leaderboard_candidate")),
        )
    elif mode == "model-benchmark":
        intent = ModelBenchmarkIntent(
            output_dir=output_dir,
            model_id=payload.model_id,
            benchmark_id=payload.benchmark_id,
            model_variant_id=_optional_path(payload.model_variant_id or payload.variant_id),
            requests_path=_optional_path(payload.requests_path),
            dataset_root=_optional_path(payload.dataset_root),
            dataset_id=_optional_path(payload.dataset_id),
            num_samples=payload.limit,
            model_parameters=model_parameters,
            model_runtime=model_runtime,
            generation_cache_dir=_optional_path(payload.generation_cache_dir),
            generation_cache_mode=payload.generation_cache_mode or "read-write",
        )
    elif mode == "generate-and-score":
        intent = GenerateAndScoreIntent(
            output_dir=output_dir,
            model_id=payload.model_id,
            model_variant_id=_optional_path(payload.model_variant_id or payload.variant_id),
            dataset_manifest=payload.dataset_manifest,
            benchmark_id=_optional_path(payload.benchmark_id),
            metrics=_non_empty_list(payload.metrics),
            input_keys=_non_empty_list(payload.params.get("input_keys")),
            output_keys=_non_empty_list(payload.params.get("output_keys"), ("generated_video",)),
            required_artifacts=_non_empty_list(payload.required_artifacts),
            generation_defaults=dict(payload.params.get("generation_defaults") or {}),
            model_parameters=model_parameters,
            model_runtime=model_runtime,
            num_samples=payload.limit,
            generation_cache_dir=_optional_path(payload.generation_cache_dir),
            generation_cache_mode=payload.generation_cache_mode or "read-write",
        )
    else:
        raise ValueError(f"unsupported prepared evaluation mode: {mode}")
    return prepare_evaluation(intent)


def _run_evaluation(payload: JobCreateRequest, job: StudioJob | None = None) -> dict[str, Any]:
    from worldfoundry.evaluation.runner import EvaluateRunRequest, run_evaluate
    from worldfoundry.evaluation.tasks.execution.orchestration.plan import evaluate_request_from_run_plan, load_run_plan, validate_run_plan

    output_dir = _optional_path(payload.output_dir) or _workspace_job_output_dir("evaluations", job.job_id if job else None)
    intent_mode = (payload.eval_mode or "existing-results").strip().lower().replace("_", "-")
    if intent_mode in _PREPARED_EVALUATION_MODES:
        from worldfoundry.evaluation.tasks.execution.orchestration.service import execute_prepared_evaluation

        prepared = _prepared_evaluation_from_payload(payload, output_dir)
        if job is not None:
            job.append_log(
                "system",
                f"evaluation intent={prepared.intent_kind} ready={prepared.ready} "
                f"classification={prepared.classification}\n",
            )
            for issue in prepared.issues:
                job.append_log("system", f"{issue.severity}: {issue.code}: {issue.message}\n")
        result = execute_prepared_evaluation(prepared)
        result_payload = result.to_dict()
        result_payload["prepared_evaluation"] = prepared.to_dict()
        return result_payload
    if payload.run_plan_path:
        plan = load_run_plan(payload.run_plan_path)
        validation = validate_run_plan(plan)
        if job is not None:
            job.append_log("system", f"loaded run plan {payload.run_plan_path}\n")
            job.append_log("system", f"run plan fingerprint={validation.get('fingerprint')}\n")
        if not validation.get("ok"):
            issues = "; ".join(str(item) for item in validation.get("issues", ()))
            raise ValueError(f"Evaluation run plan is invalid: {issues}")
        request = evaluate_request_from_run_plan(plan)
        if payload.output_dir:
            request = replace(request, output_dir=output_dir)
    else:
        if workspace_benchmark_supported(payload.benchmark_id) and workspace_benchmark_has_input(payload):
            return run_workspace_benchmark(
                payload,
                output_dir,
                log_callback=job.append_log if job is not None else None,
            )
        mode = (payload.eval_mode or "existing-results").strip().lower().replace("_", "-")
        request = EvaluateRunRequest(
            output_dir=output_dir,
            mode=mode,
            requests_path=_optional_path(payload.requests_path),
            results_path=_optional_path(payload.results_path),
            metrics=_non_empty_list(payload.metrics, ("artifact_count",)),
            required_artifacts=_non_empty_list(payload.required_artifacts),
            benchmark_id=_optional_path(payload.benchmark_id),
            model_id=_optional_path(payload.model_id),
            model_runner=_optional_path(payload.model_runner),
            model_zoo_manifest_dir=_optional_path(payload.model_zoo_manifest_dir),
            model_variant_id=_optional_path(payload.model_variant_id or payload.variant_id),
            model_parameters=dict(payload.call_kwargs or {}),
            model_runtime=dict(payload.load_kwargs or {}),
            model_config=payload.params.get("model_config") if isinstance(payload.params, dict) else None,
            dataset_id=_optional_path(payload.dataset_id),
            dataset=_safe_jsonable(
                {
                    "root": _optional_path(payload.dataset_root),
                    "manifest_path": _optional_path(payload.dataset_manifest),
                }
            ),
            fail_on_sample_error=bool(payload.fail_on_sample_error),
            write_artifacts_index=bool(payload.write_artifacts_index),
            generation_cache_dir=_optional_path(payload.generation_cache_dir),
            generation_cache_mode=payload.generation_cache_mode or "off",
            generation_cache_namespace=str(payload.params.get("generation_cache_namespace") or "workspace_evaluation"),
        )

    if job is not None:
        job.append_log(
            "system",
            f"evaluate mode={request.mode} output_dir={request.output_dir} metrics={','.join(request.metrics)}\n",
        )
    result = run_evaluate(request)
    payload_dict = result.to_dict()
    payload_dict["request"] = _safe_jsonable(asdict(request))
    return payload_dict


def _job_result_payload(result: Any) -> dict[str, Any] | None:
    if isinstance(result, RunRecord):
        return {
            "run_id": result.run_id,
            "output_dir": result.output_dir,
            "manifest_path": result.manifest_path,
            "preview_video": result.preview_video,
            "preview_image": result.preview_image,
            "preview_model": result.preview_model,
            "preview_splat": result.preview_splat,
            "gallery": result.gallery,
            "artifacts": result.artifacts,
            "metadata": result.metadata,
        }
    if isinstance(result, dict):
        return dict(result)
    if result is None:
        return None
    return {"value": str(result)}


def _result_output_dir(result: Any) -> str:
    if isinstance(result, RunRecord):
        return result.output_dir
    if isinstance(result, dict):
        return str(result.get("output_dir") or "")
    return ""


def _result_artifact_paths(result: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    def add_path(name: str, path: Any) -> None:
        if not path:
            return
        text = str(path)
        if text in seen:
            return
        seen.add(text)
        rows.append((name or Path(text).name, text))

    if isinstance(result, RunRecord):
        for path in (
            result.manifest_path,
            result.preview_video,
            result.preview_image,
            result.preview_model,
            result.preview_splat,
            result.rrd_path,
            *list(result.artifacts or ()),
        ):
            add_path(Path(str(path)).name if path else "", path)
        return rows

    if isinstance(result, dict):
        for key, value in result.items():
            if not key.endswith(("_path", "_file")):
                continue
            if isinstance(value, (str, Path)) and str(value):
                add_path(key, value)
        output_dir = result.get("output_dir")
        if output_dir:
            add_path("output_dir", output_dir)
    return rows


def _resolve_under_output_dir(path_text: str, output_dir: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text).expanduser()
    if path.is_absolute() and path.exists():
        return str(path.resolve())
    if output_dir:
        candidate = Path(output_dir).expanduser() / path
        if candidate.exists():
            return str(candidate.resolve())
    if path.exists():
        return str(path.resolve())
    return str(path)


def _result_preview_fields(result: Any) -> tuple[str, str, str]:
    if isinstance(result, RunRecord):
        return (
            str(result.preview_model or ""),
            str(result.preview_splat or ""),
            str(result.rrd_path or ""),
        )
    if isinstance(result, dict):
        return (
            str(result.get("preview_model") or ""),
            str(result.get("preview_splat") or ""),
            str(result.get("rrd_path") or ""),
        )
    return ("", "", "")


def _primary_visualization_path_by_mode(result: Any) -> dict[str, str]:
    """Pick one canonical artifact per visualizer mode for gallery/detail actions."""

    artifact_paths = [path for _, path in _result_artifact_paths(result)]
    output_dir = _result_output_dir(result)
    preview_model, preview_splat, rrd_path = _result_preview_fields(result)
    selected: dict[str, str] = {}

    if rrd_path and _visualizer_mode_for_artifact(rrd_path) == "rerun":
        selected["rerun"] = rrd_path
    else:
        for path in artifact_paths:
            if _visualizer_mode_for_artifact(path) == "rerun":
                selected["rerun"] = path
                break

    splat_path = ""
    if preview_splat and _visualizer_mode_for_artifact(preview_splat) == "spark":
        splat_path = preview_splat
    if not splat_path:
        guessed_path, _ = first_splat_asset(artifact_paths, gs_ply_predicate=_is_gaussian_splat_ply)
        splat_path = guessed_path or ""
    if splat_path:
        selected["spark"] = splat_path

    points_path = ""
    if preview_model and _visualizer_mode_for_artifact(preview_model) == "points":
        points_path = preview_model
    if not points_path and output_dir:
        geometry_rel = first_geometry_point_candidate(
            artifact_paths,
            output_dir,
            gs_ply_predicate=_is_gaussian_splat_ply,
        )
        if geometry_rel:
            points_path = _resolve_under_output_dir(geometry_rel, output_dir)
    if not points_path:
        for path in artifact_paths:
            if _visualizer_mode_for_artifact(path) == "points":
                points_path = path
                break
    if points_path:
        selected["points"] = points_path

    return selected


def _result_visualization_actions(result: Any, *, model_id: str = "") -> list[dict[str, Any]]:
    output_dir = _result_output_dir(result)
    resolved_model_id = model_id
    if isinstance(result, RunRecord):
        resolved_model_id = result.model_id
    selected = _primary_visualization_path_by_mode(result)
    actions: list[dict[str, Any]] = []
    for mode in ("rerun", "points", "spark"):
        path = selected.get(mode)
        if not path:
            continue
        action = _artifact_visualization_action(
            Path(path).name,
            path,
            model_id=resolved_model_id,
            output_dir=output_dir,
        )
        if action is not None:
            actions.append(action)
    return actions


def _active_run_ids() -> set[str]:
    rows: set[str] = set()
    for job in JOBS.list():
        if isinstance(job.result, RunRecord):
            rows.add(job.result.run_id)
    return rows


def _recent_persisted_runs(limit: int = 100) -> list[RunRecord]:
    active = _active_run_ids()
    return [record for record in MANAGER.list_recent_runs(limit=limit) if record.run_id not in active]


def _registered_artifact_paths() -> set[Path]:
    paths: set[Path] = set()
    for job in JOBS.list():
        for _, path in _result_artifact_paths(job.result):
            try:
                paths.add(Path(path).expanduser().resolve())
            except (OSError, RuntimeError):
                continue
    for record in _recent_persisted_runs():
        for _, path in _result_artifact_paths(record):
            try:
                paths.add(Path(path).expanduser().resolve())
            except (OSError, RuntimeError):
                continue
    return paths


def _gallery_row_from_job(job: StudioJob) -> dict[str, Any] | None:
    if not isinstance(job.result, RunRecord):
        return None
    return {
        "job_id": job.job_id,
        "run_id": job.result.run_id,
        "title": job.title,
        "model_name": job.display_name,
        "model_id": job.model_id,
        "prompt": dict(job.metadata).get("prompt", ""),
        "video_url": f"/api/jobs/{job.job_id}/video" if job.result.preview_video else "",
        "image_url": f"/api/jobs/{job.job_id}/image" if job.result.preview_image else "",
        "model_url": f"/api/jobs/{job.job_id}/model" if job.result.preview_model else "",
        "output_dir": job.result.output_dir,
        "visualization_actions": _result_visualization_actions(job.result, model_id=job.model_id),
    }


def _gallery_row_from_run(record: RunRecord) -> dict[str, Any]:
    metadata = dict(record.metadata or {})
    request = metadata.get("request")
    prompt = request.get("prompt", "") if isinstance(request, dict) else ""
    return {
        "job_id": "",
        "run_id": record.run_id,
        "title": record.display_name or record.model_id or record.run_id,
        "model_name": record.display_name or record.model_id,
        "model_id": record.model_id,
        "prompt": prompt,
        "video_url": f"/api/runs/{record.run_id}/video" if record.preview_video else "",
        "image_url": f"/api/runs/{record.run_id}/image" if record.preview_image else "",
        "model_url": f"/api/runs/{record.run_id}/model" if record.preview_model else "",
        "output_dir": record.output_dir,
        "visualization_actions": _result_visualization_actions(record, model_id=record.model_id),
    }


def _job_payload(job: StudioJob, *, include_logs: bool = False) -> dict[str, Any]:
    return {
        "id": job.job_id,
        "job_id": job.job_id,
        "title": job.title,
        "job_type": job.job_type,
        "model_id": job.model_id,
        "model_name": job.display_name,
        "action": job.action,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "elapsed": format_elapsed(job),
        "error": job.error,
        "metadata": dict(job.metadata),
        "result": _job_result_payload(job.result),
        "visualization_actions": _result_visualization_actions(job.result, model_id=job.model_id),
        "output_dir": _result_output_dir(job.result) or str(dict(job.metadata).get("output_dir") or ""),
        "logs": job.logs[-200:] if include_logs else [],
    }


def _run_payload(record: RunRecord) -> dict[str, Any]:
    result = _job_result_payload(record) or {}
    return {
        "id": record.run_id,
        "run_id": record.run_id,
        "job_id": "",
        "title": record.display_name or record.model_id or record.run_id,
        "job_type": "inference",
        "model_id": record.model_id,
        "model_name": record.display_name or record.model_id,
        "status": "completed",
        "metadata": dict(record.metadata or {}),
        "result": result,
        "visualization_actions": _result_visualization_actions(record, model_id=record.model_id),
        "output_dir": record.output_dir,
    }


def _file_media_type(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _iter_file_range(path: Path, start: int, end: int):
    chunk_size = 1024 * 1024
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _range_response(path: Path, range_header: str | None, media_type: str, headers: dict[str, str]) -> Response | None:
    if not range_header:
        return None
    unit, _, raw_range = range_header.partition("=")
    if unit.strip().lower() != "bytes" or "," in raw_range:
        return None
    file_size = path.stat().st_size
    start_text, _, end_text = raw_range.strip().partition("-")
    try:
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        else:
            suffix_size = int(end_text)
            start = max(file_size - suffix_size, 0)
            end = file_size - 1
    except ValueError:
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{file_size}"})
    if start < 0 or end < start or start >= file_size:
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{file_size}"})
    end = min(end, file_size - 1)
    content_length = end - start + 1
    return StreamingResponse(
        _iter_file_range(path, start, end),
        status_code=206,
        media_type=media_type,
        headers={
            **headers,
            "Content-Length": str(content_length),
            "Content-Range": f"bytes {start}-{end}/{file_size}",
        },
    )


def _safe_file_response(path_text: str | None, request: Request | None = None) -> Response:
    if not path_text:
        raise HTTPException(status_code=404, detail="file not found")
    path = Path(path_text).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    workspace_root = Path(MANAGER.workspace_root).resolve()
    registered = path in _registered_artifact_paths()
    if not registered:
        try:
            path.relative_to(workspace_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="file is outside Studio workspace") from exc
    media_type = _file_media_type(path)
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=86400",
        "X-Content-Type-Options": "nosniff",
    }
    range_response = _range_response(path, request.headers.get("range") if request else None, media_type, headers)
    if range_response is not None:
        return range_response
    return FileResponse(path, media_type=media_type, headers=headers)


def create_app() -> FastAPI:
    _load_settings_from_disk()
    app = FastAPI(title="OpenEnvision Workspace")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return WORKSPACE_HTML

    @app.get("/favicon.ico")
    def favicon() -> FileResponse:
        return FileResponse(OPENENVISION_LOGO_PATH, media_type="image/png")

    @app.get("/assets/openenvision-logo.png")
    def openenvision_logo() -> FileResponse:
        return FileResponse(OPENENVISION_LOGO_PATH, media_type="image/png")

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return dict(SETTINGS)

    @app.post("/api/settings")
    def update_settings(payload: SettingsUpdateRequest) -> dict[str, Any]:
        SETTINGS.update({key: _coerce_setting_value(key, value) for key, value in payload.values.items()})
        _save_settings_to_disk()
        return dict(SETTINGS)

    @app.get("/api/models")
    def list_models(workload_type: str | None = None) -> list[dict[str, Any]]:
        models = [dict(model) for model in _workspace_models()]
        if workload_type and workload_type != "all":
            models = [model for model in models if model["workload_type"] == workload_type or workload_type == "inference"]
        return models

    @app.get("/api/evaluation/catalog")
    def evaluation_catalog() -> dict[str, Any]:
        return _evaluation_catalog_payload()

    @app.post("/api/evaluation/prepare")
    def prepare_evaluation_request(payload: JobCreateRequest) -> dict[str, Any]:
        output_dir = _optional_path(payload.output_dir) or Path(MANAGER.workspace_root) / "evaluations" / "prepared"
        try:
            return _prepared_evaluation_from_payload(payload, output_dir).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/evaluation/vbench/dimensions")
    def evaluation_vbench_dimensions() -> dict[str, Any]:
        return workspace_benchmark_runtime_hint("vbench")

    @app.get("/api/evaluation/benchmarks/{benchmark_id}/runtime")
    def evaluation_benchmark_runtime(benchmark_id: str) -> dict[str, Any]:
        hint = workspace_benchmark_runtime_hint(benchmark_id)
        if not hint:
            raise HTTPException(status_code=404, detail=f"no Workspace runtime hint for benchmark: {benchmark_id}")
        return hint

    @app.get("/api/visualizers")
    def list_visualizers() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for mode in sorted(STUDIO_VISUALIZATIONS.modes):
            if mode in WORKSPACE_HIDDEN_VISUALIZER_MODES:
                continue
            _cleanup_finished_visualizer(mode)
            backend = STUDIO_VISUALIZATIONS.backend_for(mode)
            running = VISUALIZER_MANAGED.get(mode)
            rows.append(
                {
                    "mode": backend.mode,
                    "title": backend.title,
                    "default_port": backend.default_port,
                    "default_model": DEFAULT_VISUALIZER_MODELS.get(mode, ""),
                    "aliases": list(backend.aliases),
                    "native": backend.native,
                    "capabilities": sorted(backend.capabilities.layer_kinds),
                    "requires_asset": mode in VISUALIZER_ASSET_REQUIRED,
                    "requires_url": mode in VISUALIZER_URL_REQUIRED,
                    "accepts_external_url": mode in {"embodied", "rerun"},
                    "status": _visualizer_status(running) if running else None,
                }
            )
        return rows

    @app.post("/api/visualizers/{mode}/launch")
    def launch_visualizer(mode: str, payload: VisualizerLaunchRequest) -> dict[str, Any]:
        return _launch_visualizer(mode.strip().lower(), payload)

    @app.post("/api/visualizers/{mode}/stop")
    def stop_visualizer(mode: str) -> dict[str, Any]:
        mode = mode.strip().lower()
        ok = _stop_visualizer(mode)
        return {"ok": ok, "mode": mode}

    @app.get("/api/jobs")
    def list_jobs(job_type: str | None = None) -> list[dict[str, Any]]:
        jobs = JOBS.list()
        if job_type and job_type != "all":
            jobs = [job for job in jobs if job.job_type == job_type]
        return [_job_payload(job) for job in jobs]

    @app.post("/api/jobs")
    def create_job(payload: JobCreateRequest) -> dict[str, Any]:
        payload.job_type = (payload.job_type or "inference").strip().lower()
        if payload.job_type not in SUPPORTED_WORKSPACE_JOB_TYPES:
            raise HTTPException(status_code=400, detail=f"unsupported job type: {payload.job_type}")

        if payload.job_type == "inference":
            try:
                entry = find_entry(payload.model_id)
            except KeyError as exc:
                raise HTTPException(status_code=400, detail=f"unknown model id: {payload.model_id}") from exc
            if entry.model_id not in _workspace_model_ids():
                raise HTTPException(
                    status_code=400,
                    detail=f"{entry.model_id} is not available in the Workspace inference catalog",
                )
            variant, task, _, _, _, contract = _resolve_inference_contract(entry, payload)
            _validate_inference_payload(entry, task, payload)

            def run_callable(job: StudioJob) -> Any:
                job.append_log(
                    "system",
                    f"model={entry.model_id} variant={variant.variant_id} task={task.task_id} type={payload.job_type}\n",
                )
                result = _run_inference(payload, job)
                job.append_log("system", "job finished\n")
                return result

            title = f"{entry.display_name} {variant.label} {payload.job_type}"
            metadata = {
                "job_type": payload.job_type,
                "workload_type": payload.workload_type,
                "variant_id": variant.variant_id,
                "task_profile_id": task.task_id,
                "infer_contract": contract,
                "prompt": payload.prompt,
                "input_path": payload.input_path,
                "device": payload.device or SETTINGS.get("device"),
            }
            model_id = entry.model_id
            display_name = entry.display_name
        elif payload.job_type == "evaluation":
            mode = (payload.eval_mode or "existing-results").strip().lower().replace("_", "-")
            supported_modes = {"existing-results", "model"} | _PREPARED_EVALUATION_MODES
            if mode not in supported_modes:
                raise HTTPException(status_code=400, detail=f"unsupported evaluation mode: {payload.eval_mode}")
            if mode in _PREPARED_EVALUATION_MODES:
                output_dir = _optional_path(payload.output_dir) or _workspace_job_output_dir("evaluations", None)
                prepared = _prepared_evaluation_from_payload(payload, output_dir)
                if not prepared.ready:
                    messages = "; ".join(issue.message for issue in prepared.issues if issue.severity == "error")
                    raise HTTPException(status_code=400, detail=messages or "evaluation preflight failed")
            elif not payload.run_plan_path:
                if mode == "existing-results" and not payload.results_path and not (
                    workspace_benchmark_supported(payload.benchmark_id) and workspace_benchmark_has_input(payload)
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="existing-results evaluation requires results_path or a benchmark-specific input path",
                    )
                if mode == "model" and not payload.requests_path:
                    raise HTTPException(status_code=400, detail="model evaluation requires requests_path")

            def run_callable(job: StudioJob) -> Any:
                job.append_log("system", f"evaluation mode={mode} model={payload.model_id or 'materialized-results'}\n")
                result = _run_evaluation(payload, job)
                job.append_log("system", "job finished\n")
                return result

            title = f"Evaluation {payload.benchmark_id or mode}"
            metadata = {
                "job_type": payload.job_type,
                "eval_mode": mode,
                "benchmark_id": payload.benchmark_id,
                "model_id": payload.model_id,
                "requests_path": payload.requests_path,
                "results_path": payload.results_path,
                "output_dir": payload.output_dir,
                "metrics": list(payload.metrics),
            }
            model_id = payload.model_id or "materialized-results"
            display_name = payload.model_id or "Evaluation"

        job = JOBS.submit_run(
            title=title,
            model_id=model_id,
            display_name=display_name,
            action=payload.job_type,
            job_type=payload.job_type,
            metadata=metadata,
            run_callable=run_callable,
        )
        return _job_payload(job, include_logs=True)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return _job_payload(job, include_logs=True)

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict[str, Any]:
        ok, message = JOBS.cancel(job_id)
        job = JOBS.get(job_id)
        return {"ok": ok, "message": message, "job": _job_payload(job, include_logs=True) if job else None}

    @app.get("/api/jobs/{job_id}/logs")
    def get_job_logs(job_id: str, after: int = 0) -> dict[str, Any]:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        logs = job.logs[max(after, 0) :]
        return {"offset": len(job.logs), "logs": logs, "text": "".join(str(row.get("text") or "") for row in logs)}

    @app.get("/api/jobs/{job_id}/video")
    def get_job_video(job_id: str, request: Request) -> Response:
        job = JOBS.get(job_id)
        if job is None or not isinstance(job.result, RunRecord):
            raise HTTPException(status_code=404, detail="video not found")
        return _safe_file_response(job.result.preview_video, request=request)

    @app.get("/api/jobs/{job_id}/image")
    def get_job_image(job_id: str, request: Request) -> Response:
        job = JOBS.get(job_id)
        if job is None or not isinstance(job.result, RunRecord):
            raise HTTPException(status_code=404, detail="image not found")
        return _safe_file_response(job.result.preview_image, request=request)

    @app.get("/api/jobs/{job_id}/model")
    def get_job_model(job_id: str, request: Request) -> Response:
        job = JOBS.get(job_id)
        if job is None or not isinstance(job.result, RunRecord):
            raise HTTPException(status_code=404, detail="model not found")
        return _safe_file_response(job.result.preview_model, request=request)

    @app.get("/api/runs/{run_id}/video")
    def get_run_video(run_id: str, request: Request) -> Response:
        try:
            record = MANAGER.load_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return _safe_file_response(record.preview_video, request=request)

    @app.get("/api/runs/{run_id}/image")
    def get_run_image(run_id: str, request: Request) -> Response:
        try:
            record = MANAGER.load_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return _safe_file_response(record.preview_image, request=request)

    @app.get("/api/runs/{run_id}/model")
    def get_run_model(run_id: str, request: Request) -> Response:
        try:
            record = MANAGER.load_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return _safe_file_response(record.preview_model, request=request)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            record = MANAGER.load_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return _run_payload(record)

    @app.get("/api/gallery")
    def gallery() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for job in JOBS.list():
            row = _gallery_row_from_job(job)
            if row is not None:
                rows.append(row)
        rows.extend(_gallery_row_from_run(record) for record in _recent_persisted_runs())
        return rows

    @app.get("/api/artifacts/file")
    def artifact_file(path: str, request: Request) -> Response:
        target = Path(path).expanduser().resolve()
        if target not in _registered_artifact_paths():
            raise HTTPException(status_code=404, detail="artifact is not registered in this workspace session")
        return _safe_file_response(str(target), request=request)

    @app.get("/api/artifacts")
    def artifacts() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for job in JOBS.list():
            for name, path in _result_artifact_paths(job.result):
                visualization = _artifact_visualization_action(
                    name,
                    path,
                    model_id=job.model_id,
                    output_dir=_result_output_dir(job.result),
                )
                rows.append(
                    {
                        "job_id": job.job_id,
                        "model_name": job.display_name,
                        "model_id": job.model_id,
                        "job_type": job.job_type,
                        "name": name,
                        "path": path,
                        "output_dir": _result_output_dir(job.result),
                        "visualizer_mode": visualization["mode"] if visualization else "",
                        "visualizer_label": visualization["label"] if visualization else "",
                    }
                )
        for record in _recent_persisted_runs():
            for name, path in _result_artifact_paths(record):
                visualization = _artifact_visualization_action(
                    name,
                    path,
                    model_id=record.model_id,
                    output_dir=record.output_dir,
                )
                rows.append(
                    {
                        "job_id": "",
                        "run_id": record.run_id,
                        "model_name": record.display_name,
                        "model_id": record.model_id,
                        "job_type": "inference",
                        "name": name,
                        "path": path,
                        "output_dir": record.output_dir,
                        "visualizer_mode": visualization["mode"] if visualization else "",
                        "visualizer_label": visualization["label"] if visualization else "",
                    }
                )
        return rows

    return app


WORKSPACE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenEnvision Workspace</title>
  <link rel="icon" href="/assets/openenvision-logo.png" type="image/png" />
  <style>
    :root {
      color-scheme: dark;
      --bg: #0d0d0f;
      --panel: #161719;
      --panel-2: #1e1f22;
      --line: #2b2d31;
      --line-hover: #3a3d42;
      --text: #ecedef;
      --muted: #8b929a;
      --accent: #10b981;
      --accent-hover: #059669;
      --accent-2: #f59e0b;
      --danger: #ef4444;
      --danger-hover: #dc2626;
      --warn: #f59e0b;
      --radius: 12px;
      --radius-sm: 8px;
      font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #3a3d42; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #4b5057; }
    .app {
      display: grid;
      grid-template-columns: 240px minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: #111214;
      padding: 20px 16px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 4px 8px 16px;
      border-bottom: 1px solid var(--line);
    }
    .brandLogo {
      width: 40px;
      height: 40px;
      border-radius: var(--radius-sm);
      background: #fff;
      object-fit: contain;
      padding: 2px;
      border: 1px solid rgba(255,255,255,0.08);
    }
    .brand strong { display: block; font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
    .brand span { display: block; color: var(--muted); font-size: 12px; margin-top: 2px; }
    .navGroup { display: grid; gap: 4px; }
    .navLabel {
      padding: 6px 12px;
      color: #6b7280;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .navBtn {
      width: 100%;
      border: 0;
      background: transparent;
      color: var(--muted);
      padding: 8px 12px;
      border-radius: var(--radius-sm);
      text-align: left;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      transition: all 0.15s ease;
    }
    .navBtn.active {
      background: #232529;
      color: #fff;
    }
    .navBtn:hover:not(.active) {
      background: #1a1c1e;
      color: var(--text);
    }
    .main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .topbar {
      height: 64px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      background: rgba(13, 13, 15, 0.8);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .topbar h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: -0.01em; }
    .topbarRight { display: flex; align-items: center; gap: 16px; }
    .statusDot {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 500;
    }
    .statusDot:before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
    }
    button, input, textarea, select {
      font: inherit;
    }
    .btn {
      border: 1px solid var(--line);
      background: #232529;
      color: var(--text);
      border-radius: var(--radius-sm);
      padding: 8px 14px;
      font-weight: 500;
      font-size: 13px;
      cursor: pointer;
      min-height: 36px;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }
    .btn:hover { border-color: var(--line-hover); background: #2b2d31; }
    .btn.primary {
      border-color: transparent;
      background: var(--accent);
      color: #fff;
      box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .btn.primary:hover { background: var(--accent-hover); }
    .btn.danger {
      border-color: transparent;
      background: var(--danger);
      color: #fff;
    }
    .btn.danger:hover { background: var(--danger-hover); }
    .content {
      padding: 24px 28px 40px;
      overflow: auto;
    }
    .view { display: none; animation: fadeIn 0.2s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .view.active { display: block; }
    .toolbar {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
    }
    .filters {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }
    label {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 500;
    }
    input, textarea, select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: #09090b;
      color: var(--text);
      padding: 8px 12px;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    input:focus, textarea:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.15);
    }
    textarea {
      min-height: 100px;
      resize: vertical;
      line-height: 1.5;
    }
    .split {
      --split-left: 360px;
      --split-handle: 10px;
      display: grid;
      grid-template-columns: var(--split-left) var(--split-handle) minmax(0, 1fr);
      gap: 0;
      align-items: stretch;
      min-width: 0;
    }
    .splitHandle {
      position: relative;
      width: var(--split-handle);
      cursor: col-resize;
      touch-action: none;
      user-select: none;
      border-radius: 999px;
      background: transparent;
      align-self: stretch;
      z-index: 2;
    }
    .splitHandle::before {
      content: "";
      position: absolute;
      top: 12px;
      bottom: 12px;
      left: 50%;
      width: 3px;
      transform: translateX(-50%);
      border-radius: 999px;
      background: var(--line);
      transition: background 0.15s ease, box-shadow 0.15s ease;
    }
    .splitHandle:hover::before,
    .splitHandle:focus-visible::before,
    .split.is-resizing .splitHandle::before {
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.18);
    }
    .splitHandle:focus-visible {
      outline: none;
    }
    .split.is-resizing {
      cursor: col-resize;
      user-select: none;
    }
    .split.is-resizing .panel {
      pointer-events: none;
    }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: var(--radius);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      overflow: hidden;
      min-width: 0;
    }
    .panelHead {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }
    .panelHead strong { font-size: 15px; font-weight: 600; }
    .jobList { display: grid; }
    .jobRow {
      border: 0;
      border-bottom: 1px solid var(--line);
      background: transparent;
      color: inherit;
      text-align: left;
      padding: 16px 20px;
      cursor: pointer;
      display: grid;
      gap: 8px;
      transition: background 0.15s ease;
    }
    .jobRow:last-child { border-bottom: 0; }
    .jobRow:hover, .jobRow.active { background: #1c1e22; }
    .jobRow.active { border-left: 3px solid var(--accent); padding-left: 17px; }
    .jobTitleLine {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .jobTitleLine strong { font-size: 14px; font-weight: 500; }
    .pill {
      display: inline-flex;
      align-items: center;
      background: #232529;
      border-radius: 999px;
      padding: 4px 10px;
      color: var(--text);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      white-space: nowrap;
    }
    .pill.running { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .pill.completed { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }
    .pill.failed, .pill.cancelled { background: rgba(239, 68, 68, 0.15); color: #f87171; }
    .muted { color: var(--muted); }
    .tiny { font-size: 12px; }
    .detail {
      align-content: start;
      display: grid;
      gap: 14px;
      padding: 16px 20px 20px;
      min-height: 0;
    }
    .detailTabs {
      align-items: center;
      align-self: start;
      display: flex;
      flex-wrap: wrap;
      flex: none;
      gap: 8px;
      margin: 0;
    }
    .detailTab {
      appearance: none;
      align-items: center;
      border: 1px solid var(--line);
      background: #111214;
      color: var(--muted);
      border-radius: 999px;
      display: inline-flex;
      flex: none;
      gap: 2px;
      height: auto;
      line-height: 1.2;
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      white-space: nowrap;
    }
    .detailTab:hover { border-color: var(--line-hover); color: var(--text); }
    .detailTab.active {
      background: rgba(16, 185, 129, 0.12);
      border-color: rgba(16, 185, 129, 0.45);
      color: #34d399;
    }
    .detailTabPanel { display: none; gap: 16px; min-width: 0; }
    .detailTabPanel.active { display: grid; align-content: start; }
    .logToolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }
    .logFilter {
      appearance: none;
      border: 1px solid var(--line);
      background: #111214;
      color: var(--muted);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
    }
    .logFilter.active {
      color: var(--text);
      border-color: var(--line-hover);
      background: #1a1c1e;
    }
    .logFilter.system.active { color: #fbbf24; border-color: rgba(251, 191, 36, 0.35); }
    .logFilter.stdout.active { color: #60a5fa; border-color: rgba(96, 165, 250, 0.35); }
    .logFilter.stderr.active { color: #f87171; border-color: rgba(248, 113, 113, 0.35); }
    .logFollow {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-left: auto;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      user-select: none;
    }
    .logFollow input {
      width: 14px;
      height: 14px;
      min-height: 0;
      margin: 0;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .logPanel {
      margin: 0;
      border: 1px solid var(--line);
      background: #09090b;
      border-radius: var(--radius-sm);
      padding: 12px 14px;
      color: #e5e7eb;
      overflow: auto;
      min-height: 50vh;
      max-height: min(72vh, 900px);
      font-family: 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', monospace;
      font-size: 12px;
      line-height: 1.55;
    }
    .logLine {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 8px;
      align-items: start;
      padding: 2px 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .logMeta {
      color: #71717a;
      font-weight: 600;
      white-space: nowrap;
    }
    .logLine.log-stream-system .logMeta { color: #fbbf24; }
    .logLine.log-stream-stdout .logMeta { color: #60a5fa; }
    .logLine.log-stream-stderr .logMeta { color: #f87171; }
    .logText { color: #e5e7eb; }
    .logBadge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 18px;
      height: 18px;
      margin-left: 4px;
      padding: 0 5px;
      border-radius: 999px;
      background: #232529;
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
    }
    .detailTab.active .logBadge { background: rgba(16, 185, 129, 0.18); color: #6ee7b7; }
    .detailGrid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      background: #111214;
      border-radius: var(--radius-sm);
      padding: 14px;
      min-height: 64px;
    }
    .metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; font-weight: 500; }
    .metric strong { font-size: 14px; font-weight: 500; overflow-wrap: anywhere; }
    .metric-wide { grid-column: 1 / -1; }
    .pathMetric { min-height: 0; }
    .pathRow {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    .pathBasename {
      font-size: 14px;
      font-weight: 600;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      min-width: 0;
      flex: 1;
    }
    .pathParent {
      margin-top: 6px;
      font-family: 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', monospace;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .btn.tiny {
      min-height: 28px;
      padding: 4px 10px;
      font-size: 11px;
      flex-shrink: 0;
    }
    .artifactLinks {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .artifactLink {
      appearance: none;
      border: 1px solid var(--line);
      background: #111214;
      color: var(--text);
      border-radius: var(--radius-sm);
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 500;
      text-decoration: none;
      max-width: 100%;
      overflow-wrap: anywhere;
      cursor: pointer;
    }
    .artifactLink:hover { border-color: var(--line-hover); background: #1a1c1e; }
    pre {
      margin: 0;
      border: 1px solid var(--line);
      background: #09090b;
      border-radius: var(--radius-sm);
      padding: 16px;
      color: #e5e7eb;
      overflow: auto;
      min-height: 240px;
      max-height: min(72vh, 900px);
      font-family: 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', monospace;
      font-size: 12px;
      line-height: 1.6;
    }
    .gridCards {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }
    .itemCard {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: var(--radius);
      padding: 16px;
      display: grid;
      gap: 10px;
      min-height: 160px;
      min-width: 0;
      overflow: hidden;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .itemCard:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
      border-color: var(--line-hover);
    }
    .itemCard strong { font-size: 15px; font-weight: 600; overflow-wrap: anywhere; }
    .itemCard p {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .visualizerCard {
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 100%;
      height: 100%;
    }
    #visualizerGrid.gridCards {
      align-items: stretch;
    }
    .visualizerHead {
      display: grid;
      gap: 8px;
    }
    .visualizerHead strong {
      min-height: 2.5em;
      line-height: 1.25;
    }
    .visualizerBadges {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
      min-height: 28px;
    }
    .visualizerTags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-content: flex-start;
      min-height: 112px;
    }
    .visualizerAliases {
      margin: 0;
      min-height: 2.8em;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }
    .visualizerControls {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      min-height: 214px;
      align-content: start;
    }
    .visualizerControls label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }
    .visualizerControls .wide { grid-column: 1 / -1; }
    .visualizerControls textarea {
      min-height: 118px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
    }
    .visualizerFieldReserved {
      opacity: 0;
      pointer-events: none;
      user-select: none;
    }
    .visualizerFooter {
      margin-top: auto;
      display: grid;
      gap: 10px;
    }
    .visualizerActions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .visualizerStatus {
      border: 1px solid var(--line);
      background: #111214;
      border-radius: var(--radius-sm);
      padding: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .visualizerPreviewPanel {
      margin-top: 16px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: var(--shadow-md);
    }
    .visualizerPreviewPanel.hidden { display: none; }
    .visualizerPreviewPanel .panelHead {
      flex-wrap: wrap;
      gap: 12px;
    }
    .visualizerPreviewTabs {
      display: flex;
      flex: 1 1 auto;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }
    .visualizerPreviewTab {
      appearance: none;
      border: 1px solid var(--line);
      background: #111214;
      color: var(--text);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
    }
    .visualizerPreviewTab:hover { border-color: var(--line-hover); background: #1a1c1e; }
    .visualizerPreviewTab.active {
      border-color: var(--accent);
      background: rgba(16, 185, 129, 0.12);
      color: #6ee7b7;
    }
    .visualizerPreviewFrame {
      width: 100%;
      height: min(68vh, 720px);
      min-height: 420px;
      border: 0;
      display: block;
      background: #050505;
    }
    .mediaBox {
      border: 1px solid var(--line);
      background: #09090b;
      border-radius: var(--radius-sm);
      aspect-ratio: 16 / 9;
      width: 100%;
      max-width: 100%;
      min-width: 0;
      min-height: 180px;
      display: grid;
      grid-template: 1fr / 1fr;
      overflow: hidden;
      position: relative;
      contain: layout paint style;
      isolation: isolate;
    }
    .mediaBox > * {
      grid-area: 1 / 1;
    }
    .mediaBox video, .mediaBox img {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #09090b;
    }
    .mediaBox video:not([src]):not([poster]) { visibility: hidden; }
    .mediaLoad {
      appearance: none;
      border: 1px solid rgba(255,255,255,0.18);
      background: rgba(255,255,255,0.08);
      color: #f8fafc;
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      backdrop-filter: blur(10px);
      place-self: center;
      position: relative;
      z-index: 2;
    }
    .mediaLoad:hover { background: rgba(255,255,255,0.14); border-color: rgba(255,255,255,0.28); }
    .mediaHint {
      position: absolute;
      right: 10px;
      bottom: 10px;
      color: var(--muted);
      font-size: 11px;
      background: rgba(0,0,0,0.45);
      border-radius: 999px;
      padding: 4px 8px;
      z-index: 2;
      pointer-events: none;
    }
    .media-loaded .mediaLoad, .media-loaded .mediaHint { display: none; }
    dialog {
      width: min(800px, calc(100vw - 32px));
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      color: var(--text);
      padding: 0;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    dialog::backdrop { background: rgba(0,0,0,0.7); backdrop-filter: blur(4px); }
    .modalHead, .modalFoot {
      padding: 16px 24px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--panel-2);
    }
    .modalFoot { border-top: 1px solid var(--line); border-bottom: 0; justify-content: flex-end; gap: 12px; }
    .modalBody { padding: 24px; display: grid; gap: 16px; max-height: calc(100vh - 160px); overflow-y: auto; }
    .formGrid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .formGrid .wide { grid-column: 1 / -1; }
    .dynamicGrid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .dynamicGrid .wide { grid-column: 1 / -1; }
    .checks {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 8px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 10px;
      border: 1px solid var(--line);
      background: #111214;
      border-radius: var(--radius-sm);
      padding: 10px 14px;
      color: var(--text);
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .check:hover { background: #1a1c1e; border-color: var(--line-hover); }
    .check input { width: 16px; height: 16px; min-height: 0; margin: 0; cursor: pointer; accent-color: var(--accent); }
    .hidden { display: none !important; }
    @media (max-width: 920px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { position: sticky; top: 0; z-index: 20; flex-direction: row; overflow-x: auto; border-right: 0; border-bottom: 1px solid var(--line); padding: 12px 16px; gap: 12px; align-items: center; }
      .brand { border-bottom: 0; min-width: max-content; padding: 0 12px 0 0; }
      .navGroup { display: flex; align-items: center; gap: 8px; }
      .navLabel { display: none; }
      .navBtn { white-space: nowrap; }
      .split { grid-template-columns: 1fr; gap: 16px; }
      .splitHandle { display: none; }
      .formGrid, .checks, .visualizerControls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand"><img class="brandLogo" src="/assets/openenvision-logo.png" alt="" /><div><strong>WorldFoundry Workspace</strong><span>OpenEnvision</span></div></div>
      <div class="navGroup">
        <div class="navLabel">Jobs</div>
        <button class="navBtn active" data-view="inference">Inference</button>
        <button class="navBtn" data-view="evaluation">Evaluation</button>
      </div>
      <div class="navGroup">
        <div class="navLabel">Assets</div>
        <button class="navBtn" data-view="catalog">Catalog</button>
        <button class="navBtn" data-view="gallery">Gallery</button>
        <button class="navBtn" data-view="artifacts">Artifacts</button>
        <button class="navBtn" data-view="visualizers">Visualizers</button>
      </div>
    </aside>
    <main class="main">
      <header class="topbar">
        <h1 id="pageTitle">Inference</h1>
        <div class="topbarRight">
          <span class="statusDot" id="serverState">API connected</span>
          <button class="btn primary" id="openCreate">Create Job</button>
        </div>
      </header>
      <div class="content">
        <section id="view-inference" class="view active">
          <div class="toolbar"><div class="filters"><label>Type<select id="jobTypeFilter"><option value="all">All jobs</option><option value="inference">Inference</option><option value="evaluation">Evaluation</option></select></label><label>Status<select id="statusFilter"><option value="all">All status</option><option>queued</option><option>running</option><option>completed</option><option>failed</option><option>cancelled</option></select></label></div><button class="btn" id="refreshJobs">Refresh</button></div>
          <div class="split" id="jobSplit" data-split-key="worldfoundry.studio.jobSplitLeft" data-split-default="360" data-split-min-left="240" data-split-min-right="360"><div class="panel"><div class="panelHead"><strong>Job Queue</strong><span class="muted tiny" id="jobCount">0 jobs</span></div><div class="jobList" id="jobList"></div></div><div class="splitHandle" role="separator" aria-orientation="vertical" aria-label="Resize job queue and detail" title="Drag to resize · double-click to reset" tabindex="0"></div><div class="panel"><div class="panelHead"><strong>Job Detail</strong><button class="btn danger" id="stopJob">Stop</button></div><div class="detail" id="jobDetail"><span class="muted">Select a job.</span></div></div></div>
        </section>
        <section id="view-catalog" class="view"><div class="toolbar"><div class="filters"><label>Search<input id="catalogSearch" placeholder="model, tag, family" /></label><label>Workload<select id="catalogWorkload"><option value="all">All</option><option value="t2v">T2V</option><option value="i2v">I2V</option><option value="v2v">V2V</option><option value="3d">3D</option><option value="geometry">Geometry</option><option value="action">Action</option><option value="api">API</option><option value="world">World</option></select></label></div></div><div class="gridCards" id="catalogGrid"></div></section>
        <section id="view-gallery" class="view"><div class="gridCards" id="galleryGrid"></div></section>
        <section id="view-artifacts" class="view"><div class="panel"><div class="panelHead"><strong>Artifacts</strong></div><div class="detail" id="artifactList"></div></div></section>
        <section id="view-visualizers" class="view">
          <div class="toolbar"><span class="muted tiny" id="visualizerLiveState">Live sync idle</span><button class="btn" id="refreshVisualizers">Refresh</button></div>
          <div class="gridCards" id="visualizerGrid"></div>
          <div class="visualizerPreviewPanel hidden" id="visualizerPreviewPanel">
            <div class="panelHead">
              <strong>Live Preview</strong>
              <div class="visualizerPreviewTabs" id="visualizerPreviewTabs"></div>
              <button class="btn" id="visualizerPreviewPopout" type="button">Open Tab</button>
            </div>
            <iframe class="visualizerPreviewFrame" id="visualizerPreviewFrame" loading="lazy" referrerpolicy="no-referrer" sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads allow-pointer-lock" allow="fullscreen; clipboard-read; clipboard-write" title="Visualizer preview"></iframe>
          </div>
        </section>
      </div>
    </main>
  </div>
  <dialog id="createDialog">
    <form method="dialog">
      <div class="modalHead"><strong>Create Job</strong><button class="btn" value="cancel">Close</button></div>
      <div class="modalBody">
        <div class="formGrid">
          <label>Job Type<select id="jobType"><option value="inference">Inference</option><option value="evaluation">Evaluation</option></select></label>
          <label>Workload<select id="workloadType"><option value="t2v">T2V</option><option value="i2v">I2V</option><option value="v2v">V2V</option><option value="3d">3D</option><option value="geometry">Geometry</option><option value="action">Action</option><option value="api">API</option><option value="world">World</option></select></label>
          <label class="wide">Model<select id="modelSelect"></select></label>
          <label>Variant<select id="variantSelect"></select></label>
          <label>Task<select id="taskProfile"></select></label>
          <div class="wide metric" id="inferSpecSummary"><span>Inference Contract</span><strong>Select a model.</strong></div>
          <div class="wide dynamicGrid" id="inferDynamicFields"></div>
          <label class="wide">Prompt<textarea id="prompt" placeholder="Describe the world, scene, or instruction"></textarea></label>
          <label class="wide">Negative Prompt<textarea id="negativePrompt" placeholder="Optional negative prompt"></textarea></label>
          <label class="wide">Input Path<input id="inputPath" placeholder="/path/to/image.mp4, image, dataset, or scene folder" /></label>
          <label>Frames<input id="numFrames" type="number" min="0" /></label>
          <label>FPS<input id="fps" type="number" min="1" /></label>
          <label>Height<input id="height" type="number" min="0" /></label>
          <label>Width<input id="width" type="number" min="0" /></label>
          <label>Steps<input id="steps" type="number" min="1" /></label>
          <label>Guidance<input id="guidance" type="number" step="0.1" /></label>
          <label>Seed<input id="seed" type="number" /></label>
          <label>Device<input id="device" placeholder="cuda" /></label>
          <label>Backend<select id="backend"><option value="auto">auto</option><option value="from_pretrained">from_pretrained</option><option value="api_init">api_init</option></select></label>
          <label>Attention<select id="attention"><option value="auto">auto</option><option value="torch">torch</option><option value="flash_attn_2">flash_attn_2</option><option value="flash_attn_3">flash_attn_3</option><option value="sage">sage</option><option value="xformers">xformers</option></select></label>
          <label class="wide">Model Ref<input id="modelRef" placeholder="checkpoint path or repo id" /></label>
          <label class="wide">Endpoint<input id="endpoint" placeholder="hosted API endpoint" /></label>
          <label class="wide">API Key<input id="apiKey" type="password" autocomplete="off" placeholder="optional hosted API key" /></label>
          <label class="wide">Call JSON<textarea id="callJson">{}</textarea></label>
          <label class="wide">Load JSON<textarea id="loadJson">{}</textarea></label>
          <label>Eval Mode<select id="evalMode"><option value="score-artifacts">Score My Artifacts</option><option value="model-benchmark">Model Benchmark</option><option value="generate-and-score">Generate &amp; Score Dataset</option><option value="existing-results">Existing Results</option><option value="model">Generic Model Runner</option></select></label>
          <label>Eval Preset<select id="evalPreset"></select></label>
          <label>Benchmark<select id="evalBenchmark"></select></label>
          <label class="wide">Results Path<input id="evalResultsPath" placeholder="/path/to/results.jsonl for existing-results mode" /></label>
          <label class="wide">Requests Path<input id="evalRequestsPath" placeholder="/path/to/requests.jsonl for model mode" /></label>
          <label class="wide">Eval Output Dir<input id="evalOutputDir" placeholder="defaults to workspace/evaluations/<job_id>" /></label>
          <label class="wide">Run Plan Path<input id="evalRunPlanPath" placeholder="optional worldfoundry run plan JSON" /></label>
          <label>Eval Model ID<input id="evalModelId" placeholder="model id for model mode" /></label>
          <label>Model Runner<input id="evalModelRunner" placeholder="optional runner id/target" /></label>
          <label>Model Variant<input id="evalModelVariant" placeholder="optional model-zoo variant" /></label>
          <label>Dataset ID<input id="evalDatasetId" placeholder="optional dataset id" /></label>
          <label class="wide">Dataset Root<input id="evalDatasetRoot" placeholder="optional benchmark dataset root" /></label>
          <label class="wide">Dataset Manifest<input id="evalDatasetManifest" placeholder="optional dataset manifest JSON/JSONL" /></label>
          <label>Cache<select id="evalCacheMode"><option value="off">off</option><option value="read">read</option><option value="write">write</option><option value="read-write">read-write</option><option value="refresh">refresh</option></select></label>
          <label class="wide">Metrics<input id="evalMetrics" value="artifact_count" list="evalMetricSuggestions" placeholder="artifact_count, required_artifacts_present" /></label>
          <datalist id="evalMetricSuggestions"></datalist>
          <label class="wide">Required Artifacts<input id="evalRequiredArtifacts" placeholder="generated_video, generated_image" /></label>
          <label class="wide">Eval Model Parameters JSON<textarea id="evalModelParameters">{}</textarea></label>
          <label class="wide">Eval Runtime JSON<textarea id="evalRuntime">{}</textarea></label>
        </div>
        <div class="checks" id="runtimeChecks">
          <label class="check"><input type="checkbox" id="torchCompile" /> Torch Compile</label>
          <label class="check"><input type="checkbox" id="cpuOffload" /> CPU Offload</label>
          <label class="check"><input type="checkbox" id="vaeOffload" /> VAE Offload</label>
          <label class="check"><input type="checkbox" id="textOffload" /> Text Encoder Offload</label>
        </div>
      </div>
      <div class="modalFoot"><button class="btn" value="cancel">Cancel</button><button class="btn primary" id="createJob" value="default">Create Job</button></div>
    </form>
  </dialog>
  <script>
    const state = { view: "inference", jobs: [], models: [], evaluationCatalog: {benchmarks: [], metrics: [], models: []}, visualizers: [], visualizerPreviewMode: "", activeJob: "", detailRenderKey: "", detailTab: "preview", detailLogCount: 0, logStreamFilter: "all", logFollowTail: true, settings: {}, lazyVideoObserver: null, autoVideoPreloads: 0, visualizerRefreshInFlight: false, visualizerLastSync: 0 };
    const MAX_AUTO_VIDEO_PRELOADS = 4;
    const JOB_VIEWS = new Set(["inference", "evaluation"]);
    const INFER_INFRA_FIELDS = ["workloadType","modelSelect","variantSelect","taskProfile","device","backend","attention","modelRef","endpoint","apiKey","callJson","loadJson"];
    const INFER_TASK_FIELD_CONTROLS = {
      prompt: ["prompt"],
      negativePrompt: ["negative_prompt", "negative-prompt"],
      inputPath: ["input_path", "image", "video", "last_frame", "reference_file", "reference_files"],
      numFrames: ["frames", "num_frames", "num-frames"],
      fps: ["fps"],
      height: ["height"],
      width: ["width"],
      steps: ["steps", "num_inference_steps", "num-inference-steps", "sampling_steps", "sampling-steps"],
      guidance: ["guidance", "guidance_scale", "guidance-scale"],
      seed: ["seed"]
    };
    const DEDICATED_INFER_FIELD_IDS = new Set(Object.values(INFER_TASK_FIELD_CONTROLS).flat().map(normalizeFieldId));
    const PARAM_KEY_BY_FIELD_ID = {
      frames: "num_frames",
      "num-frames": "num_frames",
      num_frames: "num_frames",
      steps: "num_inference_steps",
      "num-inference-steps": "num_inference_steps",
      num_inference_steps: "num_inference_steps",
      "sampling-steps": "num_inference_steps",
      sampling_steps: "num_inference_steps",
      guidance: "guidance_scale",
      "guidance-scale": "guidance_scale",
      guidance_scale: "guidance_scale"
    };
    const RUNTIME_CHECK_OPTIONS = {
      torchCompile: "torch_compile",
      cpuOffload: "cpu_offload",
      vaeOffload: "vae_cpu_offload",
      textOffload: "text_encoder_cpu_offload"
    };
    const VIEW_TITLES = {
      inference: "Inference",
      evaluation: "Evaluation",
      catalog: "Catalog",
      gallery: "Gallery",
      artifacts: "Artifacts",
      visualizers: "Visualizers"
    };
    const $ = (id) => document.getElementById(id);
    async function api(path, opts) {
      const res = await fetch(path, opts);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }
    function splitList(value) {
      return String(value || "").split(",").map(item => item.trim()).filter(Boolean);
    }
    function parseJsonField(id, fallback) {
      const text = $(id).value || "";
      if (!text.trim()) return fallback;
      return JSON.parse(text);
    }
    function optionInArgs(args, ...names) {
      return args.some(arg => names.some(name => arg === name || arg.startsWith(`${name}=`)));
    }
    function normalizeFieldId(value) {
      return String(value || "").trim().toLowerCase().replaceAll("_", "-");
    }
    function fieldControlVisible(id) {
      const el = $(id);
      if (!el) return false;
      const wrapper = el.closest("label") || el;
      return !wrapper.classList.contains("hidden");
    }
    function runtimeCheckVisible(id) {
      const el = $(id);
      if (!el) return false;
      const wrapper = el.closest(".check") || el;
      return !wrapper.classList.contains("hidden");
    }
    function inferTaskFields() {
      const task = selectedTaskProfile();
      return task && Array.isArray(task.inputs) ? task.inputs : [];
    }
    function inferTaskFieldIds() {
      return new Set(inferTaskFields().map(field => normalizeFieldId(field.field_id)));
    }
    function coerceInferFieldValue(input) {
      if (input.type === "checkbox") return input.checked;
      const raw = input.value;
      const kind = input.dataset.kind || "string";
      if (raw === "" && input.dataset.required !== "true") return undefined;
      if (kind === "integer") return Number.parseInt(raw || "0", 10);
      if (kind === "number") return Number(raw || "0");
      if (kind === "interaction_tokens") {
        try {
          const parsed = JSON.parse(raw);
          if (Array.isArray(parsed)) return parsed.map(String).filter(Boolean);
          if (typeof parsed === "string") return splitList(parsed);
        } catch {}
        return splitList(raw);
      }
      if (kind === "json") return JSON.parse(raw || "null");
      if (kind === "boolean") return Boolean(input.checked);
      return raw;
    }
    function applyInferValue(target, fieldId, value, payload) {
      if (value === undefined || value === null || value === "") return;
      const key = PARAM_KEY_BY_FIELD_ID[fieldId] || fieldId.replaceAll("-", "_");
      if (target === "prompt") payload.prompt = String(value);
      else if (target === "input_path") payload.input_path = String(value);
      else if (target === "params") payload.params[key] = value;
      else if (target === "load_kwargs") payload.load_kwargs[key] = value;
      else payload.call_kwargs[key] = value;
    }
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      })[ch]);
    }
    function safeClassToken(value) {
      return String(value ?? "").toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
    }
    function optionHtml(value, label, attrs = {}) {
      const extra = Object.entries(attrs)
        .map(([key, attrValue]) => ` ${safeClassToken(key)}="${escapeHtml(attrValue)}"`)
        .join("");
      return `<option value="${escapeHtml(value)}"${extra}>${escapeHtml(label ?? value)}</option>`;
    }
    function pillHtml(value, extraClass = "") {
      return `<span class="pill ${safeClassToken(extraClass || value)}">${escapeHtml(value)}</span>`;
    }
    function renderVideoBox(url, options = {}) {
      const src = escapeHtml(url);
      const poster = options.poster ? ` poster="${escapeHtml(options.poster)}"` : "";
      const eager = !!options.eager;
      return `<div class="mediaBox ${eager ? "media-loaded" : ""}" data-media-box>
        <video controls playsinline controlsList="nodownload" preload="${eager ? "metadata" : "none"}"${poster} data-src="${src}" data-eager="${eager ? "true" : "false"}"></video>
        ${eager ? "" : `<button class="mediaLoad" type="button" data-load-video>Play preview</button><span class="mediaHint">video</span>`}
      </div>`;
    }
    function gallerySubtitle(row) {
      const prompt = String(row.prompt || "").trim();
      if (prompt) return prompt;
      const runId = String(row.run_id || "").trim();
      if (runId) return runId;
      return pathBasename(row.output_dir || "");
    }
    function normalizePathText(path) {
      return String(path || "").trim().replace(/\/+$/, "");
    }
    function pathBasename(path) {
      const cleaned = normalizePathText(path);
      if (!cleaned) return "";
      const parts = cleaned.split("/");
      return parts[parts.length - 1] || cleaned;
    }
    function pathParentText(path) {
      const cleaned = normalizePathText(path);
      const parts = cleaned.split("/");
      if (parts.length <= 1) return cleaned;
      parts.pop();
      return parts.join("/");
    }
    function pathParentDisplay(path) {
      const parent = pathParentText(path);
      if (!parent) return "";
      if (parent.length <= 56) return parent;
      return "…" + parent.slice(-55);
    }
    function renderOutputPath(path) {
      const value = normalizePathText(path);
      if (!value) {
        return `<div class="metric metric-wide pathMetric"><span>Output</span><strong class="muted">pending</strong></div>`;
      }
      const base = pathBasename(value);
      const parent = pathParentDisplay(value);
      return `<div class="metric metric-wide pathMetric">
        <span>Output directory</span>
        <div class="pathRow">
          <strong class="pathBasename" title="${escapeHtml(value)}">${escapeHtml(base)}</strong>
          <button type="button" class="btn tiny" data-copy-path="${escapeHtml(value)}">Copy path</button>
        </div>
        ${parent ? `<div class="pathParent muted tiny" title="${escapeHtml(value)}">${escapeHtml(parent)}</div>` : ""}
      </div>`;
    }
    function renderImageBox(url) {
      return `<div class="mediaBox"><img loading="lazy" decoding="async" src="${escapeHtml(url)}" /></div>`;
    }
    function artifactUrl(path) {
      return `/api/artifacts/file?path=${encodeURIComponent(path || "")}`;
    }
    function artifactLink(label, path) {
      if (!path) return "";
      return `<a class="artifactLink" href="${artifactUrl(path)}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`;
    }
    function externalLink(label, url) {
      if (!url) return "";
      return `<a class="artifactLink" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`;
    }
    function catalogLinksHtml(links) {
      if (!links) return "";
      const parts = [
        externalLink("Project", links.project),
        externalLink("GitHub", links.github),
        externalLink("arXiv", links.paper),
      ].filter(Boolean);
      return parts.length ? `<div class="artifactLinks">${parts.join("")}</div>` : "";
    }
    function artifactVisualizeButton(label, path, mode, modelId = "") {
      if (!path || !mode) return "";
      return `<button class="artifactLink" type="button" data-visualize-artifact="${escapeHtml(mode)}" data-artifact-path="${escapeHtml(path)}" data-artifact-model="${escapeHtml(modelId)}">${escapeHtml(label || "Visualize")}</button>`;
    }
    function artifactRefineButton(jobId = "", runId = "") {
      if (!jobId && !runId) return "";
      return `<button class="artifactLink" type="button" data-refine-job="${escapeHtml(jobId || "")}" data-refine-run="${escapeHtml(runId || "")}">Refine</button>`;
    }
    function refineInputPreference(full) {
      const metadata = (full && full.metadata) || {};
      const contract = metadata.infer_contract || {};
      const task = contract.task || {};
      const fields = Array.isArray(task.inputs) ? task.inputs : [];
      const fieldIds = new Set(fields.map(field => normalizeFieldId(field.field_id)));
      const model = full && full.model_id ? state.models.find(item => item.id === full.model_id) : null;
      const workload = String((model && model.workload_type) || metadata.workload_type || "").toLowerCase();
      const taskText = [
        full && full.model_id,
        task.task_id,
        task.label,
        task.description,
        fields.map(field => [field.field_id, field.label, field.target].join(" ")).join(" ")
      ].join(" ").toLowerCase();
      if (workload === "i2v" || fieldIds.has("image") || fieldIds.has("input-image") || fieldIds.has("reference-image") || fieldIds.has("last-frame")) return "image";
      if (workload === "v2v" || fieldIds.has("video") || fieldIds.has("input-video") || fieldIds.has("video-path") || /\b(video_path|input_video|conditioning_video)\b/.test(taskText)) return "video";
      return "";
    }
    function bestRefineInputPath(result, full = null) {
      if (!result) return "";
      const preference = refineInputPreference(full);
      if (preference === "image") return result.preview_image || result.preview_video || result.preview_model || result.preview_splat || "";
      if (preference === "video") return result.preview_video || result.preview_image || result.preview_model || result.preview_splat || "";
      return result.preview_video || result.preview_image || result.preview_model || result.preview_splat || "";
    }
    function renderResultActions(full, result) {
      const actions = [];
      if (result.preview_video) actions.push(artifactLink("Open Video", result.preview_video));
      if (result.preview_image) actions.push(artifactLink("Open Image", result.preview_image));
      if (result.preview_model) actions.push(artifactLink("Open Model", result.preview_model));
      if (result.preview_splat) actions.push(artifactLink("Open Splat", result.preview_splat));
      if (full.job_type === "inference" && full.status === "completed" && bestRefineInputPath(result, full)) {
        actions.push(artifactRefineButton(full.job_id || full.id || "", full.run_id || ""));
      }
      (full.visualization_actions || []).forEach(action => {
        actions.push(artifactVisualizeButton(action.label, action.path, action.mode, action.model_id || full.model_id));
      });
      const unique = [];
      const seen = new Set();
      actions.filter(Boolean).forEach(html => {
        if (seen.has(html)) return;
        seen.add(html);
        unique.push(html);
      });
      return unique.length ? `<div class="artifactLinks">${unique.join("")}</div>` : "";
    }
    function bindArtifactVisualizerButtons(root = document) {
      root.querySelectorAll("[data-visualize-artifact]").forEach(button => {
        button.onclick = () => visualizeArtifact(
          button.dataset.visualizeArtifact,
          button.dataset.artifactPath,
          button.dataset.artifactModel
        );
      });
      root.querySelectorAll("[data-refine-job], [data-refine-run]").forEach(button => {
        button.onclick = () => refineFromRecord(button.dataset.refineJob || "", button.dataset.refineRun || "");
      });
    }
    function setSelectValue(id, value, label = "") {
      if (!value) return;
      ensureSelectOption(id, value, label || value);
      $(id).value = value;
    }
    function selectInferModel(modelId, variantId, taskId) {
      const model = state.models.find(item => item.id === modelId);
      const workload = (model && model.workload_type) || $("workloadType").value || "world";
      setSelectValue("workloadType", workload);
      populateModelSelect();
      setSelectValue("modelSelect", modelId);
      populateInferSpecControls();
      if (variantId) {
        setSelectValue("variantSelect", variantId);
      }
      if (taskId) {
        setSelectValue("taskProfile", taskId);
      }
      renderInferDynamicFields();
      renderInferSpecSummary();
      applySelectedModelDefaults();
      updateCreateMode();
    }
    function firstDefined(...values) {
      return values.find(value => value !== undefined && value !== null && value !== "");
    }
    function recordRequest(full) {
      const metadata = (full && full.metadata) || {};
      const resultMetadata = (full && full.result && full.result.metadata) || {};
      const request = resultMetadata.request || metadata.request || {};
      return request && typeof request === "object" ? request : {};
    }
    function recordInferContract(full, request) {
      const metadata = (full && full.metadata) || {};
      const resultMetadata = (full && full.result && full.result.metadata) || {};
      const contract = (request && request.infer_contract) || metadata.infer_contract || resultMetadata.infer_contract || {};
      return contract && typeof contract === "object" ? contract : {};
    }
    const REFINE_PATH_KWARG_KEYS = new Set([
      "input_path", "input-dir", "input_dir", "image", "images", "image_path", "image-path",
      "input_image", "input-image", "input_image_path", "input-image-path", "video", "videos",
      "video_path", "video-path", "input_video", "input-video", "input_video_path", "input-video-path",
      "last_frame", "last-frame", "reference_file", "reference-file", "reference_files", "reference-files",
      "validation_image", "validation-image", "validation_images", "validation-images",
      "output", "output_path", "output-path", "output_dir", "output-dir", "out_path", "out-path",
      "out_dir", "out-dir", "outdir", "save_path", "save-path", "save_dir", "save-dir"
    ]);
    function sanitizedRefineKwargs(value) {
      if (Array.isArray(value)) {
        return value.map(item => sanitizedRefineKwargs(item));
      }
      if (!value || typeof value !== "object") {
        return value;
      }
      const clean = {};
      Object.entries(value).forEach(([key, item]) => {
        const normalized = normalizeFieldId(key);
        if (REFINE_PATH_KWARG_KEYS.has(normalized) || REFINE_PATH_KWARG_KEYS.has(key)) return;
        clean[key] = sanitizedRefineKwargs(item);
      });
      return clean;
    }
    function payloadFromRefineRecord(full) {
      const metadata = full.metadata || {};
      const request = recordRequest(full);
      const contract = recordInferContract(full, request);
      const variant = contract.variant || {};
      const task = contract.task || {};
      const result = full.result || {};
      return {
        modelId: full.model_id || contract.model_family_id || metadata.model_id || "",
        variantId: firstDefined(metadata.variant_id, request.variant_id, contract.variant_id, variant.variant_id, "default"),
        taskId: firstDefined(metadata.task_profile_id, request.task_profile_id, contract.task_profile_id, task.task_id, "default"),
        prompt: firstDefined(metadata.prompt, request.prompt, ""),
        negativePrompt: firstDefined(metadata.negative_prompt, request.negative_prompt, ""),
        inputPath: bestRefineInputPath(result, full),
        device: firstDefined(metadata.device, request.device, "cuda"),
        modelRef: firstDefined(request.model_ref, variant.model_ref, metadata.model_ref, ""),
        callKwargs: sanitizedRefineKwargs({...((variant.call_kwargs || {})), ...((request.call_kwargs || {}))}),
        loadKwargs: sanitizedRefineKwargs({...((variant.load_kwargs || {})), ...((request.load_kwargs || {}))})
      };
    }
    async function refineFromRecord(jobId, runId) {
      try {
        const full = jobId
          ? await api(`/api/jobs/${encodeURIComponent(jobId)}`)
          : await api(`/api/runs/${encodeURIComponent(runId)}`);
        const refine = payloadFromRefineRecord(full);
        if (!refine.modelId || !refine.inputPath) throw new Error("Selected record does not contain enough model/output metadata to refine.");
        setView("inference");
        $("jobType").value = "inference";
        selectInferModel(refine.modelId, refine.variantId, refine.taskId);
        setControlDefault("prompt", refine.prompt);
        setControlDefault("negativePrompt", refine.negativePrompt);
        setControlDefault("inputPath", refine.inputPath);
        setControlDefault("device", refine.device);
        if (refine.modelRef) setControlDefault("modelRef", refine.modelRef);
        delete refine.callKwargs.output_path;
        delete refine.callKwargs.output_dir;
        setJsonInputValue("callJson", refine.callKwargs);
        setJsonInputValue("loadJson", refine.loadKwargs);
        $("createDialog").showModal();
      } catch (error) {
        alert(error.message || String(error));
      }
    }
    async function visualizeArtifact(mode, path, modelId) {
      if (!mode || !path) return;
      const payload = {
        model_id: modelId || "",
        asset_path: path,
        host: "127.0.0.1",
        port: null,
        reuse: false,
        params: visualizerDefaultParams(mode)
      };
      setView("visualizers");
      state.visualizers = await api("/api/visualizers");
      renderVisualizers(false);
      updateVisualizerStatus(mode, "Launching...");
      try {
        const status = await api(`/api/visualizers/${encodeURIComponent(mode)}/launch`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        state.visualizers = await api("/api/visualizers");
        const row = state.visualizers.find(item => item.mode === mode);
        if (row) row.status = status;
      } catch (error) {
        updateVisualizerStatus(mode, `Launch failed: ${escapeHtml(error.message || error)}`);
        return;
      }
      renderVisualizers(false);
    }
    function renderEvaluationSummary(result) {
      if (!result || !(result.schema_version === "worldfoundry-evaluate-run-result" || result.scorecard_path)) return "";
      const outputDir = String(result.output_dir || "").replace(/\/+$/, "");
      const reportPath = outputDir ? `${outputDir}/report.md` : "";
      const links = [
        artifactLink("Scorecard", result.scorecard_path),
        artifactLink("Manifest", result.manifest_path),
        artifactLink("Execution Plan", result.execution_plan_path),
        artifactLink("Report", reportPath)
      ].filter(Boolean).join("");
      return `
        <div class="detailGrid">
          <div class="metric"><span>Samples</span><strong>${escapeHtml(result.sample_count ?? "0")}</strong></div>
          <div class="metric"><span>Successful</span><strong>${escapeHtml(result.successful_sample_count ?? "0")}</strong></div>
          <div class="metric"><span>Failed</span><strong>${escapeHtml(result.failed_sample_count ?? "0")}</strong></div>
          <div class="metric"><span>Artifacts</span><strong>${escapeHtml(result.artifact_count ?? "0")}</strong></div>
          <div class="metric"><span>Mode</span><strong>${escapeHtml(result.mode || "")}</strong></div>
          <div class="metric"><span>Runner</span><strong>${escapeHtml(result.delegate_runner || "")}</strong></div>
        </div>
        ${links ? `<div class="artifactLinks">${links}</div>` : ""}`;
    }
    function activateVideo(video, autoplay = false) {
      if (!video || !video.dataset.src) return;
      if (!video.getAttribute("src")) {
        video.setAttribute("src", video.dataset.src);
        video.preload = "metadata";
        video.load();
      }
      const box = video.closest("[data-media-box]");
      if (box) box.classList.add("media-loaded");
      if (autoplay) video.play().catch(() => {});
    }
    function lazyVideoObserver() {
      if (!("IntersectionObserver" in window)) return null;
      if (!state.lazyVideoObserver) {
        state.lazyVideoObserver = new IntersectionObserver(entries => {
          entries.forEach(entry => {
            if (!entry.isIntersecting || state.autoVideoPreloads >= MAX_AUTO_VIDEO_PRELOADS) return;
            state.autoVideoPreloads += 1;
            activateVideo(entry.target, false);
            state.lazyVideoObserver.unobserve(entry.target);
          });
        }, { rootMargin: "360px 0px", threshold: 0.01 });
      }
      return state.lazyVideoObserver;
    }
    function hydrateLazyMedia(root = document) {
      root.querySelectorAll("video[data-src]").forEach(video => {
        const button = video.closest("[data-media-box]")?.querySelector("[data-load-video]");
        if (button) button.onclick = () => activateVideo(video, true);
        if (video.dataset.eager === "true") {
          activateVideo(video, false);
          return;
        }
        const observer = lazyVideoObserver();
        if (observer) observer.observe(video);
      });
    }
    function setFieldVisible(id, visible) {
      const el = $(id);
      if (!el) return;
      const wrapper = el.closest("label") || el;
      wrapper.classList.toggle("hidden", !visible);
    }
    function setRuntimeCheckVisible(id, visible) {
      const el = $(id);
      if (!el) return;
      const wrapper = el.closest(".check") || el;
      wrapper.classList.toggle("hidden", !visible);
    }
    function setView(view) {
      state.view = view;
      document.querySelectorAll(".navBtn").forEach(btn => btn.classList.toggle("active", btn.dataset.view === view));
      const contentView = JOB_VIEWS.has(view) ? "inference" : view;
      document.querySelectorAll(".view").forEach(el => el.classList.toggle("active", el.id === "view-" + contentView));
      if (JOB_VIEWS.has(view)) {
        $("jobTypeFilter").value = view;
        $("jobType").value = view;
        updateCreateMode();
        renderJobs();
      }
      $("pageTitle").textContent = VIEW_TITLES[view] || (view[0].toUpperCase() + view.slice(1));
    }
    function statusClass(status) { return "pill " + safeClassToken(status); }
    function logStreamClass(stream) { return "log-stream-" + safeClassToken(stream || "log"); }
    function logRowsFiltered(logRows) {
      const filter = state.logStreamFilter || "all";
      if (filter === "all") return logRows;
      return logRows.filter(row => String(row.stream || "log") === filter);
    }
    function renderLogLine(row) {
      const stream = String(row.stream || "log");
      return `<div class="logLine ${logStreamClass(stream)}"><span class="logMeta">[${escapeHtml(stream)}]</span><span class="logText">${escapeHtml(row.text || "")}</span></div>`;
    }
    function renderLogPanelHtml(logRows) {
      const rows = logRowsFiltered(logRows);
      if (!rows.length) {
        const filter = state.logStreamFilter || "all";
        if (filter !== "all") return `<span class="muted">No ${escapeHtml(filter)} logs.</span>`;
        return `<span class="muted">No logs yet.</span>`;
      }
      return rows.map(renderLogLine).join("");
    }
    function logPanelNearBottom(panel) {
      if (!panel) return true;
      return panel.scrollHeight - panel.scrollTop - panel.clientHeight < 48;
    }
    function scrollLogPanelToBottom(panel) {
      if (!panel) return;
      panel.scrollTop = panel.scrollHeight;
    }
    function updateLogBadge(count) {
      const badge = $("jobLogBadge");
      if (badge) badge.textContent = String(count || 0);
    }
    function setDetailTab(tab) {
      state.detailTab = tab;
      document.querySelectorAll("[data-detail-tab]").forEach(btn => btn.classList.toggle("active", btn.dataset.detailTab === tab));
      document.querySelectorAll("[data-detail-panel]").forEach(panel => panel.classList.toggle("active", panel.dataset.detailPanel === tab));
    }
    function bindDetailTabs(root = document) {
      root.querySelectorAll("[data-detail-tab]").forEach(btn => {
        btn.onclick = () => setDetailTab(btn.dataset.detailTab || "preview");
      });
      root.querySelectorAll("[data-log-filter]").forEach(btn => {
        btn.onclick = () => {
          state.logStreamFilter = btn.dataset.logFilter || "all";
          state.detailLogCount = -1;
          root.querySelectorAll("[data-log-filter]").forEach(el => el.classList.toggle("active", el.dataset.logFilter === state.logStreamFilter));
          const panel = $("jobLogPanel");
          if (panel && panel.dataset.logSource) {
            try {
              updateJobLogs(JSON.parse(panel.dataset.logSource), { forceRebuild: true });
            } catch {}
          }
        };
      });
      const follow = $("jobLogFollow");
      if (follow) {
        follow.checked = state.logFollowTail;
        follow.onchange = () => { state.logFollowTail = follow.checked; };
      }
      const copyBtn = $("jobLogCopy");
      if (copyBtn) {
        copyBtn.onclick = async () => {
          const panel = $("jobLogPanel");
          if (!panel) return;
          const text = panel.innerText || "";
          try {
            await navigator.clipboard.writeText(text);
            copyBtn.textContent = "Copied";
            setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
          } catch {
            alert("Copy failed.");
          }
        };
      }
      root.querySelectorAll("[data-copy-path]").forEach(btn => {
        btn.onclick = async () => {
          const text = btn.dataset.copyPath || "";
          if (!text) return;
          try {
            await navigator.clipboard.writeText(text);
            const label = btn.textContent;
            btn.textContent = "Copied";
            setTimeout(() => { btn.textContent = label; }, 1200);
          } catch {
            alert("Copy failed.");
          }
        };
      });
      setDetailTab(state.detailTab || "preview");
    }
    function updateJobLogs(full, options = {}) {
      const panel = $("jobLogPanel");
      if (!panel) return false;
      const logRows = full.logs || [];
      const forceRebuild = !!options.forceRebuild;
      updateLogBadge(logRows.length);
      panel.dataset.logSource = JSON.stringify(logRows);
      const prevCount = state.detailLogCount || 0;
      const filterChanged = panel.dataset.logFilter !== state.logStreamFilter;
      if (filterChanged || forceRebuild || prevCount > logRows.length || state.logStreamFilter !== panel.dataset.activeFilter) {
        const nearBottom = logPanelNearBottom(panel);
        panel.innerHTML = renderLogPanelHtml(logRows);
        panel.dataset.logFilter = state.logStreamFilter;
        panel.dataset.activeFilter = state.logStreamFilter;
        state.detailLogCount = logRows.length;
        if (state.logFollowTail && (nearBottom || full.status === "running" || full.status === "queued")) scrollLogPanelToBottom(panel);
        return true;
      }
      if (logRows.length <= prevCount) return false;
      const nearBottom = logPanelNearBottom(panel);
      logRows.slice(prevCount).forEach(row => {
        if (state.logStreamFilter !== "all" && String(row.stream || "log") !== state.logStreamFilter) return;
        panel.insertAdjacentHTML("beforeend", renderLogLine(row));
      });
      state.detailLogCount = logRows.length;
      if (state.logFollowTail && (nearBottom || full.status === "running" || full.status === "queued")) scrollLogPanelToBottom(panel);
      return true;
    }
    function renderJobs() {
      const type = $("jobTypeFilter").value;
      const status = $("statusFilter").value;
      let jobs = state.jobs;
      if (type !== "all") jobs = jobs.filter(j => j.job_type === type);
      if (status !== "all") jobs = jobs.filter(j => j.status === status);
      $("jobCount").textContent = jobs.length + " jobs";
      $("jobList").innerHTML = jobs.length ? jobs.map(job => `
        <button class="jobRow ${job.id === state.activeJob ? "active" : ""}" data-job="${escapeHtml(job.id)}">
          <div class="jobTitleLine"><strong>${escapeHtml(job.title)}</strong><span class="${statusClass(job.status)}">${escapeHtml(job.status)}</span></div>
          <div class="muted tiny">${escapeHtml(job.model_name)} · ${escapeHtml(job.elapsed)} · ${escapeHtml(job.id)}</div>
        </button>`).join("") : `<div class="detail muted">No jobs yet. Create one above.</div>`;
      document.querySelectorAll("[data-job]").forEach(btn => btn.onclick = () => {
        const nextJob = btn.dataset.job;
        if (nextJob !== state.activeJob) {
          state.activeJob = nextJob;
          state.detailRenderKey = "";
          state.detailLogCount = 0;
          const selected = state.jobs.find(item => item.id === nextJob);
          state.detailTab = (selected && (selected.status === "running" || selected.status === "queued")) ? "logs" : "preview";
        }
        renderJobs();
      });
      renderDetail();
    }
    async function refreshJobs() {
      state.jobs = await api("/api/jobs");
      if (!state.activeJob && state.jobs[0]) state.activeJob = state.jobs[0].id;
      renderJobs();
    }
    async function renderDetail() {
      const job = state.jobs.find(j => j.id === state.activeJob);
      if (!job) {
        state.detailRenderKey = "";
        state.detailLogCount = 0;
        $("jobDetail").innerHTML = `<span class="muted">Select a job.</span>`;
        return;
      }
      let full = job;
      try { full = await api("/api/jobs/" + job.id); } catch {}
      const logRows = full.logs || [];
      const result = full.result || {};
      const actionKey = (full.visualization_actions || []).map(action => [action.mode, action.path, action.label].join(":")).join("|");
      const detailKey = [job.id, full.status, full.error || "", result.preview_video || "", result.preview_image || "", result.preview_model || "", result.preview_splat || "", actionKey].join("|");
      const needsShellRender = state.detailRenderKey !== detailKey;
      if (needsShellRender) {
        state.detailRenderKey = detailKey;
        state.detailLogCount = 0;
        const video = result.preview_video ? renderVideoBox(`/api/jobs/${job.id}/video`, { eager: true }) : "";
        const image = result.preview_image ? renderImageBox(`/api/jobs/${job.id}/image`) : "";
        const evaluationSummary = renderEvaluationSummary(result);
        const resultActions = renderResultActions(full, result);
        const resultJson = full.result ? JSON.stringify(full.result, null, 2) : "";
        const logFilterBtn = (filter, label) => `<button type="button" class="logFilter ${safeClassToken(filter)} ${state.logStreamFilter === filter ? "active" : ""}" data-log-filter="${escapeHtml(filter)}">${escapeHtml(label)}</button>`;
        $("jobDetail").innerHTML = `
          <div class="detailTabs">
            <button type="button" class="detailTab ${state.detailTab === "preview" ? "active" : ""}" data-detail-tab="preview">Preview</button>
            <button type="button" class="detailTab ${state.detailTab === "logs" ? "active" : ""}" data-detail-tab="logs">Logs<span class="logBadge" id="jobLogBadge">${logRows.length}</span></button>
            <button type="button" class="detailTab ${state.detailTab === "result" ? "active" : ""}" data-detail-tab="result">Result</button>
          </div>
          <div class="detailTabPanel ${state.detailTab === "preview" ? "active" : ""}" data-detail-panel="preview">
            ${video || image || `<span class="muted">No preview yet.</span>`}
            ${resultActions}
            <div class="detailGrid">
              <div class="metric"><span>Status</span><strong>${escapeHtml(full.status)}</strong></div>
              <div class="metric"><span>Model</span><strong>${escapeHtml(full.model_name)}</strong></div>
              <div class="metric"><span>Type</span><strong>${escapeHtml(full.job_type)}</strong></div>
            </div>
            ${renderOutputPath(full.output_dir)}
            ${full.error ? `<div class="metric"><span>Error</span><strong>${escapeHtml(full.error)}</strong></div>` : ""}
            ${evaluationSummary}
          </div>
          <div class="detailTabPanel ${state.detailTab === "logs" ? "active" : ""}" data-detail-panel="logs">
            <div class="logToolbar">
              ${logFilterBtn("all", "All")}
              ${logFilterBtn("system", "System")}
              ${logFilterBtn("stdout", "Stdout")}
              ${logFilterBtn("stderr", "Stderr")}
              <button type="button" class="btn" id="jobLogCopy">Copy</button>
              <label class="logFollow"><input type="checkbox" id="jobLogFollow" ${state.logFollowTail ? "checked" : ""} />Follow tail</label>
            </div>
            <div class="logPanel" id="jobLogPanel">${renderLogPanelHtml(logRows)}</div>
          </div>
          <div class="detailTabPanel ${state.detailTab === "result" ? "active" : ""}" data-detail-panel="result">
            ${resultJson ? `<pre>${escapeHtml(resultJson)}</pre>` : `<span class="muted">No structured result yet.</span>`}
          </div>`;
        hydrateLazyMedia($("jobDetail"));
        bindArtifactVisualizerButtons($("jobDetail"));
        bindDetailTabs($("jobDetail"));
        const panel = $("jobLogPanel");
        if (panel) {
          panel.dataset.logSource = JSON.stringify(logRows);
          panel.dataset.logFilter = state.logStreamFilter;
          panel.dataset.activeFilter = state.logStreamFilter;
          state.detailLogCount = logRows.length;
          if (state.logFollowTail && (full.status === "running" || full.status === "queued")) scrollLogPanelToBottom(panel);
        }
      } else {
        updateJobLogs(full);
        updateLogBadge(logRows.length);
      }
    }
    async function loadModels() {
      state.models = await api("/api/models");
      populateModelSelect();
      renderCatalog();
    }
    async function loadEvaluationCatalog() {
      state.evaluationCatalog = await api("/api/evaluation/catalog");
      populateEvaluationControls();
    }
    function populateEvaluationControls() {
      const rows = (state.evaluationCatalog.benchmarks || []).slice(0, 500);
      $("evalBenchmark").innerHTML = optionHtml("", "optional") + rows.map(row => {
        const label = `${row.task_type || row.name} · ${row.benchmark_name || row.name}`;
        return optionHtml(row.benchmark_name || row.name, label, {"data-task": row.task_type || row.name});
      }).join("");
      const examples = state.evaluationCatalog.examples || [];
      $("evalPreset").innerHTML = optionHtml("", "Custom") + examples.map(example => optionHtml(example.id, example.label || example.id)).join("");
      if (!$("evalPreset").dataset.initialized && examples[0]) {
        $("evalPreset").value = examples[0].id;
        $("evalPreset").dataset.initialized = "true";
        applyEvaluationPreset();
      }
      syncEvaluationBenchmarkDefaults();
      syncEvaluationMode();
    }
    function selectedEvaluationBenchmarkId() {
      return String($("evalBenchmark").value || "").trim().toLowerCase().replace(/_/g, "-");
    }
    function evaluationBenchmarkHints(id = selectedEvaluationBenchmarkId()) {
      const hints = state.evaluationCatalog.benchmark_runtime_hints || {};
      return hints[id] || {};
    }
    function syncEvaluationMetricSuggestions() {
      const datalist = $("evalMetricSuggestions");
      if (!datalist) return;
      const hints = evaluationBenchmarkHints();
      const dimensions = hints.dimensions || hints.metrics || [];
      const presets = Object.keys(hints.presets || {});
      datalist.innerHTML = [...dimensions, ...presets].map(item => `<option value="${escapeHtml(item)}"></option>`).join("");
    }
    function syncEvaluationBenchmarkDefaults() {
      const benchmarkId = selectedEvaluationBenchmarkId();
      const metrics = $("evalMetrics");
      const datasetRoot = $("evalDatasetRoot");
      const resultsPath = $("evalResultsPath");
      syncEvaluationMetricSuggestions();
      const hints = evaluationBenchmarkHints(benchmarkId);
      const dimensions = hints.dimensions || hints.metrics || [];
      const defaultMetrics = hints.default_metrics || (hints.primary_presets && hints.primary_presets.default) || dimensions.slice(0, 1);
      if (hints && Object.keys(hints).length) {
        metrics.placeholder = dimensions.slice(0, 4).concat(Object.keys(hints.presets || {}).slice(0, 2)).join(", ") || "metric_or_dimension";
        if (!metrics.value.trim() || ["artifact_count", "required_artifacts_present"].includes(metrics.value.trim())) {
          metrics.value = (defaultMetrics && defaultMetrics.length ? defaultMetrics : dimensions.slice(0, 1)).join(", ");
        }
        datasetRoot.placeholder = "generated videos or benchmark data root";
        resultsPath.placeholder = "official results file, or generated video directory for supported runners";
      } else {
        metrics.placeholder = "artifact_count, required_artifacts_present";
        datasetRoot.placeholder = "optional benchmark dataset root";
        resultsPath.placeholder = "/path/to/results.jsonl for existing-results mode";
      }
    }
    function selectedEvaluationPreset() {
      const id = $("evalPreset").value;
      return (state.evaluationCatalog.examples || []).find(example => example.id === id);
    }
    function setInputValue(id, value) {
      const el = $(id);
      if (!el) return;
      el.value = Array.isArray(value) ? value.join(", ") : String(value ?? "");
    }
    function ensureSelectOption(id, value, label) {
      if (!value) return;
      const el = $(id);
      if (!el || Array.from(el.options).some(option => option.value === value)) return;
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label || value;
      el.appendChild(option);
    }
    function setJsonInputValue(id, value) {
      const el = $(id);
      if (!el) return;
      el.value = JSON.stringify(value || {}, null, 2);
    }
    function applyEvaluationPreset() {
      const preset = selectedEvaluationPreset();
      if (!preset) {
        syncEvaluationMode();
        return;
      }
      setInputValue("evalMode", preset.eval_mode || "existing-results");
      ensureSelectOption("evalBenchmark", preset.benchmark_id || "", preset.benchmark_id || "");
      setInputValue("evalBenchmark", preset.benchmark_id || "");
      setInputValue("evalResultsPath", preset.results_path || "");
      setInputValue("evalRequestsPath", preset.requests_path || "");
      setInputValue("evalModelId", preset.model_id || "");
      setInputValue("evalModelRunner", preset.model_runner || "");
      setInputValue("evalModelVariant", preset.model_variant_id || "");
      setInputValue("evalDatasetId", preset.dataset_id || "");
      setInputValue("evalDatasetRoot", preset.dataset_root || "");
      setInputValue("evalDatasetManifest", preset.dataset_manifest || "");
      setInputValue("evalMetrics", preset.metrics || ["artifact_count"]);
      setInputValue("evalRequiredArtifacts", preset.required_artifacts || []);
      setJsonInputValue("evalModelParameters", preset.call_kwargs || {});
      setJsonInputValue("evalRuntime", preset.load_kwargs || {});
      syncEvaluationBenchmarkDefaults();
      syncEvaluationMode();
    }
    function syncEvaluationMode() {
      const evaluation = $("jobType").value === "evaluation";
      const mode = $("evalMode").value;
      const modelMode = mode === "model" || mode === "model-benchmark" || mode === "generate-and-score";
      const existingMode = mode === "existing-results";
      const benchmarkRuntimeMode = evaluation && Object.keys(evaluationBenchmarkHints()).length > 0;
      setFieldVisible("evalResultsPath", evaluation && existingMode);
      setFieldVisible("evalRequestsPath", evaluation && (mode === "model" || mode === "model-benchmark"));
      ["evalModelId", "evalModelRunner", "evalModelVariant", "evalCacheMode", "evalModelParameters"].forEach(id => setFieldVisible(id, evaluation && modelMode));
      setFieldVisible("evalRuntime", evaluation && (modelMode || benchmarkRuntimeMode));
      setFieldVisible("evalDatasetId", evaluation);
      setFieldVisible("evalDatasetRoot", evaluation && mode !== "generate-and-score");
      setFieldVisible("evalDatasetManifest", evaluation && mode === "generate-and-score");
    }
    function populateModelSelect() {
      const workload = $("workloadType").value;
      const models = state.models.filter(m => m.workload_type === workload || workload === "inference");
      $("modelSelect").innerHTML = models.map(m => optionHtml(m.id, `${m.name} · ${m.category}`)).join("");
      populateInferSpecControls();
      applySelectedModelDefaults();
    }
    function inputTypeForInferField(field) {
      const kind = field.kind || "string";
      if (kind === "integer" || kind === "number") return "number";
      if (kind === "path") return "text";
      if (kind === "boolean") return "checkbox";
      return "text";
    }
    function inferFieldInputValue(field) {
      const value = field.default;
      if (value === null || value === undefined) return "";
      if (Array.isArray(value) || (typeof value === "object" && value !== null)) return JSON.stringify(value);
      return String(value);
    }
    function renderInferDynamicFields() {
      const fields = inferTaskFields().filter(field => !DEDICATED_INFER_FIELD_IDS.has(normalizeFieldId(field.field_id)));
      $("inferDynamicFields").innerHTML = fields.map(field => {
        const fieldId = normalizeFieldId(field.field_id);
        const inputId = `inferDyn_${fieldId.replace(/[^a-z0-9-]/g, "_")}`;
        const label = escapeHtml(field.label || field.field_id);
        const target = escapeHtml(field.target || "call_kwargs");
        const kind = escapeHtml(field.kind || "string");
        const required = field.required ? "true" : "false";
        const description = field.description ? ` placeholder="${escapeHtml(field.description)}"` : "";
        const common = `id="${inputId}" data-infer-field-id="${escapeHtml(fieldId)}" data-target="${target}" data-kind="${kind}" data-required="${required}"`;
        if (Array.isArray(field.choices) && field.choices.length) {
          return `<label>${label}<select ${common}>${field.choices.map(choice => `<option value="${escapeHtml(choice)}" ${String(choice) === String(field.default) ? "selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select></label>`;
        }
        if (field.kind === "boolean") {
          return `<label class="check"><input type="checkbox" ${common} ${field.default ? "checked" : ""} /> ${label}</label>`;
        }
        const value = ` value="${escapeHtml(inferFieldInputValue(field))}"`;
        return `<label>${label}<input type="${inputTypeForInferField(field)}" ${common}${value}${description} /></label>`;
      }).join("");
    }
    function updateCreateMode() {
      const type = $("jobType").value;
      const inference = type === "inference";
      const evaluation = type === "evaluation";
      INFER_INFRA_FIELDS.forEach(id => setFieldVisible(id, inference));
      const currentModel = selectedModel();
      const apiBackend = inference && currentModel && currentModel.supports_api_init && $("backend").value === "api_init";
      const attentionVisible = inference && currentModel && currentModel.supports_attention_backend;
      setFieldVisible("endpoint", apiBackend);
      setFieldVisible("apiKey", apiBackend);
      setFieldVisible("attention", attentionVisible);
      if (!attentionVisible) $("attention").value = "auto";
      const fieldIds = inferTaskFieldIds();
      Object.entries(INFER_TASK_FIELD_CONTROLS).forEach(([controlId, aliases]) => {
        const visible = inference && aliases.some(alias => fieldIds.has(normalizeFieldId(alias)));
        setFieldVisible(controlId, visible);
      });
      $("inferDynamicFields").classList.toggle("hidden", !inference || !inferTaskFields().some(field => !DEDICATED_INFER_FIELD_IDS.has(normalizeFieldId(field.field_id))));
      $("inferSpecSummary").classList.toggle("hidden", !inference);
      [
        "evalMode","evalPreset","evalBenchmark","evalResultsPath","evalRequestsPath","evalOutputDir","evalRunPlanPath",
        "evalModelId","evalModelRunner","evalModelVariant","evalDatasetId","evalDatasetRoot","evalDatasetManifest","evalCacheMode","evalMetrics",
        "evalRequiredArtifacts","evalModelParameters","evalRuntime"
      ].forEach(id => setFieldVisible(id, evaluation));
      syncEvaluationMode();
      const runtimeOptions = (selectedModel() && selectedModel().runtime_options) || {};
      let runtimeVisible = false;
      Object.entries(RUNTIME_CHECK_OPTIONS).forEach(([id, key]) => {
        const visible = inference && !!(runtimeOptions[key] && runtimeOptions[key].supported);
        setRuntimeCheckVisible(id, visible);
        if (!visible) $(id).checked = false;
        runtimeVisible = runtimeVisible || visible;
      });
      $("runtimeChecks").classList.toggle("hidden", !runtimeVisible);
    }
    function selectedModel() {
      return state.models.find(m => m.id === $("modelSelect").value);
    }
    function selectedVariant() {
      const model = selectedModel();
      const variants = model && model.variants && model.variants.length ? model.variants : [{variant_id: "default", label: "Default", model_ref: model ? model.model_ref : ""}];
      return variants.find(v => v.variant_id === $("variantSelect").value) || variants[0];
    }
    function selectedTaskProfile() {
      const model = selectedModel();
      const tasks = model && model.tasks && model.tasks.length ? model.tasks : [{task_id: "default", label: "Default Inference", inputs: [], outputs: []}];
      return tasks.find(t => t.task_id === $("taskProfile").value) || tasks[0];
    }
    function taskDefaultForAliases(aliases, fallback) {
      const normalized = new Set(aliases.map(alias => normalizeFieldId(alias)));
      const field = inferTaskFields().find(item => normalized.has(normalizeFieldId(item.field_id)));
      if (!field) return undefined;
      if (field.default === null || field.default === undefined) return fallback;
      return field.default;
    }
    function taskFieldForAliases(aliases) {
      const normalized = new Set(aliases.map(alias => normalizeFieldId(alias)));
      return inferTaskFields().find(item => normalized.has(normalizeFieldId(item.field_id)));
    }
    function applyDedicatedInferValue(controlId, aliases, value, payload) {
      if (!fieldControlVisible(controlId)) return;
      const field = taskFieldForAliases(aliases);
      const fieldId = field ? normalizeFieldId(field.field_id) : normalizeFieldId(aliases[0] || controlId);
      const target = field && field.target ? field.target : "params";
      applyInferValue(target, fieldId, value, payload);
    }
    function setControlDefault(id, value) {
      if (value === undefined || value === null) return;
      const el = $(id);
      if (!el) return;
      el.value = String(value);
    }
    function applyInferTaskDefaults() {
      setControlDefault("prompt", taskDefaultForAliases(["prompt"], ""));
      setControlDefault("inputPath", taskDefaultForAliases(INFER_TASK_FIELD_CONTROLS.inputPath, ""));
      setControlDefault("numFrames", taskDefaultForAliases(INFER_TASK_FIELD_CONTROLS.numFrames, state.settings.num_frames));
      setControlDefault("fps", taskDefaultForAliases(INFER_TASK_FIELD_CONTROLS.fps, state.settings.fps));
      setControlDefault("height", taskDefaultForAliases(INFER_TASK_FIELD_CONTROLS.height, state.settings.height));
      setControlDefault("width", taskDefaultForAliases(INFER_TASK_FIELD_CONTROLS.width, state.settings.width));
      setControlDefault("steps", taskDefaultForAliases(INFER_TASK_FIELD_CONTROLS.steps, state.settings.num_inference_steps));
      setControlDefault("guidance", taskDefaultForAliases(INFER_TASK_FIELD_CONTROLS.guidance, state.settings.guidance_scale));
      setControlDefault("seed", taskDefaultForAliases(INFER_TASK_FIELD_CONTROLS.seed, state.settings.seed));
    }
    function populateInferSpecControls() {
      const model = selectedModel();
      const variants = model && model.variants && model.variants.length ? model.variants : [{variant_id: "default", label: "Default"}];
      const tasks = model && model.tasks && model.tasks.length ? model.tasks : [{task_id: "default", label: "Default Inference"}];
      $("variantSelect").innerHTML = variants.map(v => optionHtml(v.variant_id, `${v.label || v.variant_id} · ${v.status || "configured"}`)).join("");
      $("variantSelect").value = (model && model.default_variant_id) || variants[0].variant_id;
      $("taskProfile").innerHTML = tasks.map(t => optionHtml(t.task_id, t.label || t.task_id)).join("");
      $("taskProfile").value = (model && model.default_task_id) || tasks[0].task_id;
      renderInferDynamicFields();
      renderInferSpecSummary();
      updateCreateMode();
      applyInferTaskDefaults();
    }
    function backendOptionsForModel(model) {
      const options = [["auto", "auto"]];
      if (!model || model.supports_from_pretrained) options.push(["from_pretrained", "from_pretrained"]);
      if (model && model.supports_api_init) options.push(["api_init", "api_init"]);
      return options;
    }
    function refreshBackendOptions(selected) {
      const model = selected || selectedModel();
      const previous = $("backend").value || "auto";
      const options = backendOptionsForModel(model);
      $("backend").innerHTML = options.map(([value, label]) => optionHtml(value, label)).join("");
      const values = new Set(options.map(([value]) => value));
      const preferred = model && model.backend && values.has(model.backend) ? model.backend : "auto";
      $("backend").value = values.has(previous) ? previous : preferred;
    }
    function renderInferSpecSummary() {
      const variant = selectedVariant();
      const task = selectedTaskProfile();
      const inputs = ((task && task.inputs) || []).map(item => item.label || item.field_id).join(", ") || "generic inputs";
      const outputs = ((task && task.outputs) || []).map(item => item.kind || item.artifact_id).join(", ") || "artifacts";
      $("inferSpecSummary").innerHTML = `<span>Inference Contract</span><strong>${escapeHtml(variant.label || variant.variant_id)} / ${escapeHtml(task.label || task.task_id)}</strong><p class="muted tiny">Inputs: ${escapeHtml(inputs)}</p><p class="muted tiny">Outputs: ${escapeHtml(outputs)}</p>`;
    }
    function applySelectedModelDefaults() {
      const selected = selectedModel();
      const variant = selectedVariant();
      if (selected) {
        refreshBackendOptions(selected);
        const backendSetting = state.settings.backend || "";
        const backend = backendSetting && backendSetting !== "auto" ? backendSetting : (selected.backend || "auto");
        const backendValues = new Set(backendOptionsForModel(selected).map(([value]) => value));
        $("backend").value = backendValues.has(backend) ? backend : "auto";
        $("modelRef").value = (variant && variant.model_ref) || selected.model_ref || "";
        $("endpoint").value = selected.endpoint || "";
      }
      updateCreateMode();
      applyInferTaskDefaults();
    }
    function renderCatalog() {
      const q = $("catalogSearch").value.toLowerCase();
      const workload = $("catalogWorkload").value;
      let models = state.models;
      if (workload !== "all") models = models.filter(m => m.workload_type === workload);
      if (q) models = models.filter(m => [m.name, m.id, m.category, m.family, (m.tags || []).join(" ")].join(" ").toLowerCase().includes(q));
      $("catalogGrid").innerHTML = models.map(m => `<div class="itemCard"><strong>${escapeHtml(m.name)}</strong>${pillHtml(m.workload_type)}<p>${escapeHtml(m.summary)}</p><p class="muted tiny">${escapeHtml(m.id)}</p>${catalogLinksHtml(m.links)}</div>`).join("");
    }
    async function renderGallery() {
      const rows = await api("/api/gallery");
      if (state.lazyVideoObserver) state.lazyVideoObserver.disconnect();
      state.lazyVideoObserver = null;
      state.autoVideoPreloads = 0;
      $("galleryGrid").innerHTML = rows.length ? rows.map(row => {
        const media = row.video_url
          ? renderVideoBox(row.video_url, { poster: row.image_url || "" })
          : row.image_url
            ? renderImageBox(row.image_url)
            : "";
        const actions = (row.visualization_actions || [])
          .map(action => artifactVisualizeButton(action.label, action.path, action.mode, action.model_id || row.model_id || ""))
          .filter(Boolean)
          .join("");
        const refine = (row.video_url || row.image_url || row.model_url) ? artifactRefineButton(row.job_id || "", row.run_id || "") : "";
        const allActions = [refine, actions].filter(Boolean).join("");
        return `<div class="itemCard">${media}<strong>${escapeHtml(row.title)}</strong><p>${escapeHtml(gallerySubtitle(row))}</p>${allActions ? `<div class="artifactLinks">${allActions}</div>` : ""}</div>`;
      }).join("") : `<div class="muted">No completed inference outputs yet.</div>`;
      hydrateLazyMedia($("galleryGrid"));
      bindArtifactVisualizerButtons($("galleryGrid"));
    }
    async function renderArtifacts() {
      const rows = await api("/api/artifacts");
      $("artifactList").innerHTML = rows.length ? rows.map(artifactCard).join("") : `<span class="muted">No artifacts yet.</span>`;
      bindArtifactVisualizerButtons($("artifactList"));
    }
    function artifactCard(row) {
      const canOpen = row.name !== "output_dir";
      const link = canOpen ? artifactLink("Open", row.path) : "";
      const visualize = artifactVisualizeButton(row.visualizer_label, row.path, row.visualizer_mode, row.model_id);
      const actions = [link, visualize].filter(Boolean).join("");
      return `<div class="metric"><span>${escapeHtml(row.model_name)} · ${escapeHtml(row.job_type)} · ${escapeHtml(row.job_id || row.run_id || "")}</span><strong>${escapeHtml(row.name)}</strong><p class="muted tiny">${escapeHtml(row.path)}</p>${actions ? `<div class="artifactLinks">${actions}</div>` : ""}</div>`;
    }
    function viewerUrlForBrowser(raw) {
      if (!raw) return "";
      try {
        const url = new URL(raw, window.location.href);
        const localHosts = new Set(["127.0.0.1", "localhost", "0.0.0.0"]);
        if (localHosts.has(url.hostname) && !localHosts.has(window.location.hostname)) {
          url.hostname = window.location.hostname;
        }
        return url.toString();
      } catch {
        return String(raw || "");
      }
    }
    function visualizerStatusHtml(v) {
      const status = v.status;
      if (!status) return `<div class="visualizerStatus" data-visualizer-status="${escapeHtml(v.mode)}">Not running.</div>`;
      const running = status.running ? "running" : "stopped";
      const url = viewerUrlForBrowser(status.url);
      const urlLink = url ? `<a class="artifactLink" href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(url)}</a>` : "";
      const log = status.log_path ? `<br>Log: ${escapeHtml(status.log_path)}` : "";
      const params = status.params && Object.keys(status.params).length
        ? `<br>Params: ${escapeHtml(JSON.stringify(status.params))}`
        : "";
      return `<div class="visualizerStatus" data-visualizer-status="${escapeHtml(v.mode)}">${escapeHtml(running)}${urlLink ? ` · ${urlLink}` : ""}${log}${params}</div>`;
    }
    function runningVisualizers() {
      return state.visualizers.filter(item => item.status && item.status.running && item.status.url);
    }
    function syncVisualizerPreview(preferredMode = "") {
      const running = runningVisualizers();
      const panel = $("visualizerPreviewPanel");
      const frame = $("visualizerPreviewFrame");
      const tabs = $("visualizerPreviewTabs");
      if (!panel || !frame) return;
      if (!running.length) {
        panel.classList.add("hidden");
        frame.removeAttribute("src");
        state.visualizerPreviewMode = "";
        if (tabs) tabs.innerHTML = "";
        return;
      }
      panel.classList.remove("hidden");
      const modes = running.map(item => item.mode);
      if (preferredMode && modes.includes(preferredMode)) state.visualizerPreviewMode = preferredMode;
      if (!modes.includes(state.visualizerPreviewMode)) state.visualizerPreviewMode = modes[0];
      const selected = running.find(item => item.mode === state.visualizerPreviewMode) || running[0];
      if (tabs) {
        tabs.innerHTML = running.map(item => {
          const active = item.mode === selected.mode ? " active" : "";
          return `<button type="button" class="visualizerPreviewTab${active}" data-preview-mode="${escapeHtml(item.mode)}">${escapeHtml(item.title || item.mode)}</button>`;
        }).join("");
        tabs.querySelectorAll("[data-preview-mode]").forEach(button => {
          button.onclick = () => syncVisualizerPreview(button.dataset.previewMode);
        });
      }
      const url = viewerUrlForBrowser(selected.status.url);
      if (frame.getAttribute("src") !== url) frame.setAttribute("src", url);
      const popout = $("visualizerPreviewPopout");
      if (popout) popout.onclick = () => window.open(url, "_blank", "noopener,noreferrer");
    }
    function visualizerPlaceholder(mode, kind) {
      if (kind === "asset") {
        if (mode === "media") return "/path/to/preview.png or preview.mp4";
        if (mode === "points") return "/path/to/points.npz or scene.ply";
        if (mode === "rerun") return "/path/to/timeline.rrd";
        if (mode === "spark") return "/path/to/scene.splat (optional)";
        return "optional asset path";
      }
      if (mode === "embodied") return "http://127.0.0.1:18610";
      if (mode === "rerun") return "existing Rerun URL (optional)";
      return "external viewer URL";
    }
    function visualizerDefaultParams(mode) {
      if (mode === "points") {
        return {
          coordinate_preset: "auto",
          up_direction: "auto",
          alignment: "auto",
          point_size: 0.02,
          point_shape: "circle",
          max_points: 400000
        };
      }
      return {};
    }
    function visualizerParamsText(mode, status) {
      const params = status && status.params && Object.keys(status.params).length
        ? status.params
        : visualizerDefaultParams(mode);
      return JSON.stringify(params, null, 2);
    }
    function visualizerCard(v) {
      const mode = v.mode;
      const status = v.status || null;
      const caps = (v.capabilities || []).map(cap => pillHtml(cap)).join(" ");
      const aliases = (v.aliases || []).join(", ") || mode;
      const assetDisplay = v.requires_asset ? "Asset path" : "Asset path";
      const paramsText = visualizerParamsText(mode, status);
      const urlField = v.accepts_external_url
        ? `<label class="wide visualizerField">External URL<input data-visualizer-field="url" placeholder="${escapeHtml(visualizerPlaceholder(mode, "url"))}" /></label>`
        : `<label class="wide visualizerField visualizerFieldReserved" aria-hidden="true"><span>External URL</span><input disabled tabindex="-1" aria-hidden="true" placeholder="${escapeHtml(visualizerPlaceholder(mode, "url"))}" /></label>`;
      return `<div class="itemCard visualizerCard" data-visualizer-mode="${escapeHtml(mode)}" data-default-port="${escapeHtml(v.default_port || "")}">
        <div class="visualizerHead">
          <strong>${escapeHtml(v.title)}</strong>
          <div class="visualizerBadges">${pillHtml(mode)} ${v.native ? pillHtml("native") : pillHtml("web")}</div>
        </div>
        <div class="visualizerTags">${caps || `<span class="muted tiny">No capability tags</span>`}</div>
        <p class="visualizerAliases">Aliases: ${escapeHtml(aliases)}</p>
        <div class="visualizerControls">
          <label>Model<input data-visualizer-field="model" value="${escapeHtml(v.default_model || "")}" /></label>
          <label>Port<input data-visualizer-field="port" type="number" min="1" max="65535" value="${escapeHtml(v.default_port || "")}" /></label>
          <label class="wide">${escapeHtml(assetDisplay)}<input data-visualizer-field="asset" placeholder="${escapeHtml(visualizerPlaceholder(mode, "asset"))}" /></label>
          ${urlField}
          <label class="wide visualizerField">Params JSON<textarea data-visualizer-field="params" spellcheck="false">${escapeHtml(paramsText)}</textarea></label>
        </div>
        <div class="visualizerFooter">
          <div class="visualizerActions">
            <button class="btn primary" data-launch-visualizer="${escapeHtml(mode)}">Launch</button>
            <button class="btn" data-open-visualizer="${escapeHtml(mode)}">Open</button>
            <button class="btn danger" data-stop-visualizer="${escapeHtml(mode)}">Stop</button>
          </div>
          ${visualizerStatusHtml(v)}
        </div>
      </div>`;
    }
    function visualizerStatusKey(v) {
      const status = v && v.status ? v.status : {};
      return [
        Boolean(status.running),
        status.url || "",
        status.asset_path || "",
        status.model_id || "",
        JSON.stringify(status.params || {}),
        status.returncode ?? "",
        status.log_path || ""
      ].join("|");
    }
    function setVisualizerLiveState(text) {
      const node = $("visualizerLiveState");
      if (node) node.textContent = text;
    }
    function applyVisualizerRealtimeUpdate(nextItems) {
      if (!Array.isArray(nextItems)) return;
      const currentModes = state.visualizers.map(item => item.mode).join("|");
      const nextModes = nextItems.map(item => item.mode).join("|");
      if (!state.visualizers.length || currentModes !== nextModes || !$("visualizerGrid").children.length) {
        state.visualizers = nextItems;
        renderVisualizers(false);
        return;
      }
      nextItems.forEach(next => {
        const index = state.visualizers.findIndex(item => item.mode === next.mode);
        if (index < 0) return;
        const previous = state.visualizers[index];
        state.visualizers[index] = next;
        const card = document.querySelector(`[data-visualizer-mode="${CSS.escape(next.mode)}"]`);
        if (!card) return;
        if (visualizerStatusKey(previous) !== visualizerStatusKey(next)) {
          const statusNode = card.querySelector(`[data-visualizer-status="${CSS.escape(next.mode)}"]`);
          if (statusNode) statusNode.outerHTML = visualizerStatusHtml(next);
        }
      });
      syncVisualizerPreview(state.visualizerPreviewMode);
    }
    function updateVisualizerStatus(mode, html) {
      const node = document.querySelector(`[data-visualizer-mode="${CSS.escape(mode)}"] [data-visualizer-status="${CSS.escape(mode)}"]`);
      if (node) node.innerHTML = html;
    }
    function visualizerPayload(mode) {
      const card = document.querySelector(`[data-visualizer-mode="${CSS.escape(mode)}"]`);
      const field = (name) => card?.querySelector(`[data-visualizer-field="${name}"]`)?.value?.trim() || "";
      const portText = field("port");
      const defaultPort = card?.dataset.defaultPort || "";
      let params = {};
      const paramsText = field("params") || "{}";
      try {
        params = JSON.parse(paramsText);
      } catch (error) {
        throw new Error(`${mode} Params JSON must be valid JSON.`);
      }
      if (!params || Array.isArray(params) || typeof params !== "object") {
        throw new Error(`${mode} Params JSON must be an object.`);
      }
      return {
        model_id: field("model"),
        asset_path: field("asset"),
        simulator_url: field("url"),
        host: "127.0.0.1",
        port: portText && portText !== defaultPort ? Number(portText) : null,
        reuse: true,
        params
      };
    }
    async function launchVisualizer(mode) {
      updateVisualizerStatus(mode, "Launching...");
      try {
        const status = await api(`/api/visualizers/${encodeURIComponent(mode)}/launch`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(visualizerPayload(mode))
        });
        const row = state.visualizers.find(item => item.mode === mode);
        if (row) row.status = status;
        renderVisualizers(false);
        refreshVisualizersRealtime({force: true});
      } catch (error) {
        updateVisualizerStatus(mode, `Launch failed: ${escapeHtml(error.message || error)}`);
      }
    }
    async function stopVisualizer(mode) {
      await api(`/api/visualizers/${encodeURIComponent(mode)}/stop`, {method: "POST"});
      const row = state.visualizers.find(item => item.mode === mode);
      if (row) row.status = null;
      renderVisualizers(false);
      refreshVisualizersRealtime({force: true});
    }
    function openVisualizer(mode) {
      const row = state.visualizers.find(item => item.mode === mode);
      const url = row && row.status && row.status.url ? row.status.url : visualizerPayload(mode).simulator_url;
      if (!url) {
        updateVisualizerStatus(mode, "Launch the viewer first, or provide an external URL.");
        return;
      }
      syncVisualizerPreview(mode);
      window.open(viewerUrlForBrowser(url), "_blank", "noopener,noreferrer");
    }
    async function renderVisualizers(refresh = false) {
      state.visualizers = refresh || !state.visualizers.length ? await api("/api/visualizers") : state.visualizers;
      state.visualizerLastSync = Date.now();
      setVisualizerLiveState(`Live sync ${new Date(state.visualizerLastSync).toLocaleTimeString()}`);
      $("visualizerGrid").innerHTML = state.visualizers.map(v => {
        return visualizerCard(v);
      }).join("");
      $("visualizerGrid").querySelectorAll("[data-launch-visualizer]").forEach(btn => btn.onclick = () => launchVisualizer(btn.dataset.launchVisualizer));
      $("visualizerGrid").querySelectorAll("[data-open-visualizer]").forEach(btn => btn.onclick = () => openVisualizer(btn.dataset.openVisualizer));
      $("visualizerGrid").querySelectorAll("[data-stop-visualizer]").forEach(btn => btn.onclick = () => stopVisualizer(btn.dataset.stopVisualizer));
      syncVisualizerPreview(state.visualizerPreviewMode);
    }
    async function refreshVisualizersRealtime(options = {}) {
      if (state.view !== "visualizers" && !options.force) return;
      if (state.visualizerRefreshInFlight) return;
      state.visualizerRefreshInFlight = true;
      setVisualizerLiveState("Live sync updating...");
      try {
        const nextItems = await api("/api/visualizers");
        applyVisualizerRealtimeUpdate(nextItems);
        state.visualizerLastSync = Date.now();
        setVisualizerLiveState(`Live sync ${new Date(state.visualizerLastSync).toLocaleTimeString()}`);
      } catch (error) {
        setVisualizerLiveState(`Live sync failed: ${error.message || error}`);
      } finally {
        state.visualizerRefreshInFlight = false;
      }
    }
    function applySettingsToCreateForm() {
      $("numFrames").value = state.settings.num_frames;
      $("fps").value = state.settings.fps;
      $("height").value = state.settings.height;
      $("width").value = state.settings.width;
      $("steps").value = state.settings.num_inference_steps;
      $("guidance").value = state.settings.guidance_scale;
      $("seed").value = state.settings.seed;
      $("device").value = state.settings.device;
      $("attention").value = state.settings.attention_backend;
      const selected = state.models.find(m => m.id === $("modelSelect").value);
      refreshBackendOptions(selected);
      const backend = state.settings.backend || "auto";
      const backendValues = new Set(backendOptionsForModel(selected).map(([value]) => value));
      if (backend === "auto") {
        $("backend").value = selected && selected.backend && backendValues.has(selected.backend) ? selected.backend : "auto";
      } else {
        $("backend").value = backendValues.has(backend) ? backend : "auto";
      }
      $("torchCompile").checked = !!state.settings.torch_compile;
      $("cpuOffload").checked = !!state.settings.cpu_offload;
      updateCreateMode();
    }
    async function loadSettings() {
      state.settings = await api("/api/settings");
      applySettingsToCreateForm();
    }
    async function loadOptional(label, loader) {
      try {
        await loader();
      } catch (err) {
        console.warn(`${label} unavailable`, err);
        const message = err && err.message ? err.message : String(err);
        $("serverState").textContent = `API connected · ${label} unavailable`;
        return message;
      }
      return "";
    }
    async function initializeWorkspace() {
      const required = await Promise.allSettled([loadSettings(), loadModels(), refreshJobs()]);
      const failed = required.find(result => result.status === "rejected");
      if (failed) {
        throw failed.reason || new Error("Workspace failed to initialize.");
      }
      await Promise.all([
        loadOptional("evaluation catalog", loadEvaluationCatalog),
      ]);
      setView(state.view);
    }
    async function createJob() {
      const jobType = $("jobType").value;
      let payload = { job_type: jobType };
      try {
        if (jobType === "inference") {
          const callJson = parseJsonField("callJson", {});
          const loadJson = parseJsonField("loadJson", {});
          const params = {};
          const dynamic = { params: {}, call_kwargs: {}, load_kwargs: {} };
          document.querySelectorAll("[data-infer-field-id]").forEach(input => {
            const value = coerceInferFieldValue(input);
            applyInferValue(input.dataset.target || "call_kwargs", input.dataset.inferFieldId, value, dynamic);
          });
          applyDedicatedInferValue("numFrames", INFER_TASK_FIELD_CONTROLS.numFrames, Number($("numFrames").value || 0), dynamic);
          applyDedicatedInferValue("fps", INFER_TASK_FIELD_CONTROLS.fps, Number($("fps").value || 16), dynamic);
          applyDedicatedInferValue("height", INFER_TASK_FIELD_CONTROLS.height, Number($("height").value || 0), dynamic);
          applyDedicatedInferValue("width", INFER_TASK_FIELD_CONTROLS.width, Number($("width").value || 0), dynamic);
          applyDedicatedInferValue("steps", INFER_TASK_FIELD_CONTROLS.steps, Number($("steps").value || 0), dynamic);
          applyDedicatedInferValue("guidance", INFER_TASK_FIELD_CONTROLS.guidance, Number($("guidance").value || 0), dynamic);
          applyDedicatedInferValue("seed", INFER_TASK_FIELD_CONTROLS.seed, Number($("seed").value || -1), dynamic);
          if (fieldControlVisible("attention")) params.attention_backend = $("attention").value;
          if (runtimeCheckVisible("torchCompile")) params.torch_compile = $("torchCompile").checked;
          if (runtimeCheckVisible("cpuOffload")) params.cpu_offload = $("cpuOffload").checked;
          if (runtimeCheckVisible("vaeOffload")) params.vae_cpu_offload = $("vaeOffload").checked;
          if (runtimeCheckVisible("textOffload")) params.text_encoder_cpu_offload = $("textOffload").checked;
          Object.assign(params, dynamic.params);
          payload = {
            job_type: "inference",
            workload_type: $("workloadType").value,
            model_id: $("modelSelect").value,
            variant_id: $("variantSelect").value,
            task_profile_id: $("taskProfile").value,
            prompt: fieldControlVisible("prompt") ? $("prompt").value : (dynamic.prompt || ""),
            negative_prompt: fieldControlVisible("negativePrompt") ? $("negativePrompt").value : "",
            input_path: fieldControlVisible("inputPath") ? $("inputPath").value : (dynamic.input_path || ""),
            model_ref: $("modelRef").value,
            backend: $("backend").value,
            endpoint: fieldControlVisible("endpoint") ? $("endpoint").value : "",
            api_key: fieldControlVisible("apiKey") ? $("apiKey").value : "",
            device: $("device").value,
            params,
            call_kwargs: {...dynamic.call_kwargs, ...callJson},
            load_kwargs: {...dynamic.load_kwargs, ...loadJson}
          };
        } else if (jobType === "evaluation") {
          payload = {
            job_type: "evaluation",
            eval_mode: $("evalMode").value,
            benchmark_id: $("evalBenchmark").value,
            results_path: $("evalResultsPath").value,
            requests_path: $("evalRequestsPath").value,
            output_dir: $("evalOutputDir").value,
            run_plan_path: $("evalRunPlanPath").value,
            model_id: $("evalModelId").value,
            model_runner: $("evalModelRunner").value,
            model_variant_id: $("evalModelVariant").value,
            dataset_id: $("evalDatasetId").value,
            dataset_root: $("evalDatasetRoot").value,
            dataset_manifest: $("evalDatasetManifest").value,
            generation_cache_mode: $("evalCacheMode").value,
            metrics: splitList($("evalMetrics").value),
            required_artifacts: splitList($("evalRequiredArtifacts").value),
            call_kwargs: parseJsonField("evalModelParameters", {}),
            load_kwargs: parseJsonField("evalRuntime", {})
          };
        } else {
          throw new Error(`Unsupported job type: ${jobType}`);
        }
      } catch (err) {
        alert(err.message || "JSON fields must be valid.");
        return;
      }
      if (jobType === "evaluation" && ["score-artifacts", "model-benchmark", "generate-and-score"].includes(payload.eval_mode)) {
        const prepared = await api("/api/evaluation/prepare", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify(payload)
        });
        if (!prepared.ready) {
          const errors = (prepared.issues || []).filter(issue => issue.severity === "error").map(issue => issue.message);
          throw new Error(errors.join("\n") || "Evaluation preflight failed.");
        }
      }
      const job = await api("/api/jobs", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(payload) });
      state.activeJob = job.id;
      state.detailRenderKey = "";
      state.detailLogCount = 0;
      state.detailTab = "logs";
      $("createDialog").close();
      setView(jobType);
      await refreshJobs();
    }
    document.querySelectorAll(".navBtn").forEach(btn => btn.onclick = () => { setView(btn.dataset.view); if (btn.dataset.view === "gallery") renderGallery(); if (btn.dataset.view === "artifacts") renderArtifacts(); if (btn.dataset.view === "visualizers") renderVisualizers(true); });
    $("openCreate").onclick = () => { updateCreateMode(); $("createDialog").showModal(); };
    $("createJob").onclick = (event) => { event.preventDefault(); createJob().catch(err => alert(err.message)); };
    $("refreshJobs").onclick = () => refreshJobs();
    $("refreshVisualizers").onclick = () => refreshVisualizersRealtime({force: true});
    $("jobType").onchange = updateCreateMode;
    $("jobTypeFilter").onchange = renderJobs;
    $("statusFilter").onchange = renderJobs;
    $("workloadType").onchange = populateModelSelect;
    $("evalPreset").onchange = applyEvaluationPreset;
    $("evalMode").onchange = syncEvaluationMode;
    $("evalBenchmark").onchange = () => { syncEvaluationBenchmarkDefaults(); syncEvaluationMode(); };
    $("modelSelect").onchange = () => { populateInferSpecControls(); applySelectedModelDefaults(); };
    $("variantSelect").onchange = () => { renderInferSpecSummary(); applySelectedModelDefaults(); };
    $("taskProfile").onchange = () => { renderInferDynamicFields(); renderInferSpecSummary(); updateCreateMode(); applyInferTaskDefaults(); };
    $("backend").onchange = updateCreateMode;
    $("catalogSearch").oninput = renderCatalog;
    $("catalogWorkload").onchange = renderCatalog;
    $("stopJob").onclick = async () => { if (!state.activeJob) return; await api(`/api/jobs/${state.activeJob}/stop`, {method:"POST"}); await refreshJobs(); };
    function readStoredSplitLeft(key) {
      if (!key) return null;
      try {
        const raw = localStorage.getItem(key);
        if (raw == null || raw === "") return null;
        const value = Number(raw);
        return Number.isFinite(value) && value > 0 ? value : null;
      } catch (_) {
        return null;
      }
    }
    function splitHandleWidth(split) {
      const raw = getComputedStyle(split).getPropertyValue("--split-handle");
      const value = Number.parseFloat(raw);
      return Number.isFinite(value) && value > 0 ? value : 10;
    }
    function clampSplitLeft(split, width) {
      const minLeft = Number(split.dataset.splitMinLeft || 240);
      const minRight = Number(split.dataset.splitMinRight || 360);
      const handle = splitHandleWidth(split);
      const total = split.getBoundingClientRect().width;
      const available = Math.max(0, total - handle);
      // Absolute floors so a narrow container never forces an impossible layout.
      const hardMinLeft = Math.min(minLeft, Math.max(160, Math.floor(available * 0.25)));
      const hardMinRight = Math.min(minRight, Math.max(240, Math.floor(available * 0.35)));
      const maxLeft = Math.max(hardMinLeft, available - hardMinRight);
      const minAllowed = Math.min(hardMinLeft, maxLeft);
      return Math.round(Math.min(Math.max(Number(width) || minAllowed, minAllowed), maxLeft));
    }
    function applySplitLeft(split, width, { persist = true } = {}) {
      if (!split || split.clientWidth <= 0) return null;
      const handle = split.querySelector(".splitHandle");
      if (handle && getComputedStyle(handle).display === "none") return null;
      const next = clampSplitLeft(split, width);
      split.style.setProperty("--split-left", `${next}px`);
      if (persist && split.dataset.splitKey) {
        try { localStorage.setItem(split.dataset.splitKey, String(next)); } catch (_) {}
      }
      return next;
    }
    function currentSplitLeft(split, fallback) {
      const inline = Number.parseFloat(split.style.getPropertyValue("--split-left"));
      if (Number.isFinite(inline) && inline > 0) return inline;
      const computed = Number.parseFloat(getComputedStyle(split).getPropertyValue("--split-left"));
      if (Number.isFinite(computed) && computed > 0) return computed;
      return fallback;
    }
    function initResizableSplits() {
      document.querySelectorAll(".split[data-split-key]").forEach(split => {
        const handle = split.querySelector(".splitHandle");
        if (!handle || handle.dataset.splitBound === "1") return;
        handle.dataset.splitBound = "1";
        const defaultWidth = Number(split.dataset.splitDefault || 360);
        const stored = readStoredSplitLeft(split.dataset.splitKey);
        const syncFromStorageOrDefault = () => {
          applySplitLeft(split, stored == null ? defaultWidth : stored, { persist: false });
        };
        // Layout may not be ready on first paint; sync now and on next frames.
        syncFromStorageOrDefault();
        requestAnimationFrame(syncFromStorageOrDefault);

        let dragging = false;
        let activePointerId = null;

        const onPointerMove = (event) => {
          if (!dragging || event.pointerId !== activePointerId) return;
          const left = event.clientX - split.getBoundingClientRect().left;
          applySplitLeft(split, left);
        };
        const stopDrag = (event) => {
          if (!dragging) return;
          if (event && activePointerId != null && event.pointerId !== activePointerId) return;
          dragging = false;
          const pointerId = activePointerId;
          activePointerId = null;
          split.classList.remove("is-resizing");
          handle.removeEventListener("pointermove", onPointerMove);
          handle.removeEventListener("pointerup", stopDrag);
          handle.removeEventListener("pointercancel", stopDrag);
          if (pointerId != null) {
            try { handle.releasePointerCapture(pointerId); } catch (_) {}
          }
        };
        handle.addEventListener("pointerdown", (event) => {
          if (event.button !== 0) return;
          if (getComputedStyle(handle).display === "none") return;
          event.preventDefault();
          dragging = true;
          activePointerId = event.pointerId;
          split.classList.add("is-resizing");
          handle.setPointerCapture(event.pointerId);
          handle.addEventListener("pointermove", onPointerMove);
          handle.addEventListener("pointerup", stopDrag);
          handle.addEventListener("pointercancel", stopDrag);
          onPointerMove(event);
        });
        handle.addEventListener("dblclick", () => {
          if (getComputedStyle(handle).display === "none") return;
          applySplitLeft(split, defaultWidth);
        });
        handle.addEventListener("keydown", (event) => {
          if (getComputedStyle(handle).display === "none") return;
          const current = currentSplitLeft(split, defaultWidth);
          if (event.key === "ArrowLeft") {
            event.preventDefault();
            applySplitLeft(split, current - 24);
          } else if (event.key === "ArrowRight") {
            event.preventDefault();
            applySplitLeft(split, current + 24);
          } else if (event.key === "Home") {
            event.preventDefault();
            applySplitLeft(split, defaultWidth);
          }
        });
      });

      // One shared resize listener reclamps all splits when the viewport changes.
      if (!window.__wfSplitResizeBound) {
        window.__wfSplitResizeBound = true;
        let resizeTimer = 0;
        window.addEventListener("resize", () => {
          window.clearTimeout(resizeTimer);
          resizeTimer = window.setTimeout(() => {
            document.querySelectorAll(".split[data-split-key]").forEach(split => {
              const fallback = Number(split.dataset.splitDefault || 360);
              applySplitLeft(split, currentSplitLeft(split, fallback), { persist: false });
            });
          }, 50);
        });
      }
    }
    initResizableSplits();
    initializeWorkspace()
      .catch(err => { $("serverState").textContent = err.message; });
    setInterval(refreshJobs, 3000);
    setInterval(() => refreshVisualizersRealtime(), 1500);
  </script>
</body>
</html>
"""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the WorldFoundry FastVideo-style workspace UI.")
    parser.add_argument("--host", default=os.getenv("WORLDFOUNDRY_WORKSPACE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WORLDFOUNDRY_WORKSPACE_PORT", "7870") or "7870"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
