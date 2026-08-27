from __future__ import annotations

import inspect
import json
import os
import shutil
import socket
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io import file_sha256, load_serialized, resolve_data_path
from worldfoundry.core.io.paths import project_root
from worldfoundry.core.process import read_text_tail, run_logged_subprocess
from worldfoundry.runtime.assets import expand_worldfoundry_path


def _expand_value(value: Any) -> Any:
    if isinstance(value, str) and ("$WORLDFOUNDRY_" in value or "${WORLDFOUNDRY_" in value):
        return str(expand_worldfoundry_path(value))
    if isinstance(value, str) and value.startswith("models/"):
        return str(resolve_data_path(*Path(value).parts))
    if isinstance(value, Mapping):
        return {str(key): _expand_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    return value


def _existing_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else None


def _first_existing(items: Any) -> Path | None:
    if isinstance(items, (str, Path)):
        return _existing_path(str(items))
    if not isinstance(items, list):
        return None
    for item in items:
        found = _existing_path(item)
        if found is not None:
            return found
    return None


def _missing_required_paths(
    items: Any,
    *,
    repo_root: Path | None,
    checkpoint_path: Path | None,
) -> list[str]:
    if not isinstance(items, list):
        return []

    variables = {
        "repo_root": str(repo_root or ""),
        "checkpoint_path": str(checkpoint_path or ""),
    }
    missing: list[str] = []
    for item in items:
        label = ""
        base = "checkpoint"
        raw_path: Any = item
        if isinstance(item, Mapping):
            raw_path = item.get("path")
            label = str(item.get("id") or item.get("label") or "")
            base = str(item.get("base") or base)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        rendered = str(_expand_value(raw_path)).format(**variables)
        candidate = Path(rendered).expanduser()
        if not candidate.is_absolute():
            if base == "repo":
                candidate = (repo_root / candidate) if repo_root else candidate
            elif base == "cwd":
                candidate = candidate
            else:
                candidate = (checkpoint_path / candidate) if checkpoint_path else candidate
        if not candidate.exists():
            prefix = f"{label}: " if label else ""
            missing.append(f"{prefix}{rendered} -> {candidate}")
    return missing


def _format_command(items: list[str], variables: Mapping[str, Any]) -> list[str]:
    rendered: list[str] = []
    format_variables = {
        key: " ".join(str(part) for part in value) if isinstance(value, (list, tuple)) else str(value)
        for key, value in variables.items()
    }
    for item in items:
        placeholder = str(item).strip()
        if placeholder == "{pyramid_num_inference_steps_list}" and isinstance(
            variables.get("pyramid_num_inference_steps_list"), (list, tuple)
        ):
            rendered.extend(str(part) for part in variables["pyramid_num_inference_steps_list"])
            continue
        if str(item).strip() == "{torchrun}" and isinstance(variables.get("torchrun"), (list, tuple)):
            python = str(variables.get("python") or "")
            if python and (not rendered or rendered[-1] != python):
                rendered.append(python)
            rendered.extend(str(part) for part in variables["torchrun"])
            continue
        rendered.append(str(item).format(**format_variables))
    return rendered


def _absolute_cli_path(value: Any) -> Any:
    if isinstance(value, Path):
        return value.expanduser().resolve()
    if not isinstance(value, str) or not value.strip():
        return value
    stripped = value.strip()
    if "://" in stripped:
        return value
    return Path(stripped).expanduser().resolve()


def _normalize_cli_path_variables(variables: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(variables)
    for key, value in tuple(normalized.items()):
        if key in {
            "repo_root",
            "checkpoint_path",
            "output_path",
            "output_dir",
            "image_path",
            "video_path",
            "eval_dir",
            "vae",
            "text_encoder",
        }:
            normalized[key] = _absolute_cli_path(value)
    return normalized


def _with_cli_aliases(defaults: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    values = {str(key): value for key, value in extra.items()}
    alias_groups = (
        ("num_sampling_steps", "num_inference_steps", "sampling_steps", "steps", "infer_steps", "num_steps"),
        ("num_inference_steps", "num_sampling_steps", "sampling_steps", "steps", "infer_steps", "num_steps"),
        ("num_steps", "num_inference_steps", "sampling_steps", "steps", "infer_steps", "num_sampling_steps"),
        ("infer_steps", "num_inference_steps", "sampling_steps", "steps", "num_steps", "num_sampling_steps"),
        ("sample_steps", "num_inference_steps", "sampling_steps", "steps", "infer_steps", "num_steps"),
        ("video_length", "num_frames", "frames", "max_frames"),
    )
    for canonical, *aliases in alias_groups:
        if canonical not in defaults or canonical in values:
            continue
        for alias in aliases:
            if alias in values:
                values[canonical] = values[alias]
                break
    return values


def _torchrun_executable() -> tuple[str, str]:
    return ("-m", "torch.distributed.run")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _normalize_visible_devices(value: Any) -> str:
    if value in {None, ""}:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


@dataclass(frozen=True)
class RuntimeRequirementReport:
    missing: tuple[str, ...]
    repo_root: Path | None
    checkpoint_path: Path | None

    @property
    def ready(self) -> bool:
        return not self.missing


class OfficialVideoRuntime:
    """Data-backed runtime adapter for official video-family inference paths.

    The config decides whether a model uses an official CLI checkout, a diffusers
    pipeline, a Hugging Face transformers model, or an API endpoint. This class
    performs real execution for those backends and returns a structured blocked
    result when required local assets are absent.
    """

    def __init__(self, *, model_id: str, runtime_config_path: str, device: str = "cuda", **overrides: Any) -> None:
        self.model_id = model_id
        self.device = device
        self.runtime_config_path = runtime_config_path
        self.config = self._load_config(runtime_config_path)
        runtime = dict(self.config.get("runtime") or {})
        runtime.update({key: value for key, value in overrides.items() if value is not None})
        self.runtime = _expand_value(runtime)

    @staticmethod
    def _load_config(path: str) -> dict[str, Any]:
        payload = load_serialized(resolve_data_path(*Path(path).parts))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Official video runtime config must be a mapping: {path}")
        return dict(payload)

    @classmethod
    def from_model_id(cls, model_id: str, *, device: str = "cuda", **overrides: Any) -> "OfficialVideoRuntime":
        return cls(
            model_id=model_id,
            runtime_config_path=f"models/runtime/configs/video_official/{model_id}.yaml",
            device=device,
            **overrides,
        )

    def _requirement_report(self) -> RuntimeRequirementReport:
        missing: list[str] = []
        repo_sources = self.runtime.get("repo_root") or self.runtime.get("repo_root_candidates")
        checkpoint_sources = self.runtime.get("checkpoint_path") or self.runtime.get("checkpoint_candidates")
        repo_root = _first_existing(repo_sources)
        checkpoint_path = _first_existing(checkpoint_sources)

        kind = str(self.runtime.get("kind") or "")
        if kind == "official_cli" and repo_root is None:
            missing.append(
                "official repo checkout not found; checked "
                f"{repo_sources}"
            )
        checkpoint_optional = bool(self.runtime.get("checkpoint_optional", False))
        if (
            kind in {"official_cli", "diffusers_pipeline", "transformers_generation"}
            and checkpoint_path is None
            and not checkpoint_optional
        ):
            missing.append(
                "checkpoint/assets not found; checked "
                f"{checkpoint_sources}"
            )
        for path in _missing_required_paths(
            self.runtime.get("required_paths"),
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
        ):
            missing.append(f"required runtime path not found: {path}")
        if kind == "api_endpoint":
            env_name = str(self.runtime.get("api_key_env") or "")
            if env_name and not os.environ.get(env_name):
                missing.append(f"API key environment variable is not set: {env_name}")
        return RuntimeRequirementReport(tuple(missing), repo_root, checkpoint_path)

    def runtime_plan(self, *, output_path: str | Path | None = None, prompt: str = "") -> dict[str, Any]:
        report = self._requirement_report()
        defaults = self.runtime.get("defaults")
        default_variables = dict(defaults) if isinstance(defaults, Mapping) else {}
        variables = {
            "python": sys.executable,
            "torchrun": _torchrun_executable(),
            "master_port": _find_free_port(),
            "repo_root": report.repo_root or "",
            "checkpoint_path": report.checkpoint_path or "",
            "output_path": output_path or "",
            "output_dir": Path(output_path).parent if output_path else "",
            "prompt": prompt,
            "image_path": "",
            "video_path": "",
            "device": self.device,
            **default_variables,
        }
        variables = _normalize_cli_path_variables(variables)
        command = self.runtime.get("command")
        return {
            "model_id": self.model_id,
            "runtime": self.runtime.get("kind"),
            "backend_quality": self.runtime.get("backend_quality", "official_runtime_bridge"),
            "ready": report.ready,
            "missing": list(report.missing),
            "repo_root": str(report.repo_root) if report.repo_root else None,
            "checkpoint_path": str(report.checkpoint_path) if report.checkpoint_path else None,
            "command": _format_command(command, variables) if isinstance(command, list) else None,
            "notes": list(self.config.get("notes") or ()),
        }

    def _blocked_result(self, *, output_path: Path, prompt: str, missing: tuple[str, ...]) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path = output_path.with_suffix(output_path.suffix + ".runtime_plan.json")
        plan_path.write_text(
            json.dumps(self.runtime_plan(output_path=output_path, prompt=prompt), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "status": "blocked",
            "runtime": self.runtime.get("kind"),
            "backend_quality": "blocked_missing_runtime_requirements",
            "blocked_reasons": list(missing),
            "blocked_reason": "; ".join(missing),
            "plan_path": str(plan_path),
            "artifact_path": str(output_path),
            "metadata": {"runtime_config_path": self.runtime_config_path},
        }

    def generate(
        self,
        *,
        prompt: str,
        output_path: str | Path,
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        plan_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        output = Path(output_path).expanduser().resolve()
        report = self._requirement_report()
        if plan_only:
            output.parent.mkdir(parents=True, exist_ok=True)
            plan_path = output.with_suffix(output.suffix + ".runtime_plan.json")
            plan_path.write_text(
                json.dumps(self.runtime_plan(output_path=output, prompt=prompt), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return {
                "status": "blocked" if not report.ready else "planned",
                "runtime": self.runtime.get("kind"),
                "backend_quality": self.runtime.get("backend_quality", "official_runtime_plan"),
                "blocked_reasons": list(report.missing),
                "blocked_reason": "; ".join(report.missing) if report.missing else None,
                "plan_path": str(plan_path),
                "artifact_path": str(output),
            }
        if not report.ready:
            return self._blocked_result(output_path=output, prompt=prompt, missing=report.missing)

        kind = str(self.runtime.get("kind"))
        if kind == "official_cli":
            return self._run_official_cli(
                prompt=prompt,
                output_path=output,
                image_path=image_path,
                video_path=video_path,
                repo_root=report.repo_root,
                checkpoint_path=report.checkpoint_path,
                extra=kwargs,
            )
        if kind == "diffusers_pipeline":
            return self._run_diffusers(
                prompt=prompt,
                output_path=output,
                image_path=image_path,
                checkpoint_path=report.checkpoint_path,
                extra=kwargs,
            )
        if kind == "modelscope_text_to_video":
            return self._run_modelscope_text_to_video(
                prompt=prompt,
                output_path=output,
                checkpoint_path=report.checkpoint_path,
                extra=kwargs,
            )
        if kind == "transformers_generation":
            return self._run_transformers_text(
                prompt=prompt,
                output_path=output,
                checkpoint_path=report.checkpoint_path,
                extra=kwargs,
            )
        if kind == "api_endpoint":
            return self._blocked_result(
                output_path=output,
                prompt=prompt,
                missing=("API execution is configured but this adapter only validates API credentials in-tree.",),
            )
        raise ValueError(f"Unsupported official video runtime kind for {self.model_id}: {kind!r}")

    def _success_result(self, output: Path, *, metadata: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "runtime": self.runtime.get("kind"),
            "backend_quality": self.runtime.get("backend_quality", "official_runtime_bridge"),
            "artifact_path": str(output),
            "artifact_sha256": file_sha256(output) if output.is_file() else None,
            "metadata": dict(metadata),
        }

    @staticmethod
    def _materialize_cli_artifacts(
        produced: Path,
        output_path: Path,
        *,
        since: float,
    ) -> tuple[Path, tuple[Path, ...]]:
        """Copy fresh direct CLI outputs without changing their media type."""

        supported = {".mp4", ".mov", ".webm", ".gif", ".png", ".jpg", ".jpeg", ".wav", ".flac", ".mp3", ".aac"}
        candidates = [produced]
        candidates.extend(
            path
            for path in produced.parent.iterdir()
            if path.is_file()
            and path != produced
            and path.suffix.lower() in supported
            and path.stat().st_mtime >= since - 1.0
        )
        materialized: list[Path] = []
        primary: Path | None = None
        used_suffixes: set[str] = set()
        for candidate in candidates:
            suffix = candidate.suffix.lower() or output_path.suffix.lower()
            if suffix in used_suffixes:
                continue
            used_suffixes.add(suffix)
            target = output_path if suffix == output_path.suffix.lower() else output_path.with_suffix(suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            if candidate.resolve() != target.resolve():
                shutil.copy2(candidate, target)
            if primary is None:
                primary = target
            materialized.append(target)
        if primary is None:
            raise RuntimeError(f"official CLI produced no materializable artifact: {produced}")
        return primary, tuple(materialized)

    def _run_official_cli(
        self,
        *,
        prompt: str,
        output_path: Path,
        image_path: str | Path | None,
        video_path: str | Path | None,
        repo_root: Path | None,
        checkpoint_path: Path | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        command = self.runtime.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError(f"{self.model_id} official_cli runtime requires a command list.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        defaults = self.runtime.get("defaults")
        default_variables = dict(defaults) if isinstance(defaults, Mapping) else {}
        variables = {
            "python": sys.executable,
            "torchrun": _torchrun_executable(),
            "master_port": _find_free_port(),
            "repo_root": repo_root or "",
            "checkpoint_path": checkpoint_path or "",
            "output_path": output_path,
            "output_dir": output_path.parent,
            "prompt": prompt,
            "image_path": image_path or "",
            "video_path": video_path or extra.get("video_path") or "",
            "device": self.device,
            **default_variables,
            **_with_cli_aliases(default_variables, extra),
        }
        variables = _normalize_cli_path_variables(variables)
        rendered = _format_command(command, variables)
        cwd_template = self.runtime.get("cwd")
        cwd = str(repo_root) if repo_root else None
        if isinstance(cwd_template, str) and cwd_template.strip():
            cwd = str(Path(cwd_template.format(**{name: str(item) for name, item in variables.items()})).expanduser())
        env = os.environ.copy()
        src_root = project_root(__file__) / "worldfoundry"
        prepend_repo_root = bool(self.runtime.get("prepend_repo_root_to_pythonpath", True))
        env["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (
                str(repo_root) if repo_root and prepend_repo_root else "",
                str(src_root),
                env.get("PYTHONPATH", ""),
            )
            if item
        )
        visible_devices = ""
        for source in (extra, self.runtime):
            if not isinstance(source, Mapping):
                continue
            for key in ("cuda_visible_devices", "visible_devices", "cuda_devices", "gpu_ids"):
                visible_devices = _normalize_visible_devices(source.get(key))
                if visible_devices:
                    break
            if visible_devices:
                break
        if visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = visible_devices
        runtime_env = self.runtime.get("env")
        if isinstance(runtime_env, Mapping):
            env.update(
                {
                    str(key): str(value).format(**{name: str(item) for name, item in variables.items()})
                    for key, value in runtime_env.items()
                }
            )
        started_at = time.time()
        stdout_log_path = output_path.with_suffix(output_path.suffix + ".stdout.log")
        stderr_log_path = output_path.with_suffix(output_path.suffix + ".stderr.log")
        completed = run_logged_subprocess(
            rendered,
            stdout_path=stdout_log_path,
            stderr_path=stderr_log_path,
            cwd=cwd,
            env=env,
            start_new_session=False,
        )
        log_metadata = {
            "command": rendered,
            "stdout_log_path": str(stdout_log_path),
            "stderr_log_path": str(stderr_log_path),
        }
        if completed.returncode != 0:
            detail = read_text_tail(stderr_log_path) or read_text_tail(stdout_log_path)
            return {
                "status": "failed",
                "runtime": "official_cli",
                "backend_quality": "official_cli_failed",
                "artifact_path": str(output_path),
                "error": (
                    f"official CLI exited with code {completed.returncode}; see {stderr_log_path}: {detail}"
                ),
                "metadata": log_metadata,
            }
        produced = self._resolve_produced_artifact(
            output_path,
            search_dirs=(variables.get("output_dir"), output_path.parent),
            since=started_at,
        )
        if produced is None:
            return {
                "status": "failed",
                "runtime": "official_cli",
                "backend_quality": "official_cli_no_artifact",
                "artifact_path": str(output_path),
                "error": f"official CLI completed but no artifact was found at or near {output_path}",
                "metadata": log_metadata,
            }
        primary, artifacts = self._materialize_cli_artifacts(produced, output_path, since=started_at)
        result = self._success_result(primary, metadata=log_metadata)
        result["artifact_paths"] = [str(path) for path in artifacts]
        return result

    def _resolve_produced_artifact(
        self,
        output_path: Path,
        *,
        search_dirs: tuple[Any, ...] = (),
        since: float | None = None,
    ) -> Path | None:
        def _fresh(path: Path) -> bool:
            if since is None:
                return True
            try:
                return path.stat().st_mtime >= since - 1.0
            except OSError:
                return False

        if output_path.is_file() and _fresh(output_path):
            return output_path
        suffixes = {".mp4", ".mov", ".webm", ".gif", ".png", ".jpg", ".jpeg", ".wav", ".flac", ".mp3", ".aac"}
        roots: list[Path] = []
        for raw_root in (*search_dirs, output_path.parent):
            if raw_root in {None, ""}:
                continue
            root = Path(raw_root).expanduser()
            try:
                root = root.resolve()
            except OSError:
                pass
            if root.is_file():
                root = root.parent
            if root.exists() and root not in roots:
                roots.append(root)

        candidates: list[Path] = []
        for root in roots:
            candidates.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in suffixes and _fresh(path)
            )
        candidates = sorted(
            set(candidates),
            key=lambda path: (path.stat().st_mtime, path.stat().st_size),
            reverse=True,
        )
        preferred_patterns = self.runtime.get("preferred_artifact_patterns")
        if isinstance(preferred_patterns, list):
            for pattern in preferred_patterns:
                matches = [
                    path
                    for path in candidates
                    if isinstance(pattern, str) and (fnmatch(path.name, pattern) or fnmatch(str(path), pattern))
                ]
                if matches:
                    return matches[0]
        if candidates:
            return candidates[0]
        if output_path.is_file():
            return output_path
        stale_candidates: list[Path] = []
        for root in roots:
            stale_candidates.extend(
                path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
            )
        candidates = sorted(
            set(stale_candidates),
            key=lambda path: (path.stat().st_mtime, path.stat().st_size),
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _run_diffusers(
        self,
        *,
        prompt: str,
        output_path: Path,
        image_path: str | Path | None,
        checkpoint_path: Path | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        import importlib

        import imageio
        import torch

        module_name, _, attr_name = str(self.runtime.get("pipeline_target") or "diffusers:DiffusionPipeline").partition(":")
        module = importlib.import_module(module_name)
        pipeline_cls = getattr(module, attr_name or "DiffusionPipeline")
        dtype_name = str(extra.get("torch_dtype") or self.runtime.get("torch_dtype") or "float16")
        torch_dtype = getattr(torch, dtype_name)
        pipe = pipeline_cls.from_pretrained(str(checkpoint_path), torch_dtype=torch_dtype)
        enable_model_cpu_offload = bool(self.runtime.get("enable_model_cpu_offload", False))
        if enable_model_cpu_offload:
            offload = getattr(pipe, "enable_model_cpu_offload", None)
            if not callable(offload):
                raise TypeError(
                    f"{pipeline_cls.__name__} does not support requested model CPU offload"
                )
            offload(device=self.device)
        elif hasattr(pipe, "to"):
            pipe = pipe.to(self.device)
        if bool(self.runtime.get("enable_vae_tiling", False)):
            vae = getattr(pipe, "vae", None)
            enable_tiling = getattr(vae, "enable_tiling", None)
            if not callable(enable_tiling):
                raise TypeError(
                    f"{pipeline_cls.__name__} does not support requested VAE tiling"
                )
            enable_tiling()
        call_kwargs = dict(self.runtime.get("call_kwargs") or {})
        call_kwargs.update(extra)
        # Some Diffusers pipelines condition on ``target_fps`` while others do
        # not accept an FPS argument at all. Preserve the requested playback
        # rate before signature filtering so the encoded artifact still honors
        # the Workspace request.
        output_fps = int(
            call_kwargs.get("fps")
            or call_kwargs.get("target_fps")
            or self.runtime.get("fps")
            or 16
        )
        seed = call_kwargs.pop("seed", None)
        signature = inspect.signature(pipe.__call__)
        parameters = signature.parameters
        accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        # ``fps`` is the Workspace playback control. Pipelines such as
        # I2VGen-XL call the corresponding generation micro-condition
        # ``target_fps``; keep conditioning and encoded playback in sync.
        if extra.get("fps") is not None and "target_fps" in parameters and "target_fps" not in extra:
            call_kwargs["target_fps"] = int(extra["fps"])
        if seed is not None and (accepts_kwargs or "generator" in parameters):
            generator_device = self.device if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
            call_kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(int(seed))
        if image_path:
            from PIL import Image

            if accepts_kwargs or "image" in parameters:
                call_kwargs["image"] = Image.open(image_path).convert("RGB")
        if not accepts_kwargs:
            call_kwargs = {
                key: value
                for key, value in call_kwargs.items()
                if key in parameters
            }
        autocast_dtype_name = self.runtime.get("autocast_dtype")
        if autocast_dtype_name and str(self.device).startswith("cuda"):
            autocast_context = torch.autocast(
                device_type="cuda",
                dtype=getattr(torch, str(autocast_dtype_name)),
                cache_enabled=bool(self.runtime.get("autocast_cache_enabled", False)),
            )
        else:
            autocast_context = nullcontext()
        with autocast_context:
            result = pipe(prompt=prompt, **call_kwargs)
        frames = getattr(result, "frames", None) or getattr(result, "videos", None) or result[0]
        if frames and isinstance(frames, list) and frames and isinstance(frames[0], list):
            frames = frames[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), frames, fps=output_fps)
        return self._success_result(output_path, metadata={"pipeline_target": self.runtime.get("pipeline_target")})

    def _run_modelscope_text_to_video(
        self,
        *,
        prompt: str,
        output_path: Path,
        checkpoint_path: Path | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        import numpy as np
        import torch
        from modelscope.outputs import OutputKeys
        from modelscope.pipelines import pipeline

        seed = int(extra.get("seed") or self.runtime.get("seed") or 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        original_torch_load = torch.load

        def _torch_load_legacy_default(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = _torch_load_legacy_default  # type: ignore[assignment]
        try:
            pipe = pipeline("text-to-video-synthesis", str(checkpoint_path))
        finally:
            torch.load = original_torch_load  # type: ignore[assignment]
        try:
            import imageio.v2 as imageio
            import torchvision

            if not hasattr(torchvision.io, "write_video"):

                def _write_video_imageio(
                    filename: str,
                    video_array: Any,
                    fps: int = 8,
                    video_codec: str | None = None,
                    options: Mapping[str, Any] | None = None,
                    **_: Any,
                ) -> None:
                    del video_codec, options
                    frames = video_array.detach().cpu().numpy() if hasattr(video_array, "detach") else video_array
                    imageio.mimsave(filename, frames, fps=fps)

                torchvision.io.write_video = _write_video_imageio  # type: ignore[attr-defined]
        except Exception:
            pass
        clip_model = getattr(getattr(getattr(pipe, "model", None), "clip_encoder", None), "model", None)
        if clip_model is not None and hasattr(clip_model, "attn_mask"):
            clip_model.attn_mask = None
        result = pipe({"text": prompt}, output_video=str(output_path))
        produced = result.get(OutputKeys.OUTPUT_VIDEO) if isinstance(result, Mapping) else None
        produced_path = Path(produced).expanduser() if produced else output_path
        if not produced_path.is_file():
            produced_path = self._resolve_produced_artifact(output_path) or produced_path
        if not produced_path.is_file():
            return {
                "status": "failed",
                "runtime": "modelscope_text_to_video",
                "backend_quality": "modelscope_no_artifact",
                "artifact_path": str(output_path),
                "error": f"ModelScope pipeline completed but no artifact was found at {output_path}",
            }
        if produced_path != output_path:
            shutil.copy2(produced_path, output_path)
        return self._success_result(
            output_path,
            metadata={
                "pipeline_target": "modelscope:text-to-video-synthesis",
                "seed": seed,
            },
        )

    def _run_transformers_text(
        self,
        *,
        prompt: str,
        output_path: Path,
        checkpoint_path: Path | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        trust_remote_code = bool(self.runtime.get("trust_remote_code", True))
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(str(checkpoint_path), trust_remote_code=trust_remote_code)
        except Exception:
            processor = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            trust_remote_code=trust_remote_code,
            torch_dtype=getattr(torch, str(self.runtime.get("torch_dtype") or "float16")),
            device_map=self.runtime.get("device_map") or "auto",
        )
        inputs = processor(prompt, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=int(extra.get("max_new_tokens") or 128))
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"prompt": prompt, "text": text}, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._success_result(output_path, metadata={"checkpoint_path": str(checkpoint_path)})


__all__ = ["OfficialVideoRuntime"]
