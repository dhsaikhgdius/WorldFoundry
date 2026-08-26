from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.evaluation.utils import REPO_ROOT
from worldfoundry.core.process import run_logged_subprocess, synthesis_timeout_seconds
from worldfoundry.synthesis.base_synthesis import BaseSynthesis
from worldfoundry.studio.visualization.core.geometry import depth_to_world_points


@dataclass(frozen=True)
class ThreeDFourDRuntimeSpec:
    model_id: str
    display_name: str
    repo_names: tuple[str, ...]
    entrypoint: str
    command_kind: str
    artifact_kind: str
    artifact_filename: str
    required_input: str = "prompt_or_image"
    default_config: str | None = None
    source_repo_url: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


_CAMERA_INPUT_SCHEMA = {
    "prompt": True,
    "image": True,
    "video": True,
    "actions": ["camera_path", "camera_pose_sequence", "navigation_path"],
}


THREE_D_FOUR_D_RUNTIME_SPECS: Mapping[str, ThreeDFourDRuntimeSpec] = {
    "4d-gs": ThreeDFourDRuntimeSpec(
        model_id="4d-gs",
        display_name="4D-GS",
        repo_names=("4DGaussians",),
        entrypoint="",
        command_kind="unsupported_inference",
        artifact_kind="generated_4d_scene",
        artifact_filename="4dgs_output",
        required_input="dataset",
        source_repo_url="https://github.com/hustvl/4DGaussians",
        input_schema={
            "prompt": False,
            "image": False,
            "video": True,
            "actions": ["dynamic_scene_dataset", "render_path"],
        },
        notes=("Official reconstruction uses a training entrypoint; disabled in the infer-only runtime registry.",),
    ),
    "lagernvs": ThreeDFourDRuntimeSpec(
        model_id="lagernvs",
        display_name="LagrNVS",
        repo_names=("lagernvs",),
        entrypoint="minimal_inference.py",
        command_kind="lagernvs_minimal",
        artifact_kind="generated_video",
        artifact_filename="lagernvs.mp4",
        required_input="image",
        source_repo_url="https://github.com/facebookresearch/lagernvs",
        input_schema={"prompt": False, "image": True, "video": False, "actions": ["camera_path"]},
    ),
    "monst3r": ThreeDFourDRuntimeSpec(
        model_id="monst3r",
        display_name="MonST3R",
        repo_names=("monst3r", "MonST3R"),
        entrypoint="demo.py",
        command_kind="monst3r_demo",
        artifact_kind="generated_4d_scene",
        artifact_filename="monst3r.glb",
        required_input="image_or_video",
        source_repo_url="https://github.com/Junyi42/monst3r",
        input_schema={"prompt": False, "image": True, "video": True, "actions": ["camera_path"]},
        notes=("Runs the in-tree MonST3R demo.py infer path and exports a GLB scene artifact.",),
    ),
    "mvdiffusion": ThreeDFourDRuntimeSpec(
        model_id="mvdiffusion",
        display_name="MVDiffusion",
        repo_names=("MVDiffusion", "mvdiffusion"),
        entrypoint="demo.py",
        command_kind="mvdiffusion_demo",
        artifact_kind="generated_3d_asset",
        artifact_filename="mvdiffusion.png",
        required_input="prompt_or_image",
        source_repo_url="https://github.com/Tangshitao/MVDiffusion",
        input_schema={"prompt": True, "image": True, "video": False, "actions": ["camera_views"]},
    ),
    "shape-of-motion": ThreeDFourDRuntimeSpec(
        model_id="shape-of-motion",
        display_name="Shape of Motion",
        repo_names=("shape-of-motion", "Shape-of-Motion"),
        entrypoint="",
        command_kind="shape_of_motion_preview",
        artifact_kind="generated_4d_scene",
        artifact_filename="shape_of_motion_scene.ply",
        required_input="dataset",
        source_repo_url="https://github.com/vye16/shape-of-motion",
        input_schema={"prompt": False, "image": False, "video": True, "actions": ["preprocessed_data_dir", "camera_path"]},
    ),
    "stable-virtual-camera": ThreeDFourDRuntimeSpec(
        model_id="stable-virtual-camera",
        display_name="Stable Virtual Camera",
        repo_names=("stable-virtual-camera",),
        entrypoint="demo.py",
        command_kind="stable_virtual_camera_demo",
        artifact_kind="generated_video",
        artifact_filename="stable_virtual_camera.mp4",
        required_input="image",
        source_repo_url="https://github.com/Stability-AI/stable-virtual-camera",
        input_schema={"prompt": False, "image": True, "video": False, "actions": ["camera_trajectory"]},
    ),
    "wonderjourney": ThreeDFourDRuntimeSpec(
        model_id="wonderjourney",
        display_name="WonderJourney",
        repo_names=("WonderJourney", "wonderjourney"),
        entrypoint="run.py",
        command_kind="wonderjourney_run",
        artifact_kind="generated_world",
        artifact_filename="wonderjourney.mp4",
        required_input="config",
        default_config="config/village.yaml",
        source_repo_url="https://github.com/KovenYu/WonderJourney",
        input_schema=_CAMERA_INPUT_SCHEMA,
    ),
    "wonderworld": ThreeDFourDRuntimeSpec(
        model_id="wonderworld",
        display_name="WonderWorld",
        repo_names=("WonderWorld", "wonderworld"),
        entrypoint="run.py",
        command_kind="wonderworld_run",
        artifact_kind="generated_world",
        artifact_filename="wonderworld.splat",
        required_input="config",
        default_config="config/inference.yaml",
        source_repo_url="https://github.com/KovenYu/WonderWorld",
        input_schema=_CAMERA_INPUT_SCHEMA,
    ),
    "worldgen": ThreeDFourDRuntimeSpec(
        model_id="worldgen",
        display_name="WorldGen",
        repo_names=("WorldGen", "worldgen"),
        entrypoint="inference.py",
        command_kind="worldgen_inference",
        artifact_kind="generated_3d_asset",
        artifact_filename="worldgen.ply",
        required_input="prompt_or_image",
        source_repo_url="https://github.com/ZiYang-xie/WorldGen",
        input_schema={"prompt": True, "image": True, "video": False, "actions": ["camera_path"]},
    ),
}


IN_TREE_RUNTIME_DIRS: Mapping[str, tuple[str, ...]] = {
    "4d-gs": ("base_models", "three_dimensions", "general_3d", "four_d_gaussians", "four_d_gaussians_runtime"),
    "lagernvs": ("base_models", "three_dimensions", "general_3d", "lagernvs", "lagernvs_runtime"),
    "monst3r": ("base_models", "three_dimensions", "general_3d", "monst3r"),
    "mvdiffusion": ("base_models", "three_dimensions", "general_3d", "mvdiffusion", "mvdiffusion_runtime"),
    "shape-of-motion": (
        "base_models",
        "three_dimensions",
        "general_3d",
        "shape_of_motion",
        "shape_of_motion_runtime",
    ),
    "stable-virtual-camera": (
        "base_models",
        "three_dimensions",
        "general_3d",
        "stable_virtual_camera",
        "stable_virtual_camera_runtime",
    ),
    "wonderjourney": ("synthesis", "visual_generation", "wonderjourney", "wonderjourney_runtime"),
    "wonderworld": ("synthesis", "visual_generation", "wonderworld", "wonderworld_runtime"),
    "worldgen": ("synthesis", "visual_generation", "worldgen", "worldgen_runtime"),
}


class ThreeDFourDRuntimeSynthesis(BaseSynthesis):
    """Subprocess adapter for 3D/4D repository inference entrypoints."""

    MODEL_ID: str | None = None

    def __init__(
        self,
        *,
        spec: ThreeDFourDRuntimeSpec,
        source_root: Path | None,
        device: str = "cuda",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.profile = spec
        self.model_id = spec.model_id
        self.model_name = spec.display_name
        self.source_root = source_root
        self.device = device
        self.options = dict(options or {})

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "ThreeDFourDRuntimeSynthesis":
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["source_root"] = str(pretrained_model_path)
        options.update(kwargs)
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID or "")
        if not resolved_model_id:
            raise ValueError("ThreeDFourDRuntimeSynthesis requires model_id/profile_id.")
        spec = three_d_four_d_runtime_spec(resolved_model_id)
        profile = load_runtime_profile(resolved_model_id, check_conda_env_exists=False)
        source_root = _resolve_source_root(spec, options, profile.source_repos)
        return cls(
            spec=spec,
            source_root=source_root,
            device=str(device or options.get("device") or "cuda"),
            options=options,
        )

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        timeout_seconds: int = 21600,
        plan_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix=f"{_safe_name(self.model_id)}_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        output = _resolve_output(output_path, run_dir, self.spec.artifact_filename)
        output = _coerce_model_artifact_output(output, self.spec)
        media = _materialize_media(images=images, video=video, run_dir=run_dir)
        context = {
            "prompt": prompt,
            "images": media["images"],
            "video": media["video"],
            "interactions": _jsonable(interactions or []),
            "run_dir": str(run_dir),
            "output_path": str(output),
            "output_dir": str(output.parent),
            "fps": fps,
            "device": self.device,
            "options": _jsonable({**self.options, **kwargs}),
            "source_root": str(self.source_root) if self.source_root is not None else "",
            "entrypoint": str(self.entrypoint) if self.entrypoint is not None else "",
        }
        command = self._build_command(context, kwargs)
        plan_path = run_dir / "subprocess_runtime_plan.json"
        plan_payload = {
            "schema_version": "worldfoundry-three-d-four-d-runtime-plan",
            "model_id": self.model_id,
            "display_name": self.model_name,
            "source_repo_url": self.spec.source_repo_url,
            "source_root": context["source_root"],
            "entrypoint": context["entrypoint"],
            "command": command,
            "context": context,
        }
        _write_json(plan_path, plan_payload)
        if plan_only:
            requested_plan_path = _plan_output_path(output, self.model_id)
            if requested_plan_path != plan_path:
                _write_json(requested_plan_path, plan_payload)
            return self._prepared(run_dir, requested_plan_path, command)

        if self.spec.command_kind == "shape_of_motion_preview":
            preview_path = _maybe_export_shape_of_motion_preview(output.parent, {**self.options, **kwargs})
            if preview_path is not None:
                if preview_path.resolve() != output.resolve():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(preview_path, output)
                    preview_path = output
                return {
                    "status": "succeeded",
                    "model_id": self.model_id,
                    "artifact_kind": self.spec.artifact_kind,
                    "artifact_path": str(preview_path),
                    "artifact_sha256": _sha256(preview_path) if preview_path.is_file() else None,
                    "run_dir": str(run_dir),
                    "plan_path": str(plan_path),
                    "command": command,
                    "runtime": "worldfoundry.three_d_four_d.shape_of_motion_preview",
                    "backend_quality": "official_preprocessed_preview",
                    "metadata": plan_payload,
                }

        missing = self._missing_runtime_requirements(context)
        if missing:
            return {
                "status": "failed",
                "model_id": self.model_id,
                "artifact_kind": self.spec.artifact_kind,
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "command": command,
                "runtime": "worldfoundry.three_d_four_d.subprocess",
                "backend_quality": "model_entrypoint",
                "error": "; ".join(missing),
                "metadata": plan_payload,
            }

        log_path = run_dir / "subprocess_runtime.log"
        stderr_path = run_dir / "subprocess_runtime.stderr.log"
        env = self._subprocess_env(kwargs)
        try:
            completed = run_logged_subprocess(
                command,
                stdout_path=log_path,
                stderr_path=stderr_path,
                cwd=str(self.source_root),
                env=env,
                timeout=synthesis_timeout_seconds(default=float(timeout_seconds) if timeout_seconds else None),
            )
        except Exception as exc:  # pragma: no cover - runtime/environment dependent.
            return {
                "status": "failed",
                "model_id": self.model_id,
                "artifact_kind": self.spec.artifact_kind,
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "command": command,
                "runtime": "worldfoundry.three_d_four_d.subprocess",
                "backend_quality": "model_entrypoint",
                "error": f"{type(exc).__name__}: {exc}",
                "metadata": plan_payload,
            }
        if completed.returncode != 0:
            return {
                "status": "failed",
                "model_id": self.model_id,
                "artifact_kind": self.spec.artifact_kind,
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "command": command,
                "runtime": "worldfoundry.three_d_four_d.subprocess",
                "backend_quality": "model_entrypoint",
                "error": f"runtime exited with code {completed.returncode}; see {log_path}",
                "metadata": {**plan_payload, "log_path": str(log_path)},
            }

        artifact_path = _select_artifact(output, run_dir, extra_dirs=_extra_artifact_dirs(self.spec, self.source_root))
        if artifact_path.exists() and artifact_path.is_file() and artifact_path.resolve() != output.resolve():
            output.parent.mkdir(parents=True, exist_ok=True)
            target = output
            if artifact_path.suffix.lower() and artifact_path.suffix.lower() != output.suffix.lower():
                target = output.with_suffix(artifact_path.suffix)
            if artifact_path.resolve() != target.resolve():
                shutil.copy2(artifact_path, target)
                artifact_path = target
        visualization_path = None
        if self.spec.command_kind == "monst3r_demo":
            seq_name = str(kwargs.get("seq_name", self.options.get("seq_name")) or "worldfoundry")
            confidence_threshold = float(
                kwargs.get("min_conf_thr", self.options.get("min_conf_thr", 1.1))
            )
            visualization_path = _build_monst3r_rerun_recording(
                output.parent / seq_name,
                confidence_threshold=confidence_threshold,
            )
        result = {
            "status": "succeeded",
            "model_id": self.model_id,
            "artifact_kind": self.spec.artifact_kind,
            "artifact_path": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path) if artifact_path.is_file() else None,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "command": command,
            "runtime": "worldfoundry.three_d_four_d.subprocess",
            "backend_quality": "model_entrypoint",
            "metadata": {**plan_payload, "log_path": str(log_path)},
        }
        if visualization_path is not None:
            result["visualization_artifact_path"] = str(visualization_path)
            result["metadata"]["rrd_path"] = str(visualization_path)
        return result

    @property
    def entrypoint(self) -> Path | None:
        return None if self.source_root is None else (self.source_root / self.spec.entrypoint).resolve()

    def _prepared(self, run_dir: Path, plan_path: Path, command: Sequence[str]) -> dict[str, Any]:
        return {
            "status": "prepared",
            "model_id": self.model_id,
            "artifact_kind": "runtime_profile_plan",
            "artifact_path": str(plan_path),
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "command": list(command),
            "runtime": "worldfoundry.three_d_four_d.subprocess.plan",
            "backend_quality": "plan",
            "metadata": {
                "source_root": str(self.source_root) if self.source_root is not None else "",
                "entrypoint": str(self.entrypoint) if self.entrypoint is not None else "",
            },
        }

    def _missing_runtime_requirements(self, context: Mapping[str, Any]) -> list[str]:
        missing: list[str] = []
        if self.spec.command_kind == "unsupported_inference":
            missing.append(f"{self.spec.display_name} does not expose an infer-only WorldFoundry runtime yet")
        if self.source_root is None:
            missing.append(
                "runtime source checkout missing; set source_root or WORLDFOUNDRY_THREE_D_FOUR_D_REPOS_ROOT "
                f"for {self.spec.source_repo_url}"
            )
        elif not self.source_root.is_dir():
            missing.append(f"runtime source root does not exist: {self.source_root}")
        if (
            self.spec.command_kind not in {"unsupported_inference", "shape_of_motion_preview"}
            and (self.entrypoint is None or not self.entrypoint.is_file())
        ):
            missing.append(f"runtime entrypoint missing: {self.spec.entrypoint}")
        if self.spec.required_input == "prompt" and not str(context.get("prompt") or "").strip():
            missing.append("prompt input required")
        if self.spec.required_input == "image" and not context.get("images"):
            missing.append("image input required")
        if self.spec.required_input == "video" and not context.get("video"):
            missing.append("video input required")
        if self.spec.required_input == "image_or_video" and not (context.get("images") or context.get("video")):
            missing.append("image or video input required")
        if self.spec.required_input == "prompt_or_image" and not (
            str(context.get("prompt") or "").strip() or context.get("images")
        ):
            missing.append("prompt or image input required")
        if self.spec.required_input == "dataset":
            options = context.get("options") if isinstance(context.get("options"), Mapping) else {}
            data_dir = (
                options.get("data_dir")
                or options.get("source_path")
                or options.get("dataset_path")
                or options.get("data_root")
            )
            if not str(data_dir or "").strip():
                missing.append("dataset source path required; pass data_dir/source_path/dataset_path in runtime options")
        return missing

    def _subprocess_env(self, options: Mapping[str, Any] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        options = options or {}
        if self.source_root is not None:
            existing = env.get("PYTHONPATH", "")
            paths = []
            if self.spec.model_id not in {"wonderjourney", "wonderworld"}:
                paths.append(Path(__file__).resolve().parent / "shims")
            paths.append(self.source_root)
            for candidate in (
                self.source_root / "src",
                self.source_root / "submodules" / "DA-2" / "src",
                self.source_root / "submodules" / "ml-sharp" / "src",
                self.source_root / "submodules" / "viser" / "src",
                self.source_root / "third_party" / "dust3r",
                self.source_root / "third_party" / "sam2",
            ):
                if candidate.is_dir():
                    paths.append(candidate)
            prefix = os.pathsep.join(str(path) for path in paths)
            env["PYTHONPATH"] = f"{prefix}{os.pathsep}{existing}" if existing else prefix
        env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        if self.device:
            env.setdefault("WORLDFOUNDRY_DEVICE", self.device)
        gpu = options.get("gpu", self.options.get("gpu"))
        if gpu is not None and str(gpu).strip():
            env["CUDA_VISIBLE_DEVICES"] = str(gpu).strip()
        if self.spec.model_id == "monst3r":
            raft_model = _monst3r_raft_model_path()
            if raft_model:
                env.setdefault("WORLDFOUNDRY_MONST3R_RAFT_MODEL", str(raft_model))
            sam2_checkpoint = _monst3r_sam2_checkpoint_path()
            if sam2_checkpoint:
                env.setdefault("WORLDFOUNDRY_MONST3R_SAM2_CHECKPOINT", str(sam2_checkpoint))
        existing_da2_model_path = env.get("WORLDFOUNDRY_DEPTH_ANYTHING2_MODEL_PATH") or env.get("WORLDFOUNDRY_DA2_MODEL_PATH")
        if existing_da2_model_path and not _has_depth_anything2_checkpoint(existing_da2_model_path):
            env.pop("WORLDFOUNDRY_DEPTH_ANYTHING2_MODEL_PATH", None)
            env.pop("WORLDFOUNDRY_DA2_MODEL_PATH", None)
            existing_da2_model_path = ""
        da2_model_path = existing_da2_model_path or _depth_anything2_model_path()
        if da2_model_path:
            env["WORLDFOUNDRY_DEPTH_ANYTHING2_MODEL_PATH"] = str(da2_model_path)
            env["WORLDFOUNDRY_DA2_MODEL_PATH"] = str(da2_model_path)
        ckpt_root = _ckpt_root()
        env.setdefault("TORCH_HOME", str(ckpt_root / "torch_hub"))
        for offline_key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            env.pop(offline_key, None)
        hf_home_candidates = [
            Path(str(env.get("WORLDFOUNDRY_HFD_ROOT") or "")).expanduser(),
            Path(str(env.get("HF_HOME") or "")).expanduser(),
            ckpt_root / "hfd",
            ckpt_root / "huggingface",
        ]
        hf_home = next((candidate for candidate in hf_home_candidates if str(candidate) != "." and candidate.exists()), None)
        if hf_home is not None:
            env["HF_HOME"] = str(hf_home)
            env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
            env["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
        env.setdefault("WORLDFOUNDRY_CKPT_DIR", str(ckpt_root))
        return env

    def _build_command(self, context: Mapping[str, Any], options: Mapping[str, Any]) -> list[str]:
        python = str(options.get("python_executable") or sys.executable)
        if os.getenv("WORLDFOUNDRY_STUDIO_CONDA_CHILD", "").strip().lower() in {"1", "true", "yes", "on"}:
            python = sys.executable
        entrypoint = str(context.get("entrypoint") or self.spec.entrypoint)
        prompt = str(context.get("prompt") or "")
        output_path = str(context["output_path"])
        output_dir = str(context["output_dir"])
        run_dir = str(context["run_dir"])
        images = [str(item) for item in context.get("images") or []]
        video = str(context.get("video") or "")
        seed = str(options.get("seed", self.options.get("seed", 1234)))
        gpu = str(options.get("gpu", self.options.get("gpu", "0")))
        config = str(options.get("config") or self.options.get("config") or self.spec.default_config or "")

        if self.spec.command_kind == "unsupported_inference":
            return [python, "-c", f"raise SystemExit({self.spec.display_name!r} + ' has no infer-only runtime')"]
        if self.spec.command_kind == "shape_of_motion_preview":
            data_dir = (
                options.get("data_dir")
                or options.get("source_path")
                or options.get("dataset_path")
                or options.get("data_root")
                or ""
            )
            return [
                python,
                "-m",
                "worldfoundry.synthesis.visual_generation.three_d_four_d.runtime",
                "--shape-of-motion-preview",
                "--data-dir",
                str(data_dir),
                "--output",
                output_path,
            ]
        if self.spec.command_kind == "lagernvs_minimal":
            length = str(options.get("video_length", self.options.get("video_length", 100)))
            model_repo = _lagernvs_model_repo(
                str(options.get("model_repo", self.options.get("model_repo", "facebook/lagernvs_general_512")))
            )
            image_args = _expand_image_inputs(images)
            return [
                python,
                entrypoint,
                "--images",
                *image_args,
                "--video_length",
                length,
                "--output",
                output_path,
                "--model_repo",
                model_repo,
            ]
        if self.spec.command_kind == "stable_virtual_camera_demo":
            data_dir = _stable_virtual_camera_image_dir(images, Path(run_dir))
            version = str(options.get("version", self.options.get("version", "1.1")))
            task = str(options.get("task", self.options.get("task", "img2trajvid_s-prob")))
            command = [
                python,
                entrypoint,
                "--data_path",
                data_dir,
                "--version",
                version,
                "--task",
                task,
                "--save_subdir",
                Path(output_dir).name,
                "--seed",
                seed,
            ]
            for key in ("pretrained_model_name_or_path", "weight_name", "H", "W", "T", "num_steps"):
                value = options.get(key, self.options.get(key))
                if value is None:
                    value = options.get(key.lower(), self.options.get(key.lower()))
                if value is not None:
                    command.extend([f"--{key}", str(value)])
            return command
        if self.spec.command_kind == "worldgen_inference":
            command = [python, entrypoint, "--prompt", prompt or "a 3D scene", "--output_dir", output_dir]
            expanded_images = _expand_image_inputs(images)
            if expanded_images:
                command.extend(["--image", expanded_images[0]])
            if not _bool_option(options, self.options, "viewer"):
                command.append("--no_viewer")
            if _bool_option(options, self.options, "use_sharp"):
                command.append("--use_sharp")
            if _bool_option(options, self.options, "return_mesh"):
                command.append("--return_mesh")
            if _bool_option(options, self.options, "save_scene", default=True):
                command.append("--save_scene")
            if _bool_option(options, self.options, "low_vram"):
                command.append("--low_vram")
            return command
        if self.spec.command_kind in {"wonderjourney_run", "wonderworld_run"}:
            config = _prepare_wonder_config(
                self.spec.command_kind,
                self.source_root,
                config,
                Path(output_dir),
                options,
                prompt=prompt,
                images=images,
            )
            command = [python, entrypoint, "--example_config", config]
            return command
        if self.spec.command_kind == "monst3r_demo":
            command = [
                python,
                entrypoint,
                "--output_dir",
                output_dir,
                "--device",
                str(context.get("device") or self.device or "cuda"),
                "--save_name",
                Path(output_path).stem or "monst3r_scene",
            ]
            if video:
                command.extend(["--input_dir", video])
            elif images:
                command.extend(["--input_dir", _monst3r_input_dir(images, Path(run_dir))])
            weights = options.get("weights", self.options.get("weights"))
            if weights:
                command.extend(["--weights", str(weights)])
            model_name = options.get("model_name", self.options.get("model_name"))
            if model_name:
                command.extend(["--model_name", str(model_name)])
            image_size = options.get("image_size", self.options.get("image_size"))
            if image_size is not None:
                command.extend(["--image_size", str(image_size)])
            seq_name = options.get("seq_name", self.options.get("seq_name"))
            command.extend(["--seq_name", str(seq_name or "worldfoundry")])
            num_frames = options.get("num_frames", self.options.get("num_frames"))
            if num_frames is not None:
                command.extend(["--num_frames", str(num_frames)])
            batch_size = options.get("batch_size", self.options.get("batch_size"))
            if batch_size is not None:
                command.extend(["--batch_size", str(batch_size)])
            if context.get("fps") is not None:
                command.extend(["--fps", str(context.get("fps"))])
            for key in ("not_batchify", "real_time", "window_wise", "silent", "skip_pair_dynamic_mask"):
                if _bool_option(options, self.options, key):
                    command.append(f"--{key}")
            for key in (
                "niter",
                "scenegraph_type",
                "winsize",
                "refid",
                "min_conf_thr",
                "cam_size",
                "temporal_smoothing_weight",
                "translation_weight",
                "flow_loss_weight",
                "flow_loss_start_iter",
                "flow_loss_threshold",
            ):
                value = options.get(key, self.options.get(key))
                if value is not None:
                    command.extend([f"--{key}", str(value)])
            window_size = options.get("window_size", self.options.get("window_size"))
            if window_size is not None:
                command.extend(["--window_size", str(window_size)])
            window_overlap_ratio = options.get("window_overlap_ratio", self.options.get("window_overlap_ratio"))
            if window_overlap_ratio is not None:
                command.extend(["--window_overlap_ratio", str(window_overlap_ratio)])
            return command
        if self.spec.command_kind == "mvdiffusion_demo":
            command = [python, entrypoint]
            if prompt:
                command.extend(["--text", prompt])
            if images:
                command.extend(["--image_path", images[0]])
            text_path = options.get("text_path", self.options.get("text_path"))
            if text_path:
                command.extend(["--text_path", str(text_path)])
            if _bool_option(options, self.options, "gen_video"):
                command.append("--gen_video")
            return command
        return [python, entrypoint]


def three_d_four_d_runtime_spec(model_id: str) -> ThreeDFourDRuntimeSpec:
    key = model_id.lower().replace("_", "-")
    try:
        return THREE_D_FOUR_D_RUNTIME_SPECS[key]
    except KeyError as exc:
        known = ", ".join(sorted(THREE_D_FOUR_D_RUNTIME_SPECS))
        raise KeyError(f"Unknown 3D/4D runtime {model_id!r}. Known ids: {known}") from exc


def _resolve_source_root(
    spec: ThreeDFourDRuntimeSpec,
    options: Mapping[str, Any],
    source_repos: Sequence[Mapping[str, Any]],
) -> Path | None:
    candidates: list[Path] = []
    in_tree_parts = IN_TREE_RUNTIME_DIRS.get(spec.model_id)
    if in_tree_parts:
        candidates.append(REPO_ROOT / "worldfoundry" / Path(*in_tree_parts))
    explicit = options.get("source_root") or options.get("runtime_root") or options.get("repo_root")
    if explicit:
        candidates.append(Path(str(explicit)).expanduser())
    env_prefix = "WORLDFOUNDRY_" + spec.model_id.upper().replace("-", "_").replace(".", "_")
    for env_key in (f"{env_prefix}_ROOT", f"{env_prefix}_REPO_ROOT", f"{env_prefix}_SOURCE_ROOT"):
        if os.environ.get(env_key):
            candidates.append(Path(os.environ[env_key]).expanduser())
    if os.getenv("WORLDFOUNDRY_ALLOW_EXTERNAL_THREE_D_FOUR_D_REPOS", "").strip().lower() in {"1", "true", "yes", "on"}:
        repo_roots = []
        if os.environ.get("WORLDFOUNDRY_THREE_D_FOUR_D_REPOS_ROOT"):
            repo_roots.append(Path(os.environ["WORLDFOUNDRY_THREE_D_FOUR_D_REPOS_ROOT"]).expanduser())
        for root in repo_roots:
            for name in spec.repo_names:
                candidates.append(root / name)
        for source in source_repos:
            local_dir = source.get("local_dir")
            if local_dir:
                candidates.append(Path(str(local_dir)).expanduser())
    existing_dirs: list[Path] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        path = candidate.expanduser()
        path = path if path.is_absolute() else path.resolve()
        if (path / spec.entrypoint).is_file():
            return path
        existing_dirs.append(path)
    if existing_dirs:
        return existing_dirs[0]
    return None


def _materialize_media(*, images: Any, video: Any, run_dir: Path) -> dict[str, Any]:
    image_paths: list[str] = []
    if images is not None:
        values = images if isinstance(images, (list, tuple)) else [images]
        for index, item in enumerate(values):
            path = _coerce_media_item(item, run_dir, f"input_image_{index:02d}.png")
            if path:
                image_paths.append(path)
    return {
        "images": image_paths,
        "video": _coerce_media_item(video, run_dir, "input_video.mp4") if video is not None else "",
    }


def _coerce_media_item(value: Any, run_dir: Path, filename: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return str(path)
    save = getattr(value, "save", None)
    if callable(save):
        path = run_dir / filename
        save(path)
        return str(path.resolve())
    return str(value)


def _resolve_output(output_path: str | Path | None, run_dir: Path, filename: str) -> Path:
    if output_path is None:
        return (run_dir / filename).resolve()
    path = Path(output_path).expanduser()
    if path.suffix:
        return path.resolve()
    return (path / filename).resolve()


def _coerce_model_artifact_output(output: Path, spec: ThreeDFourDRuntimeSpec) -> Path:
    artifact_suffix = Path(spec.artifact_filename).suffix.lower()
    if not artifact_suffix or output.suffix.lower() == artifact_suffix:
        return output
    if spec.command_kind in {"monst3r_demo", "worldgen_inference", "shape_of_motion_preview"}:
        return output.with_name(Path(spec.artifact_filename).name)
    return output


def _plan_output_path(output: Path, model_id: str) -> Path:
    if output.suffix.lower() == ".json":
        return output
    return output.with_name(f"{output.stem or _safe_name(model_id)}_subprocess_runtime_plan.json")


def _image_data_dir(images: Sequence[str], run_dir: Path) -> str:
    if not images:
        return str(run_dir)
    if len(images) == 1:
        path = Path(images[0])
        return str(path.parent if path.parent != Path("") else run_dir)
    return str(Path(images[0]).parent)


def _stable_virtual_camera_image_dir(images: Sequence[str], run_dir: Path) -> str:
    image_paths = [
        Path(item)
        for item in _expand_image_inputs(images)
        if Path(item).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ]
    if not image_paths:
        return _image_data_dir(images, run_dir)
    target_dir = run_dir / "stable_virtual_camera_inputs"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(image_paths):
        name = source.name if index == 0 else f"{index:03d}_{source.name}"
        shutil.copy2(source, target_dir / name)
    return str(target_dir)


def _lagernvs_model_repo(value: str) -> str:
    text = str(value or "").strip()
    if not text or text == "facebook/lagernvs_general_512":
        return "facebook/lagernvs_general_512"
    path = Path(text)
    if path.expanduser().exists():
        return str(path.expanduser())
    normalized = text.replace("\\", "/")
    if "lagernvs_general_512" in normalized or "lagernvs-general-512" in normalized:
        return "facebook/lagernvs_general_512"
    return text


def _monst3r_input_dir(images: Sequence[str], run_dir: Path) -> str:
    if not images:
        return ""
    if len(images) == 1:
        return str(Path(images[0]))

    expanded = [
        Path(item).expanduser()
        for item in _expand_image_inputs(images)
        if Path(item).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ]
    if len(expanded) <= 1:
        return str(Path(images[0]))

    target_dir = run_dir / "monst3r_inputs"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(expanded):
        suffix = source.suffix or ".png"
        shutil.copy2(source, target_dir / f"{index:05d}{suffix.lower()}")
    return str(target_dir)


def _ckpt_root() -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", REPO_ROOT.parent / "ckpt")).expanduser()


def _has_depth_anything2_checkpoint(path: str | Path, encoder: str = "vitl") -> bool:
    candidate = Path(path).expanduser()
    checkpoint_name = f"depth_anything_v2_{encoder}.pth"
    if candidate.is_file():
        return candidate.name == checkpoint_name or candidate.name.startswith("depth_anything_v2_")
    return (candidate / checkpoint_name).is_file() or (candidate / "checkpoints" / checkpoint_name).is_file()


def _depth_anything2_model_path(encoder: str = "vitl") -> str | None:
    ckpt_root = _ckpt_root()
    for candidate in (
        ckpt_root / "Depth-Anything-V2-Large",
        ckpt_root / "Prior-Depth-Anything",
        ckpt_root / "DA-2",
        ckpt_root / "hfd" / "depth-anything--Depth-Anything-V2-Large",
        ckpt_root / "hfd" / "depth-anything--Prior-Depth-Anything",
    ):
        if _has_depth_anything2_checkpoint(candidate, encoder=encoder):
            return str(candidate)
    return None


def _first_existing_ckpt(*parts: str) -> str | None:
    root = _ckpt_root()
    for part in parts:
        candidate = root / part
        if candidate.exists():
            return str(candidate)
    return None


def _monst3r_raft_model_path() -> Path | None:
    for candidate in (
        _ckpt_root() / "monst3r_raft_models" / "raft-things.pth",
        _ckpt_root() / "hfd" / "monst3r--MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt" / "raft-things.pth",
    ):
        if candidate.is_file():
            return candidate
    return None


def _monst3r_sam2_checkpoint_path() -> Path | None:
    for candidate in (
        _ckpt_root() / "sam2.1-hiera-large" / "sam2.1_hiera_large.pt",
        _ckpt_root() / "hfd" / "facebook--sam2.1-hiera-large" / "sam2.1_hiera_large.pt",
        _ckpt_root() / "sam2-hiera-large" / "sam2_hiera_large.pt",
        _ckpt_root() / "hfd" / "facebook--sam2-hiera-large" / "sam2_hiera_large.pt",
    ):
        if candidate.is_file():
            return candidate
    return None


def _prepare_wonder_config(
    command_kind: str,
    source_root: Path | None,
    config: str,
    output_dir: Path,
    options: Mapping[str, Any],
    *,
    prompt: str = "",
    images: Sequence[str] | None = None,
) -> str:
    if source_root is None:
        return config
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = source_root / config_path
    if not config_path.is_file():
        return str(config)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data["runs_dir"] = str(output_dir)
    data["debug"] = False
    data["use_gpt"] = False
    data["regenerate_times"] = 0
    data["enable_regenerate"] = False
    data["stable_diffusion_checkpoint"] = (
        _first_existing_ckpt("stable-diffusion-2-inpainting", "sd2-community--stable-diffusion-2-inpainting")
        or data.get("stable_diffusion_checkpoint")
        or "stabilityai/stable-diffusion-2-inpainting"
    )

    def _runtime_image_path(value: object) -> str:
        if not value:
            return ""
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return str(path)
        for root in (REPO_ROOT, Path.cwd()):
            candidate = (root / path).resolve()
            if candidate.exists():
                return str(candidate)
        return str((REPO_ROOT / path).resolve())

    if command_kind == "wonderjourney_run":
        image_path = (
            options.get("image_filepath")
            or (images[0] if images else "")
            or options.get("image_path")
            or options.get("input_path")
            or data.get("image_filepath")
            or _default_wonderworld_image()
        )
        if image_path:
            data["image_filepath"] = _runtime_image_path(image_path)
        data["content_prompt"] = str(options.get("content_prompt") or prompt or data.get("content_prompt") or "village, cottages, trees")
        data["style_prompt"] = str(options.get("style_prompt") or data.get("style_prompt") or "photorealistic")
        data["negative_prompt"] = str(
            options.get("negative_prompt")
            or data.get("negative_prompt")
            or "low quality, blurry, distorted"
        )
        data["num_scenes"] = int(options.get("num_scenes", 1))
        data["num_keyframes"] = int(options.get("num_keyframes", 2))
        data["frames"] = int(options.get("frames", 2))
        data["save_fps"] = int(options.get("save_fps", 10))
        data["skip_interp"] = bool(options.get("skip_interp", False))
        data.setdefault("rotation_path", [0, 0, 0, 1])
    else:
        image_path = (
            options.get("image_filepath")
            or (images[0] if images else "")
            or options.get("image_path")
            or options.get("input_path")
            or data.get("image_filepath")
            or _default_wonderjourney_image()
        )
        if image_path:
            data["image_filepath"] = _runtime_image_path(image_path)
        data["content_prompt"] = str(options.get("content_prompt") or prompt or data.get("content_prompt") or "campus, buildings, trees")
        data["style_prompt"] = str(options.get("style_prompt") or data.get("style_prompt") or "photorealistic")
        data["negative_prompt"] = str(
            options.get("negative_prompt")
            or data.get("negative_prompt")
            or "low quality, blurry, distorted"
        )
        data["num_scenes"] = int(options.get("num_scenes", 1))
        data["frames"] = int(options.get("frames", 2))
        data["gen_sky_image"] = bool(options.get("gen_sky_image", False))
        data["gen_sky"] = bool(options.get("gen_sky", False))
        data["gen_layer"] = bool(options.get("gen_layer", False))
        data["load_gen"] = bool(options.get("load_gen", False))
        data.setdefault("rotation_path", [2])
        data["worldfoundry_batch"] = bool(options.get("worldfoundry_batch", True))
        data["worldfoundry_gs_iterations"] = int(options.get("worldfoundry_gs_iterations", 30))
        data["worldfoundry_sky_iterations"] = int(options.get("worldfoundry_sky_iterations", 30))

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_config = output_dir / f"{command_kind}_worldfoundry_demo.yaml"
    generated_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(generated_config)


def _default_wonderjourney_image() -> str:
    for candidate in (
        REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "hunyuan_game_craft" / "village.png",
        REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "studio_demo" / "00" / "image.jpg",
        REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "wonderjourney" / "assets" / "logo.png",
    ):
        if candidate.is_file():
            return str(candidate)
    return ""


def _default_wonderworld_image() -> str:
    for candidate in (
        REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "hunyuan_game_craft" / "village.png",
        REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "studio_demo" / "00" / "image.jpg",
        REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "wonderworld" / "assets" / "logo.png",
    ):
        if candidate.is_file():
            return str(candidate)
    return ""


def _maybe_export_shape_of_motion_preview(output_dir: Path, options: Mapping[str, Any]) -> Path | None:
    data_dir_value = (
        options.get("data_dir")
        or options.get("source_path")
        or options.get("dataset_path")
        or options.get("data_root")
        or ""
    )
    data_dir = Path(str(data_dir_value)).expanduser()
    if not data_dir.is_dir():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "shape_of_motion_scene.ply"
    if ply_path.exists():
        return ply_path

    npz_candidates = [data_dir / f"{data_dir.name}.npz", *sorted(data_dir.glob("*.npz"))]
    npz_path = next((path for path in npz_candidates if path.is_file()), None)
    if npz_path is None:
        return None

    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None

    def npz_value(bundle: Any, *keys: str) -> Any:
        for key in keys:
            if key in bundle.files:
                return bundle[key]
        return None

    try:
        bundle = np.load(npz_path)
        depths = npz_value(bundle, "depths", "depth")
        intrinsics = npz_value(bundle, "intrinsic", "intrinsics", "K")
        cam_c2w = npz_value(bundle, "cam_c2w", "c2w", "poses")
    except Exception:
        return None
    if depths is None or intrinsics is None or cam_c2w is None:
        return None

    image_files = _shape_of_motion_image_files(data_dir)
    if not image_files:
        return None

    depths = np.asarray(depths)
    intrinsics = np.asarray(intrinsics)
    cam_c2w = np.asarray(cam_c2w)
    if depths.ndim == 2:
        depths = depths[None, ...]

    max_points = int(options.get("shape_preview_max_points") or 4096)
    max_frames = min(len(depths), len(image_files), int(options.get("shape_preview_frames") or 3))
    points: list[tuple[float, float, float, int, int, int]] = []
    for frame_idx in range(max_frames):
        depth = np.asarray(depths[frame_idx]).squeeze()
        if depth.ndim != 2:
            continue
        height, width = depth.shape
        intrinsic = intrinsics[frame_idx] if intrinsics.ndim == 3 else intrinsics
        c2w = cam_c2w[frame_idx] if cam_c2w.ndim == 3 else cam_c2w
        if intrinsic.shape[0] < 3 or intrinsic.shape[1] < 3 or c2w.shape[0] < 3 or c2w.shape[1] < 4:
            continue
        try:
            image = Image.open(image_files[frame_idx]).convert("RGB").resize((width, height))
            colors = np.asarray(image)
        except Exception:
            colors = np.full((height, width, 3), 192, dtype=np.uint8)

        frame_budget = max(1, max_points // max_frames)
        stride = max(1, int(((height * width) / frame_budget) ** 0.5))
        ys, xs = np.mgrid[0:height:stride, 0:width:stride]
        zs = depth[ys, xs]
        valid = np.isfinite(zs) & (zs > 0)
        if not np.any(valid):
            continue
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        zs = zs[valid].astype(np.float32)
        fx = float(intrinsic[0, 0]) or 1.0
        fy = float(intrinsic[1, 1]) or 1.0
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        camera_points = np.stack(((xs - cx) / fx * zs, (ys - cy) / fy * zs, zs), axis=-1)
        world_points = camera_points @ np.asarray(c2w[:3, :3]).T + np.asarray(c2w[:3, 3])
        rgb = colors[ys.astype(np.int64), xs.astype(np.int64)]
        for xyz, color in zip(world_points, rgb, strict=False):
            points.append(
                (
                    float(xyz[0]),
                    float(xyz[1]),
                    float(xyz[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                )
            )
            if len(points) >= max_points:
                break
        if len(points) >= max_points:
            break

    if not points:
        return None
    _write_ascii_ply(ply_path, points)
    return ply_path


def _shape_of_motion_image_files(data_dir: Path) -> list[Path]:
    image_root = data_dir / "images"
    candidates = [image_root]
    if image_root.is_dir():
        candidates.extend(path for path in sorted(image_root.iterdir()) if path.is_dir())
    for root in candidates:
        if not root.is_dir():
            continue
        images = [
            path
            for path in sorted(root.iterdir())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        ]
        if images:
            return images
    return []


def _write_ascii_ply(path: Path, points: Sequence[tuple[float, float, float, int, int, int]]) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for x, y, z, r, g, b in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f} {r:d} {g:d} {b:d}\n")


def _expand_image_inputs(images: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for item in images:
        path = Path(item)
        if path.is_dir():
            expanded.extend(
                str(candidate)
                for candidate in sorted(path.iterdir())
                if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )
        elif item:
            expanded.append(str(path))
    return expanded


def _bool_option(options: Mapping[str, Any], defaults: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = options.get(key, defaults.get(key, default))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _extra_artifact_dirs(spec: ThreeDFourDRuntimeSpec, source_root: Path | None) -> tuple[Path, ...]:
    if source_root is None:
        return ()
    if spec.command_kind == "mvdiffusion_demo":
        return (source_root / "outputs",)
    if spec.command_kind == "stable_virtual_camera_demo":
        return (source_root / "work_dirs" / "demo",)
    return ()


def _select_artifact(output: Path, run_dir: Path, *, extra_dirs: Sequence[Path] = ()) -> Path:
    if output.exists():
        return output
    search_dirs = [run_dir, output.parent, *(path for path in extra_dirs if path.is_dir())]
    for pattern in ("*.mp4", "*.glb", "*.splat", "*.ply", "*.png", "*.json"):
        matches = sorted(
            (match for directory in search_dirs for match in directory.rglob(pattern)),
            key=lambda item: item.stat().st_mtime if item.exists() else 0,
        )
        if matches:
            return matches[-1]
    return output


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-") or "item"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tum_pose(row: Any) -> Any:
    """Convert one TUM ``timestamp tx ty tz qx qy qz qw`` row to camera-to-world."""

    import numpy as np

    values = np.asarray(row, dtype=np.float64)
    x, y, z, w = values[4:8]
    norm = max(float(np.linalg.norm([x, y, z, w])), 1e-12)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = values[1:4]
    return pose


def _build_monst3r_rerun_recording(
    sequence_dir: Path,
    *,
    confidence_threshold: float,
) -> Path | None:
    """Build a time-varying 4D viewer from MonST3R's official saved outputs."""

    try:
        import imageio.v2 as iio
        import numpy as np
        import rerun as rr
    except ImportError:
        return None

    depth_paths = sorted(sequence_dir.glob("frame_*.npy"))
    trajectory_path = sequence_dir / "pred_traj.txt"
    intrinsics_path = sequence_dir / "pred_intrinsics.txt"
    if not depth_paths or not trajectory_path.is_file() or not intrinsics_path.is_file():
        return None

    trajectory = np.atleast_2d(np.loadtxt(trajectory_path))
    intrinsics = np.atleast_2d(np.loadtxt(intrinsics_path)).reshape(-1, 3, 3)
    frame_count = min(len(depth_paths), len(trajectory), len(intrinsics))
    recording_path = sequence_dir.parent / "monst3r_sequence.rrd"
    rr.init("worldfoundry_monst3r", recording_id=sequence_dir.parent.name, spawn=False)
    rr.save(recording_path)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log("trajectory", rr.ViewCoordinates.RDF, static=True)
    poses = [_tum_pose(row) for row in trajectory[:frame_count]]
    rr.log(
        "trajectory/camera_path",
        rr.LineStrips3D([np.stack([pose[:3, 3] for pose in poses])]),
        static=True,
    )
    for frame_id, (depth_path, pose, intrinsic) in enumerate(
        zip(depth_paths[:frame_count], poses, intrinsics[:frame_count])
    ):
        if hasattr(rr, "set_time"):
            rr.set_time("frame", sequence=frame_id)
        else:
            rr.set_time_sequence("frame", frame_id)
        color_path = sequence_dir / f"frame_{frame_id:04d}.png"
        confidence_path = sequence_dir / f"init_conf_{frame_id}.npy"
        if not color_path.is_file() or not confidence_path.is_file():
            continue
        color = iio.imread(color_path)
        depth = np.load(depth_path)
        confidence = np.load(confidence_path)
        points = depth_to_world_points(depth, intrinsic, pose)
        valid = (
            np.isfinite(points).all(axis=-1)
            & np.isfinite(confidence)
            & (confidence > confidence_threshold)
        )
        rr.log("world/dynamic_points", rr.Points3D(points[valid], colors=color[valid], radii=0.001))
        rr.log(
            "world/current_camera",
            rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3], from_parent=True),
        )
        rr.log(
            "world/current_camera",
            rr.Pinhole(
                image_from_camera=intrinsic,
                resolution=[color.shape[1], color.shape[0]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.03,
            ),
        )
        rr.log("world/current_camera/rgb", rr.Image(color))
        rr.log(
            "observations/depth",
            rr.DepthImage(
                depth,
                meter=1.0,
                depth_range=[float(np.percentile(depth, 1)), float(np.percentile(depth, 99))],
            ),
        )
        mask_path = sequence_dir / f"dynamic_mask_{frame_id}.png"
        if mask_path.is_file():
            dynamic_mask = iio.imread(mask_path)
            if np.any(dynamic_mask):
                rr.log("diagnostics/dynamic_mask", rr.Image(dynamic_mask))
    rr.disconnect()
    return recording_path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "THREE_D_FOUR_D_RUNTIME_SPECS",
    "ThreeDFourDRuntimeSpec",
    "ThreeDFourDRuntimeSynthesis",
    "three_d_four_d_runtime_spec",
]
