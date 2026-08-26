"""Independent in-tree runtime for Spatia."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import hfd_root_path, resolve_local_hf_model_path
from worldfoundry.runtime.conda import resolve_model_python
from worldfoundry.runtime.in_tree_cli import ensure_in_tree_runtime, execute_in_tree, require_path


def _checkpoint_file(root: Path, names: tuple[str, ...], label: str) -> Path:
    if root.is_file():
        return root
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{label} not found under {root}; checked {list(names)}")


def _first_frame(video_path: Path, output: Path) -> Path:
    import imageio.v3 as iio

    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, iio.imread(video_path, index=0))
    return output


class SpatiaRuntime:
    """Run Spatia's bundled long-horizon camera-control entrypoint."""

    SOURCE_REVISION = "f75b10c9bb5f6b0cd779ec8c41ab33dd8382dba3"

    def __init__(
        self,
        *,
        checkpoint_path: Any = None,
        base_model_path: Any = None,
        lora_path: Any = None,
        map_model_path: Any = None,
        python_executable: Any = None,
        device: str = "cuda",
        env: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo_root = ensure_in_tree_runtime(self.bundled_repo_root(), package_file=__file__)
        hfd = hfd_root_path()
        self.checkpoint_path = Path(checkpoint_path or hfd / "Jinjing713--Spatia").expanduser()
        self.base_model_path = Path(base_model_path or hfd / "Wan-AI--Wan2.2-TI2V-5B").expanduser()
        self.lora_path = Path(lora_path).expanduser() if lora_path else None
        self.map_model_path = Path(map_model_path or hfd / "facebook--map-anything").expanduser()
        self.python_executable = resolve_model_python("spatia", explicit=python_executable)
        self.device = device
        self.env = dict(env or {})

    @staticmethod
    def bundled_repo_root() -> Path:
        return Path(__file__).resolve().parent / "spatia_runtime"

    @staticmethod
    def project_root() -> Path:
        return Path(__file__).resolve().parents[4]

    @staticmethod
    def mapanything_root() -> Path:
        worldfoundry_root = Path(__file__).resolve().parents[3]
        return worldfoundry_root / "base_models" / "three_dimensions" / "general_3d" / "mapanything"

    @staticmethod
    def uniception_root() -> Path:
        worldfoundry_root = Path(__file__).resolve().parents[3]
        return (
            worldfoundry_root
            / "base_models"
            / "perception_core"
            / "general_perception"
            / "uniception"
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "SpatiaRuntime":
        return cls(checkpoint_path=pretrained_model_path, **kwargs)

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model_id": "spatia",
            "repo_root": str(self.repo_root),
            "source_revision": self.SOURCE_REVISION,
            "checkpoint_path": str(self.checkpoint_path),
            "base_model_path": str(self.base_model_path),
            "map_model_path": str(self.map_model_path),
            "inputs": {key: str(value) for key, value in kwargs.items()},
        }

    def predict(
        self,
        *,
        prompt: str,
        output_path: Any,
        w2c_trajectory_file: Any,
        intrinsics: Any,
        video_path: Any = None,
        image_path: Any = None,
        num_frames: int = 121,
        fps: int = 24,
        width: int = 1248,
        height: int = 704,
        num_inference_steps: int = 40,
        seed: int = 20917,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        output = Path(output_path).expanduser().resolve()
        input_dir = output.parent / f".{output.stem}_spatia_inputs"
        if image_path:
            image = require_path(image_path, "Spatia first frame", kind="file")
        else:
            video = require_path(video_path, "Spatia source video", kind="file")
            image = _first_frame(video, input_dir / "first_frame.png")
        trajectory = require_path(w2c_trajectory_file, "Spatia W2C trajectory", kind="file")
        if not isinstance(intrinsics, list) or not intrinsics:
            raise ValueError("Spatia requires per-frame 3x3 intrinsics")
        matrix = intrinsics[0]
        intrinsic_path = input_dir / "intrinsics.txt"
        intrinsic_path.parent.mkdir(parents=True, exist_ok=True)
        intrinsic_path.write_text(
            f"[{float(matrix[0][0])} {float(matrix[1][1])} {float(matrix[0][2])} {float(matrix[1][2])}]\n",
            encoding="utf-8",
        )
        checkpoint_root = require_path(self.checkpoint_path, "Spatia checkpoint")
        vace = _checkpoint_file(
            checkpoint_root,
            ("step-8500.safetensors", "control_weight_8500.safetensors"),
            "Spatia VACE checkpoint",
        )
        lora = self.lora_path or (
            checkpoint_root / "lora_weights_10000.safetensors" if checkpoint_root.is_dir() else None
        )
        base = require_path(self.base_model_path, "Wan2.2-TI2V-5B base model", kind="dir")
        map_model = resolve_local_hf_model_path(
            require_path(self.map_model_path, "MapAnything checkpoint", kind="dir"),
            required_files=("config.json", "model.safetensors"),
        )
        command: list[Any] = [
            self.python_executable,
            "inference.py",
            "--img_path",
            image,
            "--camera_w2c_path",
            trajectory,
            "--camera_intrinsics_path",
            intrinsic_path,
            "--vace_path",
            vace,
            "--model_root",
            base,
            "--map_model_path",
            map_model,
            "--save_path",
            output,
            "--work_dir",
            input_dir / "assets",
            "--prompt",
            prompt,
            "--width",
            str(width),
            "--height",
            str(height),
            "--max_frames",
            str(num_frames),
            "--first_round_frames",
            str(num_frames),
            "--fps",
            str(fps),
            "--num_inference_steps",
            str(num_inference_steps),
            "--seed",
            str(seed),
            "--map_device",
            self.device,
            "--render_device",
            self.device,
        ]
        if lora and Path(lora).is_file():
            command.extend(["--lora_path", lora])
        runtime_env = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            **self.env,
        }
        result = execute_in_tree(
            command,
            cwd=self.repo_root,
            output_path=output,
            search_roots=(input_dir / "assets",),
            env=runtime_env,
            python_paths=(
                self.project_root(),
                self.repo_root,
                self.mapanything_root(),
                self.uniception_root(),
            ),
        )
        return result if return_dict else result.get("video")


__all__ = ["SpatiaRuntime"]
