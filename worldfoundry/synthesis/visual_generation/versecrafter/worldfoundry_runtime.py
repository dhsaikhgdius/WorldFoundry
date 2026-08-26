"""Independent in-tree runtime for VerseCrafter camera control."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path, resolve_data_path
from worldfoundry.runtime.conda import resolve_model_python
from worldfoundry.runtime.in_tree_cli import ensure_in_tree_runtime, execute_in_tree, require_path


class VerseCrafterRuntime:
    """Run VerseCrafter depth, 4D-control rendering, and diffusion stages."""

    SOURCE_REVISION = "008693b52aa74367afb34d183046fecf88100bdc"

    def __init__(
        self,
        *,
        checkpoint_path: Any = None,
        base_model_path: Any = None,
        moge_model_path: Any = None,
        python_executable: Any = None,
        device: str = "cuda",
        env: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo_root = ensure_in_tree_runtime(self.bundled_repo_root(), package_file=__file__)
        hfd = hfd_root_path()
        checkpoints = checkpoint_root_path()
        self.checkpoint_path = Path(
            checkpoint_path or hfd / "TencentARC--VerseCrafter"
        ).expanduser()
        self.base_model_path = Path(base_model_path or checkpoints / "Wan2.1-T2V-14B").expanduser()
        self.moge_model_path = Path(moge_model_path or hfd / "Ruicheng--moge-2-vitl-normal").expanduser()
        self.python_executable = resolve_model_python("versecrafter", explicit=python_executable)
        self.device = device
        self.env = dict(env or {})

    @staticmethod
    def bundled_repo_root() -> Path:
        return Path(__file__).resolve().parent / "versecrafter_runtime"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "VerseCrafterRuntime":
        return cls(checkpoint_path=pretrained_model_path, **kwargs)

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model_id": "versecrafter",
            "repo_root": str(self.repo_root),
            "source_revision": self.SOURCE_REVISION,
            "checkpoint_path": str(self.checkpoint_path),
            "base_model_path": str(self.base_model_path),
            "moge_model_path": str(self.moge_model_path),
            "inputs": {key: str(value) for key, value in kwargs.items()},
        }

    def _python_paths(self) -> tuple[Path, ...]:
        worldfoundry_root = Path(__file__).resolve().parents[3]
        return (
            worldfoundry_root.parent,
            self.repo_root,
            worldfoundry_root / "synthesis" / "visual_generation" / "video_x_fun" / "video_x_fun_runtime",
        )

    def _process_env(self) -> tuple[dict[str, str], str]:
        env = {str(key): str(value) for key, value in self.env.items()}
        effective_device = self.device
        if self.device.startswith("cuda:") and "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]
            effective_device = "cuda"
        python_path = os.pathsep.join(str(path.resolve()) for path in self._python_paths())
        inherited = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(part for part in (python_path, inherited) if part)
        return env, effective_device

    def _run_stage(
        self,
        name: str,
        command: Sequence[Any],
        *,
        log_dir: Path,
        env: Mapping[str, str],
    ) -> None:
        process_env = os.environ.copy()
        process_env.update(env)
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=self.repo_root,
            env=process_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        log_path = log_dir / f"{name}.log"
        log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"VerseCrafter stage {name!r} exited with code {completed.returncode}; see {log_path}")

    def predict(
        self,
        *,
        prompt: str,
        image_path: Any,
        trajectory_npz: Any,
        output_path: Any,
        width: int = 832,
        height: int = 480,
        num_frames: int = 81,
        fps: int = 16,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 2025,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        image = require_path(image_path, "VerseCrafter input image", kind="file")
        trajectory = require_path(trajectory_npz, "VerseCrafter camera trajectory", kind="file")
        checkpoint = require_path(self.checkpoint_path, "VerseCrafter checkpoint", kind="dir")
        for relative in (
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model-00001-of-00004.safetensors",
            "diffusion_pytorch_model-00002-of-00004.safetensors",
            "diffusion_pytorch_model-00003-of-00004.safetensors",
            "diffusion_pytorch_model-00004-of-00004.safetensors",
        ):
            require_path(checkpoint / relative, f"VerseCrafter asset {relative}", kind="file")
        base = require_path(self.base_model_path, "Wan2.1-T2V-14B base model", kind="dir")
        for relative in (
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
        ):
            require_path(base / relative, f"Wan2.1-T2V-14B asset {relative}", kind="file")
        moge = require_path(self.moge_model_path, "MoGe-v2 checkpoint", kind="dir")
        require_path(moge / "model.pt", "MoGe-v2 weights", kind="file")
        output = Path(output_path).expanduser().resolve()
        input_dir = output.parent / f".{output.stem}_versecrafter_inputs"
        depth_dir = input_dir / "estimated_depth"
        render_dir = input_dir / "rendering_4D_maps"
        native_output = input_dir / "outputs"
        mask_dir = input_dir / "empty_masks"
        for directory in (depth_dir, render_dir, native_output, mask_dir):
            directory.mkdir(parents=True, exist_ok=True)

        ellipsoid = input_dir / "empty_3D_gaussian_trajectory.json"
        ellipsoid.write_text(
            json.dumps(
                {
                    "metadata": {"num_frames": int(num_frames), "num_objects": 0, "obj_id_to_color_idx": {}},
                    "frames": [
                        {"frame_index": index, "objects": []}
                        for index in range(int(num_frames))
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        env, effective_device = self._process_env()
        self._run_stage(
            "depth",
            [
                self.python_executable,
                "inference/moge-v2_infer.py",
                "--input",
                image,
                "--output",
                depth_dir,
                "--pretrained",
                moge,
                "--device",
                effective_device,
                "--maps",
            ],
            log_dir=input_dir,
            env=env,
        )
        depth_npz = require_path(depth_dir / "depth_intrinsics.npz", "VerseCrafter estimated depth", kind="file")
        render_command: list[Any] = [
            self.python_executable,
            "inference/rendering_4D_control_maps.py",
            "--png_path",
            image,
            "--npz_path",
            depth_npz,
            "--mask_dir",
            mask_dir,
            "--trajectory_npz",
            trajectory,
            "--ellipsoid_json",
            ellipsoid,
            "--output_dir",
            render_dir,
            "--device",
            effective_device,
            "--fps",
            str(fps),
            "--target_height",
            str(height),
            "--target_width",
            str(width),
            "--use_fp16",
        ]
        self._run_stage("render_4d", render_command, log_dir=input_dir, env=env)
        for filename in (
            "background_RGB.mp4",
            "background_depth.mp4",
            "3D_gaussian_RGB.mp4",
            "3D_gaussian_depth.mp4",
            "merged_mask.mp4",
        ):
            require_path(render_dir / filename, f"VerseCrafter control map {filename}", kind="file")

        command = [
            self.python_executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "inference/versecrafter_inference.py",
            "--transformer_path",
            checkpoint,
            "--base_model_path",
            base,
            "--config_path",
            resolve_data_path(
                "models", "runtime", "configs", "versecrafter", "wan2.1", "wan_civitai.yaml"
            ),
            "--rendering_maps_path",
            render_dir,
            "--input_image_path",
            image,
            "--save_path",
            native_output,
            "--sample_size",
            f"{int(height)},{int(width)}",
            "--video_length",
            str(num_frames),
            "--fps",
            str(fps),
            "--num_inference_steps",
            str(num_inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
            "--ulysses_degree",
            "1",
            "--ring_degree",
            "1",
            "--prompt",
            prompt,
        ]
        result = execute_in_tree(
            command,
            cwd=self.repo_root,
            output_path=output,
            search_roots=(native_output,),
            env=env,
            python_paths=self._python_paths(),
            preferred_names=("generated_video_0.mp4",),
        )
        return result if return_dict else result.get("video")


__all__ = ["VerseCrafterRuntime"]
