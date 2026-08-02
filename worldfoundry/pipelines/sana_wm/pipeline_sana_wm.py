"""SANA-WM product adapter over the canonical native diffusion runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from ..native_diffusion import NativeVisualDiffusionPipeline


class SanaWMPipeline(NativeVisualDiffusionPipeline):
    """Resident camera-controlled world generation without a private runtime."""

    MODEL_ID = "sana-wm"
    OWNER = "SANA-WM"
    CHECKPOINT_ROLES = ("dit", "text-encoder", "tokenizer", "codec")
    GENERATION_TYPE = "i2v"
    ACCEPTS_IMAGES = True
    REQUIRES_IMAGES = True
    ACCEPTS_INTERACTIONS = True
    DEFAULT_HEIGHT = 704
    DEFAULT_WIDTH = 1280
    DEFAULT_NUM_FRAMES = 161
    DEFAULT_NUM_INFERENCE_STEPS = 60
    DEFAULT_GUIDANCE_SCALE = 5.0
    DEFAULT_FPS = 16
    DEFAULT_NEGATIVE_PROMPT = ""
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 9.8}
    NUM_INFERENCE_STEP_ALIASES = ("step", "infer_steps")
    REQUEST_INPUT_DEFAULTS = {
        "camera_to_world": None,
        "camera_actions": None,
        "intrinsics": None,
    }
    REQUEST_INPUT_ALIASES = {
        "camera_poses": "camera_to_world",
        "action": "camera_actions",
        "camera_action": "camera_actions",
    }

    def __init__(self, *, native_pipeline, device: str, model_id: str | None = None) -> None:
        super().__init__(native_pipeline=native_pipeline, device=device, model_id=model_id)
        self._realtime_config: dict[str, Any] | None = None

    def prepare_realtime(self) -> dict[str, Any]:
        return {
            "realtime_spec": {
                "supports_prompt_updates": True,
                "supports_camera_actions": True,
                "resident": True,
                "runtime": "worldfoundry-native-diffusion",
            },
            "runtime_info": {
                "model_id": self.model_id,
                "device": str(self.native_pipeline.device),
                "dtype": str(self.native_pipeline.dtype),
            },
        }

    def configure_realtime(
        self,
        images: Any,
        prompt: str = "",
        seed: int = 42,
        fps: int = 16,
        window_frames: int | None = None,
        step: int = 60,
        cfg_scale: float = 5.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(images, (str, Path)):
            with Image.open(images) as source:
                images = source.convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("SANA-WM realtime requires a PIL image or image path")
        if not str(prompt).strip():
            raise ValueError("SANA-WM realtime requires a prompt")
        self._realtime_config = {
            "images": images.copy(),
            "prompt": str(prompt),
            "seed": int(seed),
            "fps": int(fps),
            "num_frames": int(window_frames or self.DEFAULT_NUM_FRAMES),
            "num_inference_steps": int(step),
            "guidance_scale": float(cfg_scale),
            **kwargs,
        }
        return self.prepare_realtime()

    def stream_realtime(
        self,
        interactions: Sequence[str] | None = None,
        prompt: str | None = None,
        realtime_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if realtime_segments:
            raise ValueError("use frame-rate interactions or explicit camera_to_world poses")
        if self._realtime_config is None:
            raise RuntimeError("SANA-WM realtime session is not configured")
        options = {**self._realtime_config, **kwargs}
        if prompt is not None:
            options["prompt"] = prompt
        if seed is not None:
            options["seed"] = int(seed)
        return self(interactions=interactions, return_dict=True, **options)

    def reset_realtime(self) -> None:
        self._realtime_config = None

    def __call__(
        self,
        images: Any = None,
        prompt: str = "",
        interactions: Sequence[str] | None = None,
        fps: int = 16,
        num_frames: int = 161,
        step: int = 60,
        cfg_scale: float = 5.0,
        seed: int = 42,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        return super().__call__(
            prompt=prompt,
            images=images,
            interactions=interactions,
            fps=fps,
            num_frames=int(kwargs.pop("window_frames", num_frames)),
            num_inference_steps=step,
            guidance_scale=cfg_scale,
            seed=seed,
            return_dict=return_dict,
            **kwargs,
        )


__all__ = ["SanaWMPipeline"]
