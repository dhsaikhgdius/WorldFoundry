"""Public Cosmos Predict2 pipeline backed by native diffusion recipes."""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.optimizations import (
    RuntimePolicy,
    parse_offload_policy,
    parse_torch_dtype,
)
from worldfoundry.core.io.video import save_image_or_video_tensor

from ..pipeline_utils import PipelineABC

COSMOS_PREDICT2_DEFAULT_FPS = 16
COSMOS_PREDICT2_DEFAULT_NUM_FRAMES = 93
COSMOS_PREDICT2_DEFAULT_NUM_INFERENCE_STEPS = 35
DEFAULT_NEGATIVE_PROMPT = (
    "The video captures ugly scenes, static motion, motion blur, over-saturation, shaky footage, "
    "low resolution, grainy texture, poor lighting, artifacts, unnatural transitions, visual noise, "
    "and flickering."
)


def _model_id(requested: str | None, source: object) -> str:
    value = str(requested or "").lower()
    if "14b" in value or "14b" in str(source).lower():
        return "cosmos-predict2-14b-video2world"
    supported = {
        "",
        "2b",
        "cosmos2",
        "cosmos-predict-2",
        "cosmos-predict2",
        "cosmos-predict2-2b",
        "cosmos-predict2-2b-video2world",
        "nvidia/cosmos-predict2-2b-video2world",
    }
    if value not in supported:
        raise ValueError(f"unsupported Cosmos Predict2 selector: {requested!r}")
    return "cosmos-predict2-2b-video2world"


def _subdirectory(
    root: Path,
    names: tuple[str, ...],
    *,
    required_file: str,
) -> Path | None:
    for name in names:
        candidate = root / name
        if (candidate / required_file).is_file():
            return candidate
    return None


class CosmosPredict2Pipeline(PipelineABC):
    """Studio-facing wrapper around the canonical Predict2 2B/14B recipes."""

    MODEL_ID = "cosmos-predict2"

    def __init__(self, *, native_pipeline: NativeDiffusionPipeline, device: str, model_id: str) -> None:
        super().__init__(model_id=model_id, synthesis_model=native_pipeline, device=device)
        self.native_pipeline = native_pipeline

    @staticmethod
    def _options(
        model_path: str | Mapping[str, Any] | None,
        required_components: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        return options

    @classmethod
    def _checkpoint_overrides(
        cls,
        model_path: str | Mapping[str, Any] | None,
        options: Mapping[str, Any],
        *,
        model_id: str,
    ) -> dict[str, str] | None:
        overrides: dict[str, str] = {}
        source = options.get("checkpoint_path", options.get("transformer_model_path", model_path))
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser()
            if path.is_file():
                overrides["transformer"] = str(path.resolve())
            elif path.is_dir():
                variant_name = (
                    "Cosmos-Predict2-14B-Video2World"
                    if "14b" in model_id
                    else "Cosmos-Predict2-2B-Video2World"
                )
                model_root = path if (path / "model-720p-16fps.pt").is_file() else _subdirectory(
                    path,
                    (variant_name, f"nvidia--{variant_name}"),
                    required_file="model-720p-16fps.pt",
                )
                if model_root is not None:
                    overrides.update({"transformer": str(model_root.resolve()), "vae": str(model_root.resolve())})
                text_root = path if (path / "pytorch_model.bin").is_file() else _subdirectory(
                    path,
                    ("t5-11b", "google-t5--t5-11b", "google-t5/t5-11b"),
                    required_file="pytorch_model.bin",
                )
                if text_root is not None:
                    overrides.update(
                        {"text-encoder": str(text_root.resolve()), "text-tokenizer": str(text_root.resolve())}
                    )
        explicit = {
            "transformer": options.get("transformer_model_path"),
            "vae": options.get("vae_model_path", options.get("tokenizer_model_path")),
            "text-encoder": options.get("text_encoder_model_path"),
            "text-tokenizer": options.get("text_tokenizer_path", options.get("text_encoder_model_path")),
        }
        for role, value in explicit.items():
            if isinstance(value, (str, Path)) and Path(value).expanduser().exists():
                overrides[role] = str(Path(value).expanduser().resolve())
        return overrides or None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "CosmosPredict2Pipeline":
        options = cls._options(model_path, required_components, kwargs)
        source = options.get("checkpoint_path", options.get("transformer_model_path", model_path))
        resolved_model_id = _model_id(str(options.get("variant_id") or model_id or ""), source)
        native = NativeDiffusionPipeline.from_pretrained(
            resolved_model_id,
            policy=RuntimePolicy(
                device=torch.device(device),
                dtype=parse_torch_dtype(
                    options.get("torch_dtype", options.get("weight_dtype", options.get("dtype"))),
                    owner="Cosmos Predict2",
                ),
                offload=parse_offload_policy(
                    options.get("offload_mode", "block"),
                    allow_disk=False,
                    owner="Cosmos Predict2",
                ),
            ),
            checkpoint_overrides=cls._checkpoint_overrides(
                model_path,
                options,
                model_id=resolved_model_id,
            ),
            component_options={
                "latent_encoder:codec": {
                    "tiled": bool(options.get("vae_tiling", False)),
                    "tile_size": tuple(options.get("vae_tile_size", (34, 34))),
                    "tile_stride": tuple(options.get("vae_tile_stride", (18, 16))),
                }
            },
        )
        return cls(native_pipeline=native, device=device, model_id=resolved_model_id)

    @classmethod
    def plan(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        options = cls._options(model_path, required_components, kwargs)
        source = options.get("checkpoint_path", options.get("transformer_model_path", model_path))
        model_id = _model_id(str(options.get("variant_id") or options.get("model_id") or ""), source)
        return {
            "model_id": model_id,
            "backend": "worldfoundry-native-diffusion",
            "native_inference": True,
            "checkpoints": cls._checkpoint_overrides(model_path, options, model_id=model_id),
            "blocked": False,
        }

    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        images: Any = None,
        image: Any = None,
        image_path: str | None = None,
        input_path: str | None = None,
        video: Any = None,
        video_path: str | None = None,
        output_path: str | Path | None = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = COSMOS_PREDICT2_DEFAULT_NUM_INFERENCE_STEPS,
        fps: int = COSMOS_PREDICT2_DEFAULT_FPS,
        num_frames: int = COSMOS_PREDICT2_DEFAULT_NUM_FRAMES,
        height: int = 704,
        width: int = 1280,
        seed: int = 0,
        num_latent_conditional_frames: int = 1,
        output_type: str = "video",
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise TypeError(f"unsupported Cosmos Predict2 inference options: {sorted(kwargs)}")
        image_value = image if image is not None else images
        image_value = image_value if image_value is not None else image_path
        video_value = video if video is not None else video_path
        if image_value is None and video_value is None and input_path is not None:
            if Path(input_path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                video_value = input_path
            else:
                image_value = input_path
        if image_value is None and video_value is None:
            raise ValueError("Cosmos Predict2 Video2World requires an image or video input")
        if image_value is not None and video_value is not None:
            raise ValueError("Cosmos Predict2 accepts either image or video conditioning, not both")
        inputs: dict[str, object] = {
            "fps": int(fps),
            "return_latent": output_type == "latent",
            "num_latent_conditional_frames": int(num_latent_conditional_frames),
        }
        if image_value is not None:
            inputs["image"] = image_value
        else:
            inputs["video"] = video_value
        actual_seed = secrets.randbits(63) if int(seed) < 0 else int(seed)
        output = self.native_pipeline(
            DiffusionRequest(
                prompt=prompt,
                negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                sampling=SamplingConfig(
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    seed=actual_seed,
                ),
                inputs=inputs,
            )
        )
        artifact_path = None
        if output_path is not None and output_type != "latent":
            artifact_path = save_image_or_video_tensor(output.sample, output_path, fps=int(fps))
        result = {
            "video": output.sample,
            "generated_video": output.sample,
            "latents": output.latents,
            "artifact_path": artifact_path,
            "generated_video_path": artifact_path,
            "metadata": dict(output.metadata),
        }
        return result if return_dict else (artifact_path or output.sample)

    def get_synthesis_model(self) -> NativeDiffusionPipeline:
        return self.native_pipeline


__all__ = ["CosmosPredict2Pipeline"]
