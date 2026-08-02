"""Public HunyuanVideo pipelines backed only by WorldFoundry native diffusion."""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.optimizations import (
    AttentionBackend,
    RuntimePolicy,
    parse_offload_policy,
    parse_torch_dtype,
)
from worldfoundry.core.io.video import save_image_or_video_tensor
from worldfoundry.operators.runtime_video_operator import RuntimeVideoOperator
from worldfoundry.synthesis.visual_generation.memory.video import VideoArtifactMemory

from ..pipeline_utils import PipelineABC


class NativeHunyuanVideoPipeline(PipelineABC):
    """Stable Studio-facing adapter around one declarative HunyuanVideo recipe."""

    MODEL_ID = "hunyuanvideo-t2v"
    GENERATION_TYPE = "t2v"
    DEFAULT_HEIGHT = 720
    DEFAULT_WIDTH = 1280
    DEFAULT_NUM_FRAMES = 129
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_FPS = 24
    DEFAULT_GUIDANCE_SCALE = 6.0

    @staticmethod
    def _attention_policy(value: object, *, model_id: str) -> AttentionBackend:
        """Select the memory-bounded official attention path for HunyuanVideo 1.5.

        The 720p/121-frame 1.5 recipes have roughly 40k visual tokens.  Native
        torch SDPA falls back to a dense masked kernel for that shape and tries
        to materialize an attention matrix larger than 23 GiB.  FlashAttention
        is the upstream full-resolution path and changes neither sampling steps
        nor output geometry, so treat ``auto`` as Flash for 1.5 while preserving
        explicit Torch/SDPA requests for diagnostics.
        """

        normalized = str(value or "auto").strip().lower().replace("-", "_")
        if normalized == "auto" and model_id.startswith("hunyuanvideo-1.5-"):
            return AttentionBackend.FLASH
        if normalized in {"flash_attention", "flash_attention_2", "flash2", "flash_attn"}:
            return AttentionBackend.FLASH
        return AttentionBackend(normalized)

    def __init__(self, *, native_pipeline: NativeDiffusionPipeline, device: str, model_id: str) -> None:
        super().__init__(
            model_id=model_id,
            operator=RuntimeVideoOperator(generation_type=self.GENERATION_TYPE),
            memory_module=VideoArtifactMemory(model_id=model_id),
            device=device,
        )
        self.native_pipeline = native_pipeline

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "NativeHunyuanVideoPipeline":
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        resolved_model_id = str(options.get("variant_id") or model_id or cls.MODEL_ID)
        source = options.get("checkpoint_path", options.get("model_path", model_path))
        overrides = None
        if isinstance(source, (str, Path)) and Path(source).expanduser().is_dir():
            source_root = Path(source).expanduser().resolve()
            root = str(source_root)
            overrides = {name: root for name in ("transformer", "vae", "resources")}
            local_vision = source_root / "vision_encoder" / "siglip"
            if resolved_model_id == "hunyuanvideo-1.5-i2v" and local_vision.is_dir():
                overrides["vision"] = str(local_vision.resolve())

        native = NativeDiffusionPipeline.from_pretrained(
            resolved_model_id,
            policy=RuntimePolicy(
                device=torch.device(device),
                dtype=parse_torch_dtype(
                    options.get("torch_dtype", options.get("weight_dtype", options.get("dtype"))),
                    owner="HunyuanVideo",
                ),
                offload=parse_offload_policy(
                    options.get("offload_mode", "block"),
                    allow_disk=False,
                    owner="HunyuanVideo",
                ),
                attention=cls._attention_policy(
                    options.get("attention_backend", options.get("attention")),
                    model_id=resolved_model_id,
                ),
            ),
            checkpoint_overrides=overrides,
            component_options={
                "latent_encoder:codec": {
                    "tiled": bool(options.get("vae_tiling", True)),
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
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        source = options.get("checkpoint_path", options.get("model_path", model_path))
        return {
            "model_id": str(options.get("variant_id") or options.get("model_id") or cls.MODEL_ID),
            "checkpoint": str(source) if source is not None else None,
            "backend": "worldfoundry-native-diffusion",
            "native_inference": True,
            "blocked": False,
        }

    def _process_prompt(self, prompt: str) -> str:
        self.operator.get_interaction(prompt)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()
        return str(interaction["processed_prompt"])

    def process(self, prompt: str | list[str], images: Any = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        if self.GENERATION_TYPE == "i2v" and images is None:
            raise ValueError(f"{self.model_id} requires an input image")
        if self.GENERATION_TYPE == "t2v" and images is not None:
            raise ValueError(f"{self.model_id} is text-to-video and does not accept images")
        processed = [self._process_prompt(value) for value in prompt] if isinstance(prompt, list) else self._process_prompt(prompt)
        return {"prompt": processed, "images": images}

    def __call__(
        self,
        prompt: str | list[str],
        images: Any = None,
        image: Any = None,
        image_path: str | None = None,
        output_path: str | Path | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        embedded_guidance_scale: float | None = None,
        seed: int = 0,
        fps: int | None = None,
        output_type: str = "video",
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        image_value = image if image is not None else images
        image_value = image_value if image_value is not None else image_path
        processed = self.process(prompt=prompt, images=image_value)
        actual_seed = secrets.randbits(63) if int(seed) < 0 else int(seed)
        scale = float(guidance_scale if guidance_scale is not None else self.DEFAULT_GUIDANCE_SCALE)
        inputs: dict[str, object] = {
            "fps": int(fps or self.DEFAULT_FPS),
            "return_latent": output_type == "latent",
            "embedded_guidance_scale": float(embedded_guidance_scale or scale),
        }
        if image_value is not None:
            inputs["image"] = image_value
        request = DiffusionRequest(
            prompt=processed["prompt"],
            height=int(height or self.DEFAULT_HEIGHT),
            width=int(width or self.DEFAULT_WIDTH),
            num_frames=int(num_frames or self.DEFAULT_NUM_FRAMES),
            sampling=SamplingConfig(
                num_inference_steps=int(num_inference_steps or self.DEFAULT_NUM_INFERENCE_STEPS),
                guidance_scale=scale,
                seed=actual_seed,
            ),
            inputs=inputs,
        )
        output = self.native_pipeline(request)
        artifact_path = None
        if output_path is not None and output_type != "latent":
            artifact_path = save_image_or_video_tensor(output.sample, output_path, fps=int(fps or self.DEFAULT_FPS))
        result = {
            "video": output.sample,
            "latents": output.latents,
            "artifact_path": artifact_path,
            "generated_video_path": artifact_path,
            "model_name": self.model_id,
            "generation_type": self.GENERATION_TYPE,
            "metadata": dict(output.metadata),
        }
        return result if return_dict else (artifact_path or output.sample)

    def stream(self, prompt: str, images: Any = None, **kwargs: Any) -> Any:
        result = self(prompt=prompt, images=images, return_dict=True, **kwargs)
        value = result.get("artifact_path") or result["video"]
        self.memory_module.record(value, metadata={"prompt": prompt, "model_name": self.model_id})
        return value

    def get_synthesis_model(self) -> NativeDiffusionPipeline:
        return self.native_pipeline


class HunyuanVideoT2VPipeline(NativeHunyuanVideoPipeline):
    MODEL_ID = "hunyuanvideo-t2v"


class HunyuanVideoI2VPipeline(NativeHunyuanVideoPipeline):
    MODEL_ID = "hunyuanvideo-i2v"
    GENERATION_TYPE = "i2v"


class HunyuanVideo15T2VPipeline(NativeHunyuanVideoPipeline):
    MODEL_ID = "hunyuanvideo-1.5-t2v"
    DEFAULT_HEIGHT = 720
    DEFAULT_WIDTH = 1280
    DEFAULT_NUM_FRAMES = 121
    DEFAULT_GUIDANCE_SCALE = 6.0


class HunyuanVideo15I2VPipeline(HunyuanVideo15T2VPipeline):
    MODEL_ID = "hunyuanvideo-1.5-i2v"
    GENERATION_TYPE = "i2v"
    DEFAULT_WIDTH = 544


__all__ = [
    "HunyuanVideo15I2VPipeline",
    "HunyuanVideo15T2VPipeline",
    "HunyuanVideoI2VPipeline",
    "HunyuanVideoT2VPipeline",
    "NativeHunyuanVideoPipeline",
]
