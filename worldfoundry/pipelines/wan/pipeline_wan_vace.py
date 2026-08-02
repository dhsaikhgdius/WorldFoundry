"""Public Wan2.1 VACE adapter for the native diffusion infra."""

from __future__ import annotations

from typing import Any

from ..native_diffusion_video import NativeTextToVideoPipeline
from .pipeline_wan_2p1_t2v import WAN21_NEGATIVE_PROMPT


class Wan2p1VACEPipeline(NativeTextToVideoPipeline):
    """Wan2.1 VACE 14B controlled generation on the shared native runner."""

    MODEL_ID = "wan2.1-vace"
    OWNER = "Wan2.1 VACE 14B"
    CHECKPOINT_ROLES = ("dit", "text-encoder", "tokenizer", "vae")
    GENERATION_TYPE = "v2v"
    ACCEPTS_IMAGES = True
    ACCEPTS_VIDEO = True
    DEFAULT_HEIGHT = 720
    DEFAULT_WIDTH = 1280
    DEFAULT_NUM_FRAMES = 81
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 5.0
    DEFAULT_NEGATIVE_PROMPT = WAN21_NEGATIVE_PROMPT
    DEFAULT_FPS = 16
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 5.0}
    REQUEST_INPUT_DEFAULTS = {
        "vace_context": None,
        "vace_context_scale": 1.0,
        "vace_mask": None,
    }

    def __call__(
        self,
        prompt: str | list[str] = "",
        images: Any = None,
        video: Any = None,
        *,
        src_ref_images: Any = None,
        src_video: Any = None,
        src_mask: Any = None,
        size: str | None = None,
        frame_num: int | None = None,
        sample_steps: int | None = None,
        sample_shift: float | None = None,
        sample_guide_scale: float | None = None,
        base_seed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        if images is None and src_ref_images is not None:
            if isinstance(src_ref_images, str) and "," in src_ref_images:
                images = [value.strip() for value in src_ref_images.split(",") if value.strip()]
            else:
                images = src_ref_images
        if video is None:
            video = src_video
        if src_mask is not None:
            kwargs.setdefault("vace_mask", src_mask)
        height = kwargs.pop("height", None)
        width = kwargs.pop("width", None)
        if size is not None:
            separator = "*" if "*" in size else "x"
            try:
                parsed_width, parsed_height = (int(value) for value in size.lower().split(separator, 1))
            except (TypeError, ValueError) as error:
                raise ValueError("Wan VACE size must use WIDTH*HEIGHT, for example 1280*720") from error
            if (width is not None and int(width) != parsed_width) or (
                height is not None and int(height) != parsed_height
            ):
                raise ValueError("Wan VACE size conflicts with explicit height/width")
            width, height = parsed_width, parsed_height
        return super().__call__(
            prompt=prompt,
            images=images,
            video=video,
            height=height,
            width=width,
            num_frames=frame_num,
            num_inference_steps=sample_steps,
            guidance_scale=sample_guide_scale,
            seed=0 if base_seed is None else base_seed,
            shift=sample_shift,
            **kwargs,
        )


__all__ = ["Wan2p1VACEPipeline"]
