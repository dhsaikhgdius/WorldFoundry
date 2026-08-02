"""Unified pipelines for independently registered Echo-Memory models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.echo_memory import (
    EchoMemoryBlockSSMSynthesis,
    EchoMemoryContextK1Synthesis,
    EchoMemoryContextK20Synthesis,
    EchoMemorySpatialConcatTextSynthesis,
    EchoMemorySpatialCrossAttnT32Synthesis,
    EchoMemorySpatialNoInjectionSynthesis,
    EchoMemorySpatialSynthesis,
    EchoMemorySSMCtx1Every4Hint21Synthesis,
    EchoMemorySSMCtx5Every1Hint21Synthesis,
    EchoMemorySSMCtx5Every4Hint81Synthesis,
    EchoMemorySynthesis,
    EchoMemoryVideoSSMHybridSynthesis,
)


class EchoMemoryPipeline(PipelineABC):
    """Common I2V interface; subclasses pin the model and checkpoint recipe."""

    ABSTRACT_PIPELINE = True
    SYNTHESIS_CLS: type[EchoMemorySynthesis]

    def __init__(
        self,
        *,
        synthesis_model: EchoMemorySynthesis | None = None,
        device: str = "cuda",
    ) -> None:
        if self.MODEL_ID is None:
            raise TypeError("EchoMemoryPipeline must be instantiated through a model-specific subclass")
        super().__init__(model_id=self.MODEL_ID, synthesis_model=synthesis_model, device=device)
        self.model_name = self.MODEL_ID
        self.generation_type = "i2v"

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        lazy: bool = True,
        **kwargs: Any,
    ) -> "EchoMemoryPipeline":
        options: dict[str, Any] = {}
        checkpoint_path = model_path
        if isinstance(model_path, Mapping):
            options.update(model_path)
            checkpoint_path = None
        elif isinstance(model_path, str) and not model_path.strip():
            checkpoint_path = None
        options.update(required_components or {})
        options.update(kwargs)
        requested_model = options.pop("model_id", cls.MODEL_ID)
        if requested_model != cls.MODEL_ID:
            raise ValueError(f"{cls.__name__} is bound to {cls.MODEL_ID!r}, got {requested_model!r}")
        synthesis = cls.SYNTHESIS_CLS.from_pretrained(
            pretrained_model_path=checkpoint_path,
            device=device,
            lazy=lazy,
            generator_overrides=options,
        )
        return cls(synthesis_model=synthesis, device=device)

    @staticmethod
    def process(prompt: str, images: Any) -> dict[str, Any]:
        if images is None:
            raise ValueError("Echo-Memory requires an initial image")
        return {"prompt": str(prompt or ""), "images": images}

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        *,
        output_path: str | None = None,
        fps: int | None = None,
        num_frames: int | None = None,
        frames: int | None = None,
        width: int | None = None,
        height: int | None = None,
        num_chunks: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        camera_trajectory: Any = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self.synthesis_model is None:
            raise RuntimeError(f"{self.MODEL_ID} is not loaded; call from_pretrained() first")
        request = self.process(prompt, images)
        result = self.synthesis_model.predict(
            prompt=request["prompt"],
            images=request["images"],
            output_path=output_path,
            fps=fps,
            return_dict=True,
            num_frames=num_frames if num_frames is not None else frames,
            width=width,
            height=height,
            num_chunks=num_chunks,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            camera_trajectory=camera_trajectory,
            **kwargs,
        )
        return result if return_dict else result["video"]

    def close(self) -> None:
        if self.synthesis_model is not None:
            close = getattr(self.synthesis_model, "close", None)
            if callable(close):
                close()


class EchoMemoryContextK1Pipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-context-k1"
    SYNTHESIS_CLS = EchoMemoryContextK1Synthesis


class EchoMemoryContextK20Pipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-context-k20"
    SYNTHESIS_CLS = EchoMemoryContextK20Synthesis


class EchoMemorySpatialPipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-spatial"
    SYNTHESIS_CLS = EchoMemorySpatialSynthesis


class EchoMemoryBlockSSMPipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-block-ssm"
    SYNTHESIS_CLS = EchoMemoryBlockSSMSynthesis


class EchoMemoryVideoSSMHybridPipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-videossm-hybrid"
    SYNTHESIS_CLS = EchoMemoryVideoSSMHybridSynthesis


class EchoMemorySpatialConcatTextPipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-spatial-concat-text"
    SYNTHESIS_CLS = EchoMemorySpatialConcatTextSynthesis


class EchoMemorySpatialNoInjectionPipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-spatial-no-injection"
    SYNTHESIS_CLS = EchoMemorySpatialNoInjectionSynthesis


class EchoMemorySpatialCrossAttnT32Pipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-spatial-cross-attn-t32"
    SYNTHESIS_CLS = EchoMemorySpatialCrossAttnT32Synthesis


class EchoMemorySSMCtx1Every4Hint21Pipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-ssm-ctx1-every4-hint21"
    SYNTHESIS_CLS = EchoMemorySSMCtx1Every4Hint21Synthesis


class EchoMemorySSMCtx5Every1Hint21Pipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-ssm-ctx5-every1-hint21"
    SYNTHESIS_CLS = EchoMemorySSMCtx5Every1Hint21Synthesis


class EchoMemorySSMCtx5Every4Hint81Pipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-ssm-ctx5-every4-hint81"
    SYNTHESIS_CLS = EchoMemorySSMCtx5Every4Hint81Synthesis


__all__ = [
    "EchoMemoryBlockSSMPipeline",
    "EchoMemoryContextK1Pipeline",
    "EchoMemoryContextK20Pipeline",
    "EchoMemoryPipeline",
    "EchoMemorySSMCtx1Every4Hint21Pipeline",
    "EchoMemorySSMCtx5Every1Hint21Pipeline",
    "EchoMemorySSMCtx5Every4Hint81Pipeline",
    "EchoMemorySpatialConcatTextPipeline",
    "EchoMemorySpatialCrossAttnT32Pipeline",
    "EchoMemorySpatialNoInjectionPipeline",
    "EchoMemorySpatialPipeline",
    "EchoMemoryVideoSSMHybridPipeline",
]
