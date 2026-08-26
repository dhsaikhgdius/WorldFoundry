"""Yume visual generation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from worldfoundry.core.io.paths import project_root
from worldfoundry.runtime.env import resolve_ckpt_dir

from ..pipeline_utils import PipelineABC


_REPO_ROOT = project_root(__file__)
_DEFAULT_CKPT_ROOT = resolve_ckpt_dir()
_DEFAULT_YUME_CKPT = _DEFAULT_CKPT_ROOT / "Yume-I2V-540P"
_INTERNAL_RUNTIME_ROOT = _REPO_ROOT / "worldfoundry" / "synthesis" / "visual_generation" / "yume"


def _resolve_local_checkpoint(value: Any, default_path: Path) -> str:
    """Resolve an explicit local checkpoint directory."""
    path = Path(str(value or default_path)).expanduser()
    if path.is_dir():
        return str(path.resolve())
    raise FileNotFoundError(
        "Yume requires a local checkpoint directory. "
        f"Expected {path}. Download weights with huggingface-cli or "
        "scripts/model_zoo/download_checkpoints.py, then set WORLDFOUNDRY_CKPT_DIR "
        "or pass an explicit local path."
    )


class YumePipeline(PipelineABC):

    """Pipeline implementation for Yume visual generation."""
    def __init__(
        self,
        synthesis_model: Optional[Any] = None,
        operators: Optional[Any] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        weight_dtype: Any = None,
        checkpoint_root: str | Path = _DEFAULT_CKPT_ROOT,
    ) -> None:

        """Initialize the pipeline and configure runtime components."""
        self.operators = operators
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype
        self.runtime_root = _INTERNAL_RUNTIME_ROOT
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Optional[dict[str, Any]] = None,
        device: str = "cuda",
        weight_dtype: Any = None,
        fsdp: bool = False,
        **kwargs: Any,
    ) -> "YumePipeline":
        """
        Load Yume from local official repo metadata and staged checkpoints.

        Args:
            model_path: Local checkpoint path, repo id, or options mapping.
            required_components: Optional repo/checkpoint path mapping.
            device: Runtime device label.
            weight_dtype: Torch dtype used by the runtime.
            fsdp: Whether to enable the official FSDP path.
            **kwargs: Additional adapter/runtime options.
        """
        options: dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            options["checkpoint_path"] = model_path
        options.update(required_components or {})
        options.update(kwargs)
        synthesis_model_path = _resolve_local_checkpoint(
            options.get("checkpoint_path") or options.get("pretrained_model_path") or options.get("model_path"),
            _DEFAULT_YUME_CKPT,
        )

        import torch

        from ...synthesis.visual_generation.memory.stream import VisualContextMemory
        from ...operators.yume_operator import YumeOperator
        from ...synthesis.visual_generation.yume.yume_synthesis import YumeSynthesis

        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype = weight_dtype or torch.bfloat16
        synthesis_model = YumeSynthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            device=device,
            weight_dtype=weight_dtype,
            fsdp=fsdp,
        )
        operators = YumeOperator()
        memory_module = VisualContextMemory(model_id="yume")

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
            weight_dtype=weight_dtype,
            checkpoint_root=options.get("checkpoint_root") or options.get("ckpt_root") or _DEFAULT_CKPT_ROOT,
        )
        return pipeline

    def _dist_rank(self) -> int:
        """Dist rank for YumePipeline."""
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _to_pil_frames(self, video: Any) -> List[Any]:
        """To pil frames for YumePipeline."""
        import numpy as np
        import torch
        from PIL import Image

        def _frame_to_pil(frame: Any) -> Any:
            """Frame to pil for YumePipeline."""
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if frame.ndim != 3:
                raise ValueError(f"Unsupported frame shape: {frame.shape}")
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]

            if frame.dtype != np.uint8:
                if np.issubdtype(frame.dtype, np.floating):
                    if frame.min() < 0:
                        frame = (frame + 1.0) / 2.0
                    if frame.max() <= 1.0:
                        frame = np.clip(frame, 0.0, 1.0)
                        frame = (frame * 255.0).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            return Image.fromarray(frame)

        if isinstance(video, list):
            if len(video) == 0:
                return []
            if isinstance(video[0], Image.Image):
                return video
            if isinstance(video[0], np.ndarray):
                return [_frame_to_pil(frame) for frame in video]
            raise TypeError(f"Unsupported frame type in list: {type(video[0])}")

        if isinstance(video, np.ndarray):
            if video.ndim != 4:
                raise ValueError(f"Expected video ndarray shape (T,H,W,C), got {video.shape}")
            return [_frame_to_pil(frame) for frame in video]

        if isinstance(video, torch.Tensor):
            v = video.detach().cpu()
            if v.ndim != 4:
                raise ValueError(f"Expected video tensor ndim=4, got shape {tuple(v.shape)}")

            if v.shape[-1] in (1, 3, 4):
                arr = v.numpy()
            elif v.shape[1] in (1, 3, 4):
                arr = v.permute(0, 2, 3, 1).numpy()
            elif v.shape[0] in (1, 3, 4):
                arr = v.permute(1, 2, 3, 0).numpy()
            else:
                raise ValueError(f"Cannot infer tensor video layout from shape {tuple(v.shape)}")
            return [_frame_to_pil(frame) for frame in arr]

        raise TypeError(f"Unsupported video type: {type(video)}")

    def process(
        self,
        interactions: Union[str, List[str]],
        images: Optional[Any] = None,
        videos: Optional[List[Any]] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:

        """Process and normalize input arguments and conditions for inference."""
        self.operators.get_interaction(interactions)
        operator_condition = self.operators.process_interaction()
        self.operators.delete_last_interaction()

        visual_context = self.operators.process_perception(images=images, videos=videos, size=size)

        return {
            "operator_condition": operator_condition,
            "visual_context": visual_context,
        }

    def __call__(
        self,
        prompt: Optional[str] = "",
        interactions: Optional[Union[str, List[str]]] = None,
        interaction_speeds: Optional[Union[float, List[float]]] = None,
        interaction_distances: Optional[Union[float, List[float]]] = None,
        images: Optional[Any] = None,
        videos: Optional[List[Any]] = None,
        size: Optional[str] = "544*960",
        seed: Optional[int] = None,
        task_type: Optional[str] = "i2v",
        num_euler_timesteps: Optional[int] = 100,
        sampling_method: Optional[str] = None,
    ) -> Any:

        """Execute the complete pipeline generation flow."""
        output_dict = self.process(
            interactions=interactions,
            images=images,
            videos=videos,
            size=size,
        )

        output_video = self.synthesis_model.predict(
            prompt=prompt,
            image=output_dict["visual_context"]["ref_images"],
            video=output_dict["visual_context"]["ref_videos"],
            interactions=interactions,
            interaction_captions=output_dict["operator_condition"],
            interaction_speeds=interaction_speeds,
            interaction_distances=interaction_distances,
            task_type=task_type,
            size=size,
            seed=seed,
            num_euler_timesteps=num_euler_timesteps,
            sampling_method=sampling_method,
        )

        return output_video

    def stream(
        self,
        prompt: Optional[str] = "",
        interactions: Optional[Union[str, List[str]]] = None,
        interaction_speeds: Optional[Union[float, List[float]]] = None,
        interaction_distances: Optional[Union[float, List[float]]] = None,
        images: Optional[Any] = None,
        videos: Optional[List[Any]] = None,
        size: Optional[str] = "544*960",
        seed: Optional[int] = None,
        task_type: Optional[str] = "i2v",
        num_euler_timesteps: Optional[int] = 100,
        sampling_method: Optional[str] = None,
    ) -> Any:
        """Stream visual generation outputs chunk by chunk."""
        from PIL import Image

        if self.memory_module is None:
            raise ValueError("memory_module is None")

        rank = self._dist_rank()

        if isinstance(interactions, str):
            interactions = [interactions]
        if interactions is None or len(interactions) == 0:
            raise ValueError("interactions must be provided in stream().")

        if isinstance(interaction_speeds, (float, int)):
            interaction_speeds = [float(interaction_speeds)] * len(interactions)
        if isinstance(interaction_distances, (float, int)):
            interaction_distances = [float(interaction_distances)] * len(interactions)

        if videos is not None and not (
            isinstance(videos, list) and (len(videos) == 0 or isinstance(videos[0], Image.Image))
        ):
            videos = self._to_pil_frames(videos)

        if images is not None or videos is not None:
            visual_context = self.operators.process_perception(images=images, videos=videos, size=size)
            input_data = images if images is not None else videos
            self.memory_module.record(
                input_data,
                visual_context=visual_context,
                as_context=True,
                record_frames=False,
            )

        ctx = self.memory_module.select_context()
        if task_type != "t2v" and ctx is None:
            raise ValueError("No context in memory. Provide 'images' or 'videos' in the first stream() call.")

        self.operators.get_interaction(interactions)
        operator_condition = self.operators.process_interaction()
        self.operators.delete_last_interaction()

        effective_task_type = task_type
        predict_image = None if ctx is None else ctx.get("ref_images")
        predict_video = None if ctx is None else ctx.get("ref_videos")

        if task_type == "i2v" and images is None and videos is None and predict_image is None:
            if predict_video is not None and getattr(self.memory_module, "n_generated_segments", 0) > 0:
                effective_task_type = "v2v"
            else:
                raise ValueError("No valid image/video context for i2v stream continuation.")

        if effective_task_type == "i2v":
            predict_video = None

        output_video = self.synthesis_model.predict(
            prompt=prompt,
            image=predict_image,
            video=predict_video,
            interactions=interactions,
            interaction_captions=operator_condition,
            interaction_speeds=interaction_speeds,
            interaction_distances=interaction_distances,
            task_type=effective_task_type,
            size=size,
            seed=seed,
            num_euler_timesteps=num_euler_timesteps,
            sampling_method=sampling_method,
        )

        output_video_frames = self._to_pil_frames(output_video)
        if len(output_video_frames) == 0:
            raise RuntimeError("Synthesis returned an empty video in stream().")
        output_visual_context = self.operators.process_perception(
            images=output_video_frames[-1],
            videos=output_video_frames,
            size=size,
        )
        self.memory_module.record(
            output_video_frames,
            visual_context=output_visual_context,
            record_frames=(rank == 0),
        )

        return output_video
