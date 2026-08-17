"""PickScore on the first frame of a decoded channel-first video."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

import torch

from ..contracts import RewardRequest
from ._model import feature_tensor, move_inputs, resolve_device
from .config import VideoPickScoreConfig


def _first_frame_pixels(video: object) -> object:
    """Convert a ``[C,T,H,W]`` video artifact to first-frame HWC pixels."""

    if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[0] not in {1, 3, 4}:
        raise ValueError("VideoPickScore requires a [C,T,H,W] video tensor with 1, 3, or 4 channels")
    if video.shape[1] == 0:
        raise ValueError("VideoPickScore requires at least one video frame")
    frame = video[:, 0].detach().cpu()
    if not frame.is_floating_point():
        frame = frame.float().div(255.0)
    elif frame.numel() and frame.max() > 1.0:
        frame = frame.div(255.0)
    frame = frame.clamp(0.0, 1.0)
    if frame.shape[0] == 1:
        frame = frame.repeat(3, 1, 1)
    elif frame.shape[0] == 4:
        frame = frame[:3]
    return frame.permute(1, 2, 0).mul(255).to(torch.uint8).numpy()


class VideoPickScoreScorer:
    """Lazy-loaded PickScore scorer over ``RewardRequest.artifacts['video']``."""

    def __init__(
        self,
        config: VideoPickScoreConfig | None = None,
        *,
        processor: object | None = None,
        model: object | None = None,
    ) -> None:
        if (processor is None) != (model is None):
            raise ValueError("processor and model must be supplied together")
        self.config = config or VideoPickScoreConfig()
        self.device = resolve_device(self.config.device)
        self._processor = processor
        self._model = model
        self._load_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> tuple[object, object]:
        if self._processor is None or self._model is None:
            with self._load_lock:
                if self._processor is None or self._model is None:
                    try:
                        from transformers import CLIPModel, CLIPProcessor
                    except ImportError as error:
                        raise RuntimeError("VideoPickScore requires transformers from the train-core extra") from error
                    self._processor = CLIPProcessor.from_pretrained(self.config.processor_id)
                    self._model = CLIPModel.from_pretrained(self.config.model_id)
                    self._model.to(device=self.device, dtype=torch.float32).eval()
                    self._model.requires_grad_(False)
        return self._processor, self._model

    def _score_batch(self, requests: Sequence[RewardRequest]) -> tuple[float, ...]:
        processor, model = self._load()
        images = [_first_frame_pixels(request.artifacts.get("video")) for request in requests]
        prompts = [request.prompt for request in requests]
        image_inputs = processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        with torch.inference_mode():
            image_features = feature_tensor(model.get_image_features(**move_inputs(image_inputs, self.device))).float()
            text_features = feature_tensor(model.get_text_features(**move_inputs(text_inputs, self.device))).float()
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            scores = model.logit_scale.float().exp() * (text_features @ image_features.T)
            scores = scores.diag().div(26.0)
        return tuple(float(value) for value in scores.cpu())

    def score(self, requests: tuple[RewardRequest, ...]) -> tuple[float, ...]:
        values: list[float] = []
        for start in range(0, len(requests), self.config.batch_size):
            values.extend(self._score_batch(requests[start : start + self.config.batch_size]))
        return tuple(values)


__all__ = ["VideoPickScoreScorer"]
