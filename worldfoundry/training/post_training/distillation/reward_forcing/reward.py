"""VideoAlign-to-Re-DMD motion-quality bridge."""

from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor, nn

from worldfoundry.training.recipes.post_training.rewards.videoalign import (
    VIDEOALIGN_REWARD_IDS,
)

from ...rewards.contracts import RewardEvaluator, RewardRequest, RewardResult
from .contracts import RewardForcingTrainingBatch


class VideoAlignMotionQualityReward:
    """Evaluate decoded videos and return ``use_norm=True`` MQ scores."""

    def __init__(
        self,
        evaluator: RewardEvaluator,
        *,
        checkpoint_identity: str,
        calibration_mean: float,
        calibration_std: float,
        normalization_epsilon: float = 0.0,
        owned_module: nn.Module | None = None,
    ) -> None:
        if not isinstance(evaluator, RewardEvaluator):
            raise TypeError("evaluator must implement RewardEvaluator")
        identity = str(checkpoint_identity).strip()
        if not identity:
            raise ValueError("motion reward checkpoint_identity must be non-empty")
        if owned_module is not None and not isinstance(owned_module, nn.Module):
            raise TypeError("motion reward owned_module must be an nn.Module or None")
        mean = float(calibration_mean)
        std = float(calibration_std)
        epsilon = float(normalization_epsilon)
        if not isfinite(mean):
            raise ValueError("motion-quality calibration mean must be finite")
        if not isfinite(std) or std <= 0:
            raise ValueError("motion-quality calibration std must be finite and positive")
        if not isfinite(epsilon) or epsilon < 0:
            raise ValueError("normalization_epsilon must be finite and non-negative")
        self.evaluator = evaluator
        self.checkpoint_identity = identity
        self.owned_module = owned_module
        self.calibration_mean = mean
        self.calibration_std = std
        self.normalization_epsilon = epsilon

    def score_motion_quality(
        self,
        videos: Tensor,
        batch: RewardForcingTrainingBatch,
    ) -> Tensor:
        if not isinstance(batch, RewardForcingTrainingBatch):
            raise TypeError("batch must be RewardForcingTrainingBatch")
        if not isinstance(videos, Tensor) or videos.ndim != 5:
            raise TypeError("decoded reward videos must have shape [B,C,T,H,W]")
        if tuple(videos.shape[:2]) != (batch.batch_size, 3):
            raise ValueError("decoded reward videos must preserve batch size and have three channels")
        if not videos.is_floating_point() or not bool(torch.isfinite(videos).all()):
            raise ValueError("decoded reward videos must be finite floating tensors")
        requests = tuple(
            RewardRequest(
                request_id=sample_id,
                rollout_id=f"reward-forcing:{sample_id}",
                prompt=prompt,
                conditions={},
                artifacts={"video": videos[index]},
                reward_ids=VIDEOALIGN_REWARD_IDS,
            )
            for index, (sample_id, prompt) in enumerate(zip(batch.sample_ids, batch.prompts, strict=True))
        )
        with torch.inference_mode():
            results = self.evaluator.evaluate(requests)
        if not isinstance(results, tuple) or len(results) != batch.batch_size:
            raise ValueError("reward evaluator must return one ordered result per video")
        raw = torch.empty(
            (batch.batch_size,),
            device=videos.device,
            dtype=torch.float32,
        )
        for index, (request, result) in enumerate(zip(requests, results, strict=True)):
            if not isinstance(result, RewardResult):
                raise TypeError("reward evaluator returned a non-RewardResult value")
            if result.request_id != request.request_id or result.rollout_id != request.rollout_id:
                raise ValueError("reward result identity differs from its request")
            if set(result.values) != set(VIDEOALIGN_REWARD_IDS):
                raise ValueError("VideoAlign reward result has an unexpected component inventory")
            if not result.valid["motion_quality"]:
                raise ValueError("VideoAlign returned an invalid motion-quality score")
            raw[index] = float(result.values["motion_quality"])
        normalized = (raw - self.calibration_mean) / (self.calibration_std + self.normalization_epsilon)
        if not bool(torch.isfinite(normalized).all()):
            raise FloatingPointError("normalized motion-quality reward is non-finite")
        return normalized


__all__ = ["VideoAlignMotionQualityReward"]
