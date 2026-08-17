# SPDX-License-Identifier: Apache-2.0
"""Audited in-memory preprocessing for the official VideoAlign reward."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from worldfoundry.training.recipes.post_training.rewards.videoalign import VideoAlignRewardSpec

VIDEOALIGN_SPECIAL_TOKENS = (
    "<|VQ_reward|>",
    "<|MQ_reward|>",
    "<|TA_reward|>",
)
VIDEOALIGN_SPECIAL_TOKEN_IDS = (151657, 151658, 151659)
VIDEOALIGN_IMAGE_FACTOR = 28
VIDEOALIGN_MIN_FRAME_PIXELS = 128 * VIDEOALIGN_IMAGE_FACTOR * VIDEOALIGN_IMAGE_FACTOR

# Adapted from the official MIT-licensed ``detailed_special`` prompt. It is a
# checked-in model input.
VIDEOALIGN_DETAILED_PROMPT = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

Please provide the ratings of Visual Quality: <|VQ_reward|>
END

**Motion Quality:**
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.

Please provide the ratings of Motion Quality: <|MQ_reward|>
END

**Text Alignment:**
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Text Alignment: <|TA_reward|>
END
"""


def build_videoalign_prompt(prompt: str) -> str:
    """Insert user text without interpreting braces as format syntax."""

    value = str(prompt).strip()
    if not value:
        raise ValueError("VideoAlign prompt must be a non-empty string")
    return VIDEOALIGN_DETAILED_PROMPT.replace("{text_prompt}", value)


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def videoalign_frame_indices(
    total_frames: int,
    *,
    source_fps: float,
    target_fps: float = 2.0,
    min_frames: int = 4,
    max_frames: int = 768,
    frame_factor: int = 2,
) -> tuple[int, ...]:
    """Reproduce VideoAlign's uniform FPS sampling without an encoded MP4."""

    if isinstance(total_frames, bool) or int(total_frames) < frame_factor:
        raise ValueError("VideoAlign requires at least frame_factor source frames")
    total = int(total_frames)
    for name, value in (("source_fps", source_fps), ("target_fps", target_fps)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (
        ("min_frames", min_frames),
        ("max_frames", max_frames),
        ("frame_factor", frame_factor),
    ):
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    factor = int(frame_factor)
    minimum = _ceil_by_factor(int(min_frames), factor)
    maximum = _floor_by_factor(min(int(max_frames), total), factor)
    requested = total / float(source_fps) * float(target_fps)
    sampled = _round_by_factor(min(max(requested, minimum), maximum), factor)
    sampled = min(sampled, total)
    if not factor <= sampled <= total:
        raise ValueError(f"VideoAlign sampled frame count must be in [{factor}, {total}]; got {sampled}")
    indices = torch.linspace(0, total - 1, sampled).round().to(torch.int64)
    return tuple(int(value) for value in indices.tolist())


def videoalign_frame_size(
    height: int,
    width: int,
    *,
    min_pixels: int = VIDEOALIGN_MIN_FRAME_PIXELS,
    max_pixels: int = 200_704,
    factor: int = VIDEOALIGN_IMAGE_FACTOR,
) -> tuple[int, int]:
    """Match Qwen-VL's factor-preserving per-frame resize contract."""

    if any(isinstance(value, bool) or int(value) <= 0 for value in (height, width, min_pixels, max_pixels, factor)):
        raise ValueError("VideoAlign frame dimensions and pixel bounds must be positive")
    height = int(height)
    width = int(width)
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)
    factor = int(factor)
    if min_pixels > max_pixels:
        raise ValueError("VideoAlign min_pixels cannot exceed max_pixels")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("VideoAlign frame aspect ratio must not exceed 200")
    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    pixels = resized_height * resized_width
    if pixels > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, _floor_by_factor(height / beta, factor))
        resized_width = max(factor, _floor_by_factor(width / beta, factor))
    elif pixels < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


@dataclass(frozen=True, slots=True)
class PreparedVideoAlignVideo:
    frames: torch.Tensor
    frame_indices: tuple[int, ...]
    source_shape: tuple[int, int, int]
    resized_shape: tuple[int, int, int]


def prepare_videoalign_video(
    video: torch.Tensor,
    spec: VideoAlignRewardSpec,
) -> PreparedVideoAlignVideo:
    """Convert one decoded ``[C,T,H,W]`` video to in-memory Qwen frames."""

    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise TypeError("VideoAlign video artifact must be a [C,T,H,W] torch.Tensor")
    if int(video.shape[0]) != 3:
        raise ValueError("VideoAlign video artifact must have exactly three channels")
    if not video.is_floating_point():
        raise TypeError("decoded VideoAlign video must use a floating dtype")
    if not bool(torch.isfinite(video).all()):
        raise ValueError("decoded VideoAlign video contains NaN or infinity")
    lower, upper = (-1.0, 1.0) if spec.input_range == "minus-one-to-one" else (0.0, 1.0)
    tolerance = 1.0e-4
    minimum = float(video.amin().item())
    maximum = float(video.amax().item())
    if minimum < lower - tolerance or maximum > upper + tolerance:
        raise ValueError(f"decoded VideoAlign video must remain in [{lower}, {upper}]; observed [{minimum}, {maximum}]")
    normalized = video.detach().float().clamp(lower, upper)
    if spec.input_range == "minus-one-to-one":
        normalized = normalized.add(1.0).mul(0.5)
    indices = videoalign_frame_indices(
        int(normalized.shape[1]),
        source_fps=spec.source_fps,
        target_fps=spec.target_fps,
        min_frames=spec.min_frames,
        max_frames=spec.max_frames,
        frame_factor=spec.frame_factor,
    )
    frames = normalized[:, list(indices)].permute(1, 0, 2, 3).contiguous()
    resized_height, resized_width = videoalign_frame_size(
        int(frames.shape[-2]),
        int(frames.shape[-1]),
        max_pixels=spec.max_frame_pixels,
    )
    if (resized_height, resized_width) != tuple(frames.shape[-2:]):
        frames = F.interpolate(
            frames,
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
    if spec.quantize_to_uint8:
        frames = frames.mul(255.0).round().div(255.0)
    frames = frames.cpu()
    return PreparedVideoAlignVideo(
        frames=frames,
        frame_indices=indices,
        source_shape=(int(video.shape[1]), int(video.shape[2]), int(video.shape[3])),
        resized_shape=(int(frames.shape[0]), resized_height, resized_width),
    )


def pool_videoalign_special_tokens(
    sequence_scores: torch.Tensor,
    input_ids: torch.Tensor,
    special_token_ids: Sequence[int] = VIDEOALIGN_SPECIAL_TOKEN_IDS,
) -> torch.Tensor:
    """Select one scalar at each ordered reward token, rejecting ambiguity."""

    if not isinstance(sequence_scores, torch.Tensor) or sequence_scores.ndim != 3:
        raise TypeError("VideoAlign sequence scores must have shape [B,L,1]")
    if sequence_scores.shape[-1] != 1:
        raise ValueError("VideoAlign reward head output dimension must be one")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise TypeError("VideoAlign input_ids must have shape [B,L]")
    if tuple(sequence_scores.shape[:2]) != tuple(input_ids.shape):
        raise ValueError("VideoAlign sequence scores and input_ids must share [B,L]")
    token_ids = tuple(int(value) for value in special_token_ids)
    if len(token_ids) != 3 or len(set(token_ids)) != 3:
        raise ValueError("VideoAlign requires three unique special token ids")
    batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
    selected: list[torch.Tensor] = []
    for token_id in token_ids:
        mask = input_ids.eq(token_id)
        counts = mask.sum(dim=1)
        if not bool(counts.eq(1).all()):
            raise ValueError(
                f"VideoAlign token {token_id} must occur exactly once per sample; counts={counts.tolist()}"
            )
        positions = mask.to(torch.int64).argmax(dim=1)
        selected.append(sequence_scores[batch_indices, positions, 0])
    return torch.stack(selected, dim=1)


__all__ = [
    "PreparedVideoAlignVideo",
    "VIDEOALIGN_DETAILED_PROMPT",
    "VIDEOALIGN_IMAGE_FACTOR",
    "VIDEOALIGN_MIN_FRAME_PIXELS",
    "VIDEOALIGN_SPECIAL_TOKEN_IDS",
    "VIDEOALIGN_SPECIAL_TOKENS",
    "build_videoalign_prompt",
    "pool_videoalign_special_tokens",
    "prepare_videoalign_video",
    "videoalign_frame_indices",
    "videoalign_frame_size",
]
