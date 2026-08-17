# SPDX-License-Identifier: Apache-2.0
"""Native VideoAlign reward inference.

The reward-head architecture, prompt, preprocessing constants, and LoRA
topology are adapted from KlingAIResearch/VideoAlign (MIT) and checked against
the official ``KlingTeam/VideoReward`` checkpoint. Reference repositories are
neither imported nor executed by WorldFoundry.

Sources:
https://github.com/KlingAIResearch/VideoAlign
https://huggingface.co/KlingTeam/VideoReward
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.loaders import (
    CheckpointSpec,
    NativeCheckpointResolver,
)
from worldfoundry.training.recipes.post_training.rewards.videoalign import (
    VIDEOALIGN_REWARD_IDS,
    VideoAlignRewardSpec,
)

from .contracts import RewardEvaluator, RewardRequest, RewardResult
from .videoalign_model import NativeVideoAlignRewardModel, load_videoalign_checkpoint
from .videoalign_preprocessing import (
    VIDEOALIGN_SPECIAL_TOKEN_IDS,
    VIDEOALIGN_SPECIAL_TOKENS,
    PreparedVideoAlignVideo,
    build_videoalign_prompt,
    prepare_videoalign_video,
)

_BASE_MODEL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "preprocessor_config.json",
)
_REWARD_TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _audit_official_model_config(path: Path, spec: VideoAlignRewardSpec) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid VideoAlign model_config.json: {path}") from error
    if not isinstance(payload, Mapping):
        raise TypeError("VideoAlign model_config.json must contain an object")
    data = payload.get("data_config")
    model = payload.get("model_config")
    peft = payload.get("peft_lora_config")
    inference = payload.get("inference_config")
    if not all(isinstance(value, Mapping) for value in (data, model, peft, inference)):
        raise ValueError("VideoAlign model_config.json is missing typed config sections")
    expected_data = {
        "max_frame_pixels": spec.max_frame_pixels,
        "fps": spec.target_fps,
        "eval_dim": ["VQ", "MQ", "TA"],
        "prompt_template_type": "detailed_special",
        "sample_type": "uniform",
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:  # type: ignore[union-attr]
            raise ValueError(f"VideoAlign data_config.{key} differs from the native contract")
    expected_model = {
        "model_name_or_path": spec.base_model_repository,
        "output_dim": 1,
        "use_special_tokens": True,
        "reward_token": "special",
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:  # type: ignore[union-attr]
            raise ValueError(f"VideoAlign model_config.{key} differs from the native contract")
    expected_peft = {
        "lora_enable": True,
        "vision_lora": False,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_task_type": "CAUSAL_LM",
        "use_rslora": False,
        "num_lora_modules": -1,
    }
    for key, expected in expected_peft.items():
        if peft.get(key) != expected:  # type: ignore[union-attr]
            raise ValueError(f"VideoAlign peft_lora_config.{key} differs from the native contract")
    if peft.get("lora_namespan_exclude") != [  # type: ignore[union-attr]
        "lm_head",
        "rm_head",
        "embed_tokens",
        "visual",
    ]:
        raise ValueError("VideoAlign LoRA exclusion list differs from the official checkpoint")
    inference_expected = {
        "VQ_mean": spec.calibration_mean["video_quality"],
        "VQ_std": spec.calibration_std["video_quality"],
        "MQ_mean": spec.calibration_mean["motion_quality"],
        "MQ_std": spec.calibration_std["motion_quality"],
        "TA_mean": spec.calibration_mean["text_alignment"],
        "TA_std": spec.calibration_std["text_alignment"],
    }
    for key, expected in inference_expected.items():
        actual = inference.get(key)  # type: ignore[union-attr]
        if not isinstance(actual, (float, int)) or float(actual) != float(expected):
            raise ValueError(f"VideoAlign inference_config.{key} differs from the recipe")


def _videoalign_base_checkpoint(spec: VideoAlignRewardSpec) -> CheckpointSpec:
    return CheckpointSpec(
        repo_id=spec.base_model_repository,
        revision=spec.base_model_revision,
        files=_BASE_MODEL_FILES,
        allow_patterns=_BASE_MODEL_FILES,
    )


def _videoalign_reward_checkpoint(spec: VideoAlignRewardSpec) -> CheckpointSpec:
    tokenizer_root = Path(spec.checkpoint_file).parent / "tokenizer"
    tokenizer_files = tuple((tokenizer_root / filename).as_posix() for filename in _REWARD_TOKENIZER_FILES)
    files = (spec.checkpoint_file, "model_config.json", *tokenizer_files)
    return CheckpointSpec(
        repo_id=spec.checkpoint_repository,
        revision=spec.checkpoint_revision,
        files=files,
        allow_patterns=files,
        file_size_bytes={spec.checkpoint_file: spec.checkpoint_size_bytes},
    )


def _model_logits(output: object) -> torch.Tensor:
    logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("VideoAlign model output must expose tensor logits")
    return logits


class VideoAlignRewardEvaluator(RewardEvaluator):
    """Score decoded videos in memory and return unnormalized VQ/MQ/TA."""

    schema = "worldfoundry-videoalign-reward"

    def __init__(
        self,
        model: nn.Module,
        processor: object,
        spec: VideoAlignRewardSpec,
        *,
        device: str | torch.device,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("VideoAlign evaluator model must be an nn.Module")
        if not callable(processor) or not callable(getattr(processor, "apply_chat_template", None)):
            raise TypeError("VideoAlign processor must be callable and expose apply_chat_template")
        if not isinstance(spec, VideoAlignRewardSpec):
            raise TypeError("VideoAlign evaluator spec must be VideoAlignRewardSpec")
        self.model = model.requires_grad_(False).eval()
        self.processor = processor
        self.spec = spec
        self.device = torch.device(device)
        self.model.to(device=self.device)
        self._identity = MappingProxyType(
            {
                "schema": self.schema,
                "base_model": {
                    "repository": spec.base_model_repository,
                    "revision": spec.base_model_revision,
                },
                "reward_checkpoint": {
                    "repository": spec.checkpoint_repository,
                    "revision": spec.checkpoint_revision,
                    "file": spec.checkpoint_file,
                    "size_bytes": spec.checkpoint_size_bytes,
                },
                "reward_ids": list(VIDEOALIGN_REWARD_IDS),
                "special_tokens": dict(
                    zip(
                        VIDEOALIGN_SPECIAL_TOKENS,
                        VIDEOALIGN_SPECIAL_TOKEN_IDS,
                        strict=True,
                    )
                ),
                "preprocessing": {
                    "source_fps": spec.source_fps,
                    "target_fps": spec.target_fps,
                    "min_frames": spec.min_frames,
                    "max_frames": spec.max_frames,
                    "frame_factor": spec.frame_factor,
                    "max_frame_pixels": spec.max_frame_pixels,
                    "quantize_to_uint8": spec.quantize_to_uint8,
                    "input_range": spec.input_range,
                },
            }
        )

    @property
    def identity(self) -> dict[str, object]:
        return dict(self._identity)

    def _evaluate_batch(
        self,
        requests: tuple[RewardRequest, ...],
    ) -> tuple[RewardResult, ...]:
        prepared: list[PreparedVideoAlignVideo] = []
        chats: list[list[dict[str, object]]] = []
        for request in requests:
            if request.reward_ids != VIDEOALIGN_REWARD_IDS:
                raise ValueError("VideoAlign request reward_ids differ from the model contract")
            prepared_video = prepare_videoalign_video(
                request.artifacts.get("video"),  # type: ignore[arg-type]
                self.spec,
            )
            prepared.append(prepared_video)
            chats.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {
                                "type": "text",
                                "text": build_videoalign_prompt(request.prompt),
                            },
                        ],
                    }
                ]
            )
        texts = self.processor.apply_chat_template(  # type: ignore[union-attr]
            chats,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.processor(  # type: ignore[operator]
            text=texts,
            images=None,
            videos=[value.frames for value in prepared],
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": False},
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("VideoAlign processor must return a tensor mapping")
        model_inputs = {
            str(key): value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }
        started = perf_counter()
        with torch.inference_mode():
            logits = _model_logits(self.model(return_dict=True, **model_inputs)).float().cpu()
        elapsed_ms = (perf_counter() - started) * 1000.0
        expected_shape = (len(requests), len(VIDEOALIGN_REWARD_IDS))
        if tuple(logits.shape) != expected_shape:
            raise ValueError(f"VideoAlign logits must have shape {expected_shape}; got {tuple(logits.shape)}")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("VideoAlign returned NaN or infinity")
        results: list[RewardResult] = []
        per_sample_ms = elapsed_ms / len(requests)
        for index, (request, prepared_video) in enumerate(zip(requests, prepared, strict=True)):
            diagnostic = MappingProxyType(
                {
                    "frame_indices": list(prepared_video.frame_indices),
                    "source_shape": list(prepared_video.source_shape),
                    "resized_shape": list(prepared_video.resized_shape),
                }
            )
            values = {
                reward_id: float(logits[index, reward_index].item())
                for reward_index, reward_id in enumerate(VIDEOALIGN_REWARD_IDS)
            }
            results.append(
                RewardResult(
                    request_id=request.request_id,
                    rollout_id=request.rollout_id,
                    values=values,
                    valid={reward_id: True for reward_id in VIDEOALIGN_REWARD_IDS},
                    diagnostics=diagnostic,
                    latency_ms=per_sample_ms,
                )
            )
        return tuple(results)

    def evaluate(
        self,
        requests: tuple[RewardRequest, ...],
    ) -> tuple[RewardResult, ...]:
        if not isinstance(requests, tuple) or any(not isinstance(request, RewardRequest) for request in requests):
            raise TypeError("VideoAlign evaluator requests must be a tuple of RewardRequest")
        results: list[RewardResult] = []
        for start in range(0, len(requests), self.spec.batch_size):
            results.extend(self._evaluate_batch(requests[start : start + self.spec.batch_size]))
        return tuple(results)


def _validate_base_loading_info(loading_info: Mapping[str, object]) -> None:
    missing = set(str(value) for value in loading_info.get("missing_keys", ()))
    unexpected = set(str(value) for value in loading_info.get("unexpected_keys", ()))
    mismatched = tuple(loading_info.get("mismatched_keys", ()))
    errors = tuple(loading_info.get("error_msgs", ()))
    if missing != {"rm_head.weight"} or unexpected or mismatched or errors:
        raise RuntimeError(
            "pinned Qwen2-VL base loading differed from the VideoAlign subclass contract: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"mismatched={mismatched}, errors={errors}"
        )


def build_videoalign_reward_evaluator(
    spec: VideoAlignRewardSpec,
    *,
    device: str | torch.device = "cuda",
    attention_implementation: str = "sdpa",
    resolver: NativeCheckpointResolver | None = None,
) -> VideoAlignRewardEvaluator:
    """Materialize pinned assets and strictly construct native VideoAlign."""

    if not isinstance(spec, VideoAlignRewardSpec):
        raise TypeError("spec must be VideoAlignRewardSpec")
    if attention_implementation not in {"sdpa", "flash_attention_2"}:
        raise ValueError("VideoAlign attention must be sdpa or flash_attention_2")
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            Qwen2VLProcessor,
            Qwen2VLVideoProcessor,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError("VideoAlign requires the train-core Transformers and PEFT dependencies") from error
    active_resolver = resolver or NativeCheckpointResolver()
    base = active_resolver.materialize(_videoalign_base_checkpoint(spec))
    reward = active_resolver.materialize(_videoalign_reward_checkpoint(spec))
    _audit_official_model_config(reward.root / "model_config.json", spec)
    tokenizer_root = reward.root / Path(spec.checkpoint_file).parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        local_files_only=True,
        padding_side="right",
    )
    resolved_ids = tuple(int(value) for value in tokenizer.convert_tokens_to_ids(VIDEOALIGN_SPECIAL_TOKENS))
    if resolved_ids != VIDEOALIGN_SPECIAL_TOKEN_IDS or len(tokenizer) != 151_660:
        raise ValueError(
            "VideoAlign tokenizer identity differs from the official checkpoint: "
            f"ids={resolved_ids}, size={len(tokenizer)}"
        )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise ValueError("VideoAlign tokenizer does not provide a chat template")
    image_processor = AutoImageProcessor.from_pretrained(
        base.root,
        local_files_only=True,
        # The pinned official environment predates Transformers' default
        # switch to the fast processor. Keep preprocessing identity stable.
        use_fast=False,
    )
    video_processor = Qwen2VLVideoProcessor.from_pretrained(
        base.root,
        local_files_only=True,
    )
    processor = Qwen2VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=chat_template,
    )
    dtype = torch.bfloat16 if spec.dtype == "bfloat16" else torch.float16
    loaded = NativeVideoAlignRewardModel.from_pretrained(
        base.root,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=attention_implementation,
        low_cpu_mem_usage=True,
        output_loading_info=True,
        output_dim=1,
        special_token_ids=resolved_ids,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("Transformers did not return VideoAlign base loading diagnostics")
    base_model, loading_info = loaded
    if not isinstance(base_model, NativeVideoAlignRewardModel) or not isinstance(
        loading_info,
        Mapping,
    ):
        raise RuntimeError("Transformers returned an invalid VideoAlign base load result")
    _validate_base_loading_info(loading_info)
    base_model.resize_token_embeddings(len(tokenizer))
    excluded = ("lm_head", "rm_head", "embed_tokens", "visual")
    target_modules = [
        name
        for name, module in base_model.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding)) and not any(value in name for value in excluded)
    ]
    if not target_modules or len(target_modules) != len(set(target_modules)):
        raise RuntimeError("VideoAlign LoRA target discovery returned an invalid module set")
    peft_model = get_peft_model(
        base_model,
        LoraConfig(
            target_modules=target_modules,
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            use_rslora=False,
            bias="none",
        ),
    )
    load_videoalign_checkpoint(peft_model, reward.root / spec.checkpoint_file)
    peft_model.requires_grad_(False).eval()
    peft_model.to(dtype=dtype)
    return VideoAlignRewardEvaluator(peft_model, processor, spec, device=device)


__all__ = ["VideoAlignRewardEvaluator", "build_videoalign_reward_evaluator"]
