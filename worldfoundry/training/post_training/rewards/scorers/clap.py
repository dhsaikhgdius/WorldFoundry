"""CLAP audio-text alignment over decoded waveform artifacts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Lock

import torch

from ..contracts import RewardRequest
from ._model import move_inputs, resolve_device
from .config import CLAPConfig

CLAP_SAMPLE_RATE = 48_000
Resample = Callable[[torch.Tensor, int, int], torch.Tensor]


def _default_resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    try:
        from torchaudio.functional import resample
    except ImportError as error:
        raise RuntimeError("CLAP sample-rate conversion requires torchaudio") from error
    return resample(waveform.unsqueeze(0), orig_freq=source_rate, new_freq=target_rate).squeeze(0)


def _mono_waveform(audio: object) -> torch.Tensor:
    if not isinstance(audio, torch.Tensor) or audio.ndim not in {1, 2}:
        raise ValueError("CLAP requires an audio waveform tensor shaped [L], [C,L], or [L,C]")
    waveform = audio.detach().float()
    if waveform.ndim == 2:
        channel_axis = 0 if waveform.shape[0] <= waveform.shape[1] else 1
        waveform = waveform.mean(dim=channel_axis)
    return waveform.reshape(-1)


class CLAPScorer:
    """Lazy-loaded CLAP scorer over audio plus metadata ``audio_sampling_rate``."""

    def __init__(
        self,
        config: CLAPConfig | None = None,
        *,
        processor: object | None = None,
        model: object | None = None,
        resample: Resample | None = None,
    ) -> None:
        if (processor is None) != (model is None):
            raise ValueError("processor and model must be supplied together")
        self.config = config or CLAPConfig()
        self.device = resolve_device(self.config.device)
        self._processor = processor
        self._model = model
        self._resample = resample or _default_resample
        self._load_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> tuple[object, object]:
        if self._processor is None or self._model is None:
            with self._load_lock:
                if self._processor is None or self._model is None:
                    try:
                        from transformers import ClapModel, ClapProcessor
                    except ImportError as error:
                        raise RuntimeError("CLAP requires transformers from the train-core extra") from error
                    self._processor = ClapProcessor.from_pretrained(self.config.model_id)
                    self._model = ClapModel.from_pretrained(self.config.model_id)
                    self._model.to(device=self.device, dtype=torch.float32).eval()
                    self._model.requires_grad_(False)
        return self._processor, self._model

    def _audio(self, request: RewardRequest) -> torch.Tensor:
        waveform = _mono_waveform(request.artifacts.get("audio"))
        sampling_rate = request.metadata.get("audio_sampling_rate")
        if not isinstance(sampling_rate, int) or isinstance(sampling_rate, bool) or sampling_rate <= 0:
            raise ValueError("CLAP requires positive integer metadata audio_sampling_rate")
        if sampling_rate != CLAP_SAMPLE_RATE:
            waveform = self._resample(waveform, sampling_rate, CLAP_SAMPLE_RATE)
        return waveform.cpu()

    def _score_batch(self, requests: Sequence[RewardRequest]) -> tuple[float, ...]:
        processor, model = self._load()
        inputs = processor(
            text=[request.prompt for request in requests],
            audios=[self._audio(request).numpy() for request in requests],
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        with torch.inference_mode():
            output = model(**move_inputs(inputs, self.device))
            audio_features = output.audio_embeds.float()
            text_features = output.text_embeds.float()
            audio_features = audio_features / audio_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            scores = (audio_features * text_features).sum(dim=-1)
        return tuple(float(value) for value in scores.cpu())

    def score(self, requests: tuple[RewardRequest, ...]) -> tuple[float, ...]:
        values: list[float] = []
        for start in range(0, len(requests), self.config.batch_size):
            values.extend(self._score_batch(requests[start : start + self.config.batch_size]))
        return tuple(values)


__all__ = ["CLAP_SAMPLE_RATE", "CLAPScorer"]
