"""Prompt-only and data-aided SANA batches for native SiD."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch

from worldfoundry.training.api.contracts import TrainingBatch
from worldfoundry.training.data.rollout_manifest import RolloutPromptRecord
from worldfoundry.training.data.shared_conditioning import SharedConditioningSample
from worldfoundry.training.models.sana import SanaTrainAdapter
from worldfoundry.training.post_training.distillation.sid.contracts import SIDTrainingBatch


def collate_sana_sid_prompts(records: list[RolloutPromptRecord]) -> TrainingBatch:
    """Collate safe prompt-manifest rows without inventing media placeholders."""

    values = tuple(records)
    if not values or not all(isinstance(value, RolloutPromptRecord) for value in values):
        raise ValueError("SANA SiD prompt collate requires RolloutPromptRecord values")
    return TrainingBatch(
        sample_ids=tuple(value.prompt_id for value in values),
        prompts=tuple(value.prompt for value in values),
    )


def _repeat_shared(value: torch.Tensor, *, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        raise ValueError("shared SANA conditioning tensors cannot be scalar")
    dtype = reference.dtype if value.is_floating_point() else value.dtype
    return value.unsqueeze(0).expand(batch_size, *value.shape).to(
        device=reference.device,
        dtype=dtype,
    )


def _cached_unconditional(
    sample: SharedConditioningSample,
    positive: Mapping[str, object],
    *,
    batch_size: int,
) -> dict[str, object]:
    if not isinstance(sample, SharedConditioningSample):
        raise TypeError("unconditional must be SharedConditioningSample")
    if set(sample.tensors) != {"context", "context_mask"}:
        raise ValueError("SANA SiD unconditional conditioning requires context and context_mask")
    context = positive.get("context")
    context_mask = positive.get("context_mask")
    if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
        raise TypeError("prepared SANA conditioning must contain context and context_mask")
    negative: dict[str, object] = {
        "context": _repeat_shared(sample.tensors["context"], batch_size=batch_size, reference=context),
        "context_mask": _repeat_shared(
            sample.tensors["context_mask"],
            batch_size=batch_size,
            reference=context_mask,
        ),
    }
    for name in ("img_hw", "aspect_ratio", "cfg_scale"):
        try:
            negative[name] = positive[name]
        except KeyError as error:
            raise ValueError(f"prepared SANA conditioning lacks {name!r}") from error
    return negative


def _encoded_unconditional(
    adapter: SanaTrainAdapter,
    batch: TrainingBatch,
    *,
    height: int,
    width: int,
) -> Mapping[str, object]:
    conditions = {
        name: value
        for name, value in batch.conditions.items()
        if name not in {"context", "context_mask", "clean_latents", "latent_loss_mask"}
    }
    empty = TrainingBatch(
        sample_ids=tuple(f"unconditional::{value}" for value in batch.sample_ids),
        prompts=("",) * batch.batch_size,
        conditions=conditions,
        sample_weights=batch.sample_weights,
        metadata=batch.metadata,
    )
    return adapter.prepare_prompt_conditioning(empty, height=height, width=width)


def prepare_sana_sid_batch(
    adapter: SanaTrainAdapter,
    batch: TrainingBatch,
    *,
    height: int,
    width: int,
    unconditional: SharedConditioningSample | None = None,
    include_real_latents: bool = False,
) -> SIDTrainingBatch:
    """Materialize prompt-only geometry, optionally retaining real data for GAN."""

    if not isinstance(adapter, SanaTrainAdapter):
        raise TypeError("adapter must be SanaTrainAdapter")
    if not isinstance(batch, TrainingBatch):
        raise TypeError("batch must be TrainingBatch")
    if not isinstance(include_real_latents, bool):
        raise TypeError("include_real_latents must be bool")
    has_media = batch.pixel_values is not None or "clean_latents" in batch.conditions
    if include_real_latents and not has_media:
        raise ValueError("data-aided SiD requires pixels or cached clean_latents")

    if has_media:
        prepared = adapter.prepare_batch(batch)
        if not isinstance(prepared.clean_latents, torch.Tensor):
            raise TypeError("SANA SiD requires one latent tensor")
        positive = dict(prepared.conditioning)
        template = torch.empty_like(prepared.clean_latents)
        sample_weights = prepared.sample_weights
        inferred_height = int(prepared.clean_latents.shape[-2]) * adapter.spatial_compression
        inferred_width = int(prepared.clean_latents.shape[-1]) * adapter.spatial_compression
        if int(height) != inferred_height or int(width) != inferred_width:
            raise ValueError("configured SiD prompt geometry differs from the prepared SANA latents")
        real_latents = prepared.clean_latents if include_real_latents else None
    else:
        positive = dict(adapter.prepare_prompt_conditioning(batch, height=height, width=width))
        template = adapter.allocate_latent_template(
            batch_size=batch.batch_size,
            height=height,
            width=width,
        )
        sample_weights = batch.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.to(device=template.device, dtype=torch.float32)
        real_latents = None

    negative = (
        _cached_unconditional(unconditional, positive, batch_size=batch.batch_size)
        if unconditional is not None
        else _encoded_unconditional(adapter, batch, height=height, width=width)
    )
    return SIDTrainingBatch(
        sample_ids=batch.sample_ids,
        latent_template=template,
        conditioning=positive,
        unconditional_conditioning=negative,
        sample_weights=sample_weights,
        real_sample_ids=(
            tuple(f"real::{value}" for value in batch.sample_ids)
            if real_latents is not None
            else ()
        ),
        real_latents=real_latents,
        real_conditioning=positive if real_latents is not None else None,
    )


class SanaSIDDataLoader:
    """Checkpoint-transparent conversion over a prompt or SANA-cache loader."""

    def __init__(
        self,
        dataloader: object,
        *,
        adapter: SanaTrainAdapter,
        height: int,
        width: int,
        unconditional: SharedConditioningSample | None = None,
        include_real_latents: bool = False,
    ) -> None:
        if not callable(getattr(dataloader, "__iter__", None)):
            raise TypeError("dataloader must be iterable")
        if not callable(getattr(dataloader, "state_dict", None)) or not callable(
            getattr(dataloader, "load_state_dict", None)
        ):
            raise TypeError("SANA SiD dataloader must be checkpointable")
        self.dataloader = dataloader
        self.adapter = adapter
        self.height = int(height)
        self.width = int(width)
        self.unconditional = unconditional
        self.include_real_latents = include_real_latents

    def __iter__(self) -> Iterator[SIDTrainingBatch]:
        for batch in self.dataloader:
            if not isinstance(batch, TrainingBatch):
                raise TypeError("SANA SiD source loader must emit TrainingBatch values")
            yield prepare_sana_sid_batch(
                self.adapter,
                batch,
                height=self.height,
                width=self.width,
                unconditional=self.unconditional,
                include_real_latents=self.include_real_latents,
            )

    def state_dict(self) -> Mapping[str, object]:
        state = self.dataloader.state_dict()
        if not isinstance(state, Mapping):
            raise TypeError("SANA SiD dataloader state_dict must return a mapping")
        return state

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.dataloader.load_state_dict(state_dict)


__all__ = ["SanaSIDDataLoader", "collate_sana_sid_prompts", "prepare_sana_sid_batch"]
