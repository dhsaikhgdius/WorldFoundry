"""Audited SANA cache conversion for native SCM-LADD sessions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch

from worldfoundry.training.api.contracts import TrainingBatch
from worldfoundry.training.data.sana_cache import SanaCachedDataset, text_sha256
from worldfoundry.training.data.shared_conditioning import SharedConditioningSample
from worldfoundry.training.models.sana import SanaTrainAdapter
from worldfoundry.training.post_training.distillation.scm_ladd.contracts import (
    SCMLADDTrainingBatch,
)


def audit_sana_scm_ladd_unconditional(
    sample: SharedConditioningSample,
    dataset: SanaCachedDataset,
) -> None:
    """Bind the empty prompt tensors to the opened SANA cache contract."""

    if not isinstance(sample, SharedConditioningSample):
        raise TypeError("unconditional conditioning must be SharedConditioningSample")
    if not isinstance(dataset, SanaCachedDataset):
        raise TypeError("dataset must be SanaCachedDataset")
    identity = sample.artifact.identity
    if identity.branch != "unconditional" or identity.prompt_sha256 != text_sha256(""):
        raise ValueError("SANA SCM-LADD requires an empty-prompt unconditional artifact")
    provenances = tuple(entry.provenance for entry in dataset.index.entries)
    if {value.model_recipe_digest for value in provenances} != {identity.model_recipe_digest}:
        raise ValueError("SANA unconditional conditioning belongs to another model contract")
    if {
        (value.conditioner_digest, value.tokenizer_digest) for value in provenances
    } != {(identity.conditioner_digest, identity.tokenizer_digest)}:
        raise ValueError("SANA unconditional conditioning encoder identity differs from the cache")
    if set(sample.tensors) != {"context", "context_mask"}:
        raise ValueError("SANA unconditional conditioning must contain context and context_mask")
    reference = dataset.index.entries[0].tensors
    for name, layout in (("context", "one-sequence-features"), ("context_mask", "sequence")):
        descriptor = identity.tensors[name]
        if descriptor.shape != reference[name].shape or descriptor.layout != layout:
            raise ValueError(f"SANA unconditional {name} tensor contract is incompatible")


def _repeat_shared_tensor(
    value: torch.Tensor,
    *,
    batch_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if value.ndim == 0:
        raise ValueError("shared SANA conditioning tensors cannot be scalar")
    expanded = value.unsqueeze(0).expand(batch_size, *value.shape)
    dtype = reference.dtype if value.is_floating_point() else value.dtype
    return expanded.to(device=reference.device, dtype=dtype)


def prepare_sana_scm_ladd_batch(
    adapter: SanaTrainAdapter,
    batch: TrainingBatch,
    unconditional: SharedConditioningSample,
) -> SCMLADDTrainingBatch:
    """Reuse SANA latent/conditioning validation, then attach empty-prompt CFG."""

    if not isinstance(adapter, SanaTrainAdapter):
        raise TypeError("adapter must be SanaTrainAdapter")
    if not isinstance(batch, TrainingBatch):
        raise TypeError("batch must be TrainingBatch")
    if not isinstance(unconditional, SharedConditioningSample):
        raise TypeError("unconditional must be SharedConditioningSample")
    prepared = adapter.prepare_batch(batch)
    positive = dict(prepared.conditioning)
    context = positive.get("context")
    context_mask = positive.get("context_mask")
    if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
        raise TypeError("prepared SANA conditioning must contain context and context_mask")
    negative: dict[str, object] = {
        "context": _repeat_shared_tensor(
            unconditional.tensors["context"],
            batch_size=prepared.batch_size,
            reference=context,
        ),
        "context_mask": _repeat_shared_tensor(
            unconditional.tensors["context_mask"],
            batch_size=prepared.batch_size,
            reference=context_mask,
        ),
    }
    for name in ("img_hw", "aspect_ratio", "cfg_scale"):
        try:
            negative[name] = positive[name]
        except KeyError as error:
            raise ValueError(f"prepared SANA conditioning lacks {name!r}") from error
    return SCMLADDTrainingBatch(
        sample_ids=prepared.sample_ids,
        clean_latents=prepared.clean_latents,
        conditioning=positive,
        unconditional_conditioning=negative,
    )


class SanaSCMLADDDataLoader:
    """Main-process model preparation over a checkpointable cache loader."""

    def __init__(
        self,
        dataloader: object,
        *,
        adapter: SanaTrainAdapter,
        unconditional: SharedConditioningSample,
    ) -> None:
        if not callable(getattr(dataloader, "__iter__", None)):
            raise TypeError("dataloader must be iterable")
        if not callable(getattr(dataloader, "state_dict", None)) or not callable(
            getattr(dataloader, "load_state_dict", None)
        ):
            raise TypeError("SANA SCM-LADD dataloader must be checkpointable")
        self.dataloader = dataloader
        self.adapter = adapter
        self.unconditional = unconditional

    def __iter__(self) -> Iterator[SCMLADDTrainingBatch]:
        for batch in self.dataloader:
            if not isinstance(batch, TrainingBatch):
                raise TypeError("SANA cache loader must emit TrainingBatch values")
            yield prepare_sana_scm_ladd_batch(self.adapter, batch, self.unconditional)

    def state_dict(self) -> Mapping[str, object]:
        state = self.dataloader.state_dict()
        if not isinstance(state, Mapping):
            raise TypeError("SANA cache dataloader state_dict must return a mapping")
        return state

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.dataloader.load_state_dict(state_dict)


__all__ = [
    "SanaSCMLADDDataLoader",
    "audit_sana_scm_ladd_unconditional",
    "prepare_sana_scm_ladd_batch",
]
