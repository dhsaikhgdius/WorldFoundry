"""Behavioral text-conditioning dropout from the released AnyFlow trainers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import AnyFlowTrainingBatch

_TEXT_FIELDS = ("context", "encoder_hidden_states")


@dataclass(frozen=True, slots=True)
class AnyFlowConditioningDropout:
    batch: AnyFlowTrainingBatch
    mask: Tensor


def apply_conditioning_dropout(
    batch: AnyFlowTrainingBatch,
    probability: float,
    *,
    generator: torch.Generator | None,
) -> AnyFlowConditioningDropout:
    """Replace selected positive text embeddings with the negative branch."""

    if not isinstance(batch, AnyFlowTrainingBatch):
        raise TypeError("batch must be AnyFlowTrainingBatch")
    chance = float(probability)
    if not 0 <= chance <= 1:
        raise ValueError("AnyFlow conditioning dropout probability must lie in [0,1]")
    clean = batch.clean_latents
    if not isinstance(clean, Tensor):
        raise TypeError("AnyFlow clean_latents must be torch.Tensor")
    if chance == 0:
        return AnyFlowConditioningDropout(
            batch=batch,
            mask=torch.zeros(
                (batch.batch_size,),
                device=clean.device,
                dtype=torch.bool,
            ),
        )

    mask = (
        torch.rand(
            (batch.batch_size,),
            device=clean.device,
            dtype=torch.float32,
            generator=generator,
        )
        < chance
    )
    positive: dict[str, object] = dict(batch.conditioning)
    replaced = False
    for name in _TEXT_FIELDS:
        if name not in positive:
            continue
        conditional = positive[name]
        unconditional = batch.unconditional_conditioning.get(name)
        if not isinstance(conditional, Tensor) or not isinstance(
            unconditional,
            Tensor,
        ):
            raise TypeError(f"AnyFlow {name} dropout requires tensor positive/negative embeddings")
        if conditional.ndim == 0 or int(conditional.shape[0]) != batch.batch_size:
            raise ValueError(f"AnyFlow positive {name} must have the training batch dimension")
        if unconditional.ndim != conditional.ndim or unconditional.shape[1:] != (conditional.shape[1:]):
            raise ValueError(f"AnyFlow positive/negative {name} embeddings must have equal shapes")
        if int(unconditional.shape[0]) == 1:
            unconditional = unconditional.expand_as(conditional)
        elif int(unconditional.shape[0]) != batch.batch_size:
            raise ValueError(f"AnyFlow negative {name} must have batch size one or the local batch")
        unconditional = unconditional.to(
            device=conditional.device,
            dtype=conditional.dtype,
        )
        selection = mask.to(device=conditional.device).reshape(
            batch.batch_size,
            *([1] * (conditional.ndim - 1)),
        )
        positive[name] = torch.where(selection, unconditional, conditional)
        replaced = True
    if not replaced:
        raise ValueError("AnyFlow conditioning dropout requires context or encoder_hidden_states")
    return AnyFlowConditioningDropout(
        batch=AnyFlowTrainingBatch(
            sample_ids=batch.sample_ids,
            clean_latents=batch.clean_latents,
            conditioning=positive,
            unconditional_conditioning=batch.unconditional_conditioning,
        ),
        mask=mask,
    )


__all__ = ["AnyFlowConditioningDropout", "apply_conditioning_dropout"]
