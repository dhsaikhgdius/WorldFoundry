"""Runtime contracts for paired Diffusion-DPO training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from ....shared.contracts import freeze_mapping, non_empty_ids


@dataclass(frozen=True, slots=True)
class DiffusionDPOBatch:
    """A batch of adjacent ``[chosen, rejected]`` clean-latent pairs."""

    batch_id: str
    sample_ids: tuple[str, ...]
    pair_ids: tuple[str, ...]
    clean_latents: torch.Tensor
    conditioning: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str) or not self.batch_id.strip():
            raise ValueError("batch_id must be a non-empty string")
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        pair_ids = non_empty_ids(self.pair_ids, field_name="pair_ids", unique=False)
        batch_size = len(sample_ids)
        if batch_size % 2:
            raise ValueError("Diffusion-DPO batches must contain adjacent chosen/rejected pairs")
        if len(pair_ids) != batch_size:
            raise ValueError("pair_ids length must match sample_ids")
        chosen_pair_ids = pair_ids[0::2]
        rejected_pair_ids = pair_ids[1::2]
        if chosen_pair_ids != rejected_pair_ids:
            raise ValueError("each chosen sample must be immediately followed by its rejected pair")
        if len(set(chosen_pair_ids)) != len(chosen_pair_ids):
            raise ValueError("pair_ids must identify exactly one adjacent chosen/rejected pair")
        if (
            not isinstance(self.clean_latents, torch.Tensor)
            or not self.clean_latents.is_floating_point()
            or self.clean_latents.ndim < 2
            or int(self.clean_latents.shape[0]) != batch_size
        ):
            raise TypeError("clean_latents must be a floating [B,...] torch.Tensor")
        if not bool(torch.isfinite(self.clean_latents).all()):
            raise ValueError("clean_latents must be finite")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "pair_ids", pair_ids)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    @property
    def pair_count(self) -> int:
        return self.batch_size // 2


__all__ = ["DiffusionDPOBatch"]
