"""Exact-resume homogeneous-domain batch cycling for DiffusionOPD."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence

from .contracts import DiffusionOPDRolloutBatch

DIFFUSION_OPD_DATA_LOADER_STATE_SCHEMA = "worldfoundry-diffusion-opd-data-loader"


class NativeDiffusionOPDDataLoader(Iterator[DiffusionOPDRolloutBatch]):
    """Cycle already materialized single-domain batches without mixing teachers."""

    def __init__(
        self,
        batches: Sequence[DiffusionOPDRolloutBatch],
        *,
        shuffle: bool = False,
        shuffle_seed: int = 42,
    ) -> None:
        values = tuple(batches)
        if not values or not all(isinstance(value, DiffusionOPDRolloutBatch) for value in values):
            raise ValueError("DiffusionOPD data loader requires typed rollout batches")
        self.batches = values
        self.shuffle = bool(shuffle)
        self.shuffle_seed = int(shuffle_seed)
        self.epoch = 0
        self.cursor = 0
        self.completed_batches = 0
        self.order = self._order_for_epoch(0)

    def _order_for_epoch(self, epoch: int) -> list[int]:
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.Random(self.shuffle_seed + epoch).shuffle(order)
        return order

    def __iter__(self) -> NativeDiffusionOPDDataLoader:
        return self

    def __next__(self) -> DiffusionOPDRolloutBatch:
        if self.cursor == len(self.order):
            self.epoch += 1
            self.cursor = 0
            self.order = self._order_for_epoch(self.epoch)
        value = self.batches[self.order[self.cursor]]
        self.cursor += 1
        self.completed_batches += 1
        return value

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": DIFFUSION_OPD_DATA_LOADER_STATE_SCHEMA,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "completed_batches": self.completed_batches,
            "order": list(self.order),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        expected = {"schema", "epoch", "cursor", "completed_batches", "order"}
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected:
            raise ValueError("DiffusionOPD data-loader state fields differ")
        if state_dict["schema"] != DIFFUSION_OPD_DATA_LOADER_STATE_SCHEMA:
            raise ValueError("unsupported DiffusionOPD data-loader state schema")
        epoch = int(state_dict["epoch"])
        cursor = int(state_dict["cursor"])
        completed = int(state_dict["completed_batches"])
        order = [int(index) for index in state_dict["order"]]  # type: ignore[union-attr]
        if (
            epoch < 0
            or completed < 0
            or not 0 <= cursor <= len(self.batches)
            or sorted(order) != list(range(len(self.batches)))
        ):
            raise ValueError("saved DiffusionOPD data-loader position is invalid")
        self.epoch = epoch
        self.cursor = cursor
        self.completed_batches = completed
        self.order = order


__all__ = [
    "DIFFUSION_OPD_DATA_LOADER_STATE_SCHEMA",
    "NativeDiffusionOPDDataLoader",
]
