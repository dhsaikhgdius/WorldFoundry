"""Checkpointed all-rank decisions for AnyFlow schedules and FAR horizons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch import Tensor

ANYFLOW_DECISION_RNG_SCHEMA = "worldfoundry-anyflow-decision-rng"


@runtime_checkable
class AnyFlowTensorSynchronizer(Protocol):
    def synchronize_tensor(self, value: Tensor) -> Tensor: ...


class ProcessGroupAnyFlowTensorSynchronizer:
    """Broadcast decisions from the first rank of an initialized group."""

    def __init__(self, process_group: object | None = None) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("AnyFlow process-group synchronization requires initialized distributed")
        if process_group is None:
            ranks = tuple(range(dist.get_world_size()))
        else:
            ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(process_group))
        if not ranks:
            raise ValueError("AnyFlow synchronization process group is empty")
        self.process_group = process_group
        self.source_rank = min(ranks)
        self.world_size = len(ranks)

    def synchronize_tensor(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor) or value.numel() == 0:
            raise TypeError("AnyFlow synchronization requires a non-empty tensor")
        if self.world_size > 1:
            dist.broadcast(value, src=self.source_rank, group=self.process_group)
        return value


class AnyFlowDecisionRNG:
    """Use identical CPU RNG advancement on every rank, then broadcast results."""

    def __init__(
        self,
        seed: int,
        *,
        synchronizer: AnyFlowTensorSynchronizer | None = None,
    ) -> None:
        if isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("AnyFlow decision seed must be a non-negative integer")
        if synchronizer is not None and not isinstance(synchronizer, AnyFlowTensorSynchronizer):
            raise TypeError("synchronizer must implement AnyFlowTensorSynchronizer")
        self.seed = int(seed)
        self.synchronizer = synchronizer
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        self.draw_count = 0

    def _synchronize(self, value: Tensor) -> Tensor:
        if self.synchronizer is None:
            return value
        result = self.synchronizer.synchronize_tensor(value)
        if not isinstance(result, Tensor):
            raise TypeError("AnyFlow synchronizer must return a tensor")
        if result.shape != value.shape or result.dtype != value.dtype or result.device != value.device:
            raise ValueError("AnyFlow synchronization must preserve tensor shape, dtype, and device")
        return result

    def randrange(self, stop: int, *, reference: Tensor) -> int:
        if isinstance(stop, bool) or int(stop) <= 0:
            raise ValueError("AnyFlow randrange stop must be positive")
        local = torch.randint(0, int(stop), (1,), generator=self.generator, device="cpu")
        self.draw_count += 1
        synchronized = self._synchronize(local.to(device=reference.device))
        result = int(synchronized.item())
        if not 0 <= result < int(stop):
            raise RuntimeError("synchronized AnyFlow index is out of range")
        return result

    def choice(self, values: Sequence[int], *, reference: Tensor) -> int:
        options = tuple(int(value) for value in values)
        if not options:
            raise ValueError("AnyFlow choice requires non-empty values")
        return options[self.randrange(len(options), reference=reference)]

    def bernoulli(self, probability: float, *, reference: Tensor) -> bool:
        chance = float(probability)
        if not 0 <= chance <= 1:
            raise ValueError("AnyFlow Bernoulli probability must lie in [0,1]")
        local = torch.rand((1,), generator=self.generator, device="cpu")
        self.draw_count += 1
        decision = (local < chance).to(device=reference.device, dtype=torch.int64)
        synchronized = self._synchronize(decision)
        return bool(int(synchronized.item()))

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": ANYFLOW_DECISION_RNG_SCHEMA,
            "seed": self.seed,
            "draw_count": self.draw_count,
            "rng_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("AnyFlow decision RNG state must be a mapping")
        expected = {"schema", "seed", "draw_count", "rng_state"}
        if set(state_dict) != expected:
            raise ValueError("AnyFlow decision RNG state fields differ from the active schema")
        if state_dict["schema"] != ANYFLOW_DECISION_RNG_SCHEMA:
            raise ValueError(f"unsupported AnyFlow decision RNG schema: {state_dict['schema']!r}")
        if int(state_dict["seed"]) != self.seed:
            raise ValueError("saved AnyFlow decision seed differs from the active run")
        count = state_dict["draw_count"]
        state = state_dict["rng_state"]
        if isinstance(count, bool) or int(count) < 0:
            raise ValueError("saved AnyFlow decision draw count is invalid")
        if not isinstance(state, Tensor) or state.dtype is not torch.uint8 or state.ndim != 1:
            raise ValueError("saved AnyFlow decision RNG state is invalid")
        self.generator.set_state(state.detach().cpu())
        self.draw_count = int(count)


__all__ = [
    "ANYFLOW_DECISION_RNG_SCHEMA",
    "AnyFlowDecisionRNG",
    "AnyFlowTensorSynchronizer",
    "ProcessGroupAnyFlowTensorSynchronizer",
]
