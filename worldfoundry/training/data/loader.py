"""Optional TorchData loader integration with exact local resume state."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Sized
from typing import Any

from .latent_token_sampler import LatentTokenBatchSampler
from .sampler import DeterministicDistributedSampler


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be non-negative; got {resolved}")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    resolved = _non_negative_int(value, field_name=field_name)
    if resolved == 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def build_stateful_dataloader(
    dataset: Sized,
    sampler: DeterministicDistributedSampler | None = None,
    *,
    batch_size: int | None = None,
    batch_sampler: LatentTokenBatchSampler | None = None,
    collate_fn: Callable[[list[Any]], Any] | None = None,
    num_workers: int = 0,
    worker_seed: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    multiprocessing_context: str | None = None,
    snapshot_every_n_steps: int = 1,
) -> Any:
    """Build a checkpointable map-style loader around the audited sampler.

    ``StatefulDataLoader.state_dict()`` is the checkpoint boundary.  Its state
    includes worker progress and the sampler seam, while cross-rank gathering
    remains the responsibility of the distributed checkpoint manager.
    """

    if not isinstance(dataset, Sized) or not callable(getattr(dataset, "__getitem__", None)):
        raise TypeError("dataset must be a sized map-style dataset")
    if (sampler is None) == (batch_sampler is None):
        raise ValueError("provide exactly one of sampler or batch_sampler")
    if sampler is not None:
        if not isinstance(sampler, DeterministicDistributedSampler):
            raise TypeError("sampler must be a DeterministicDistributedSampler")
        if sampler.data_source is not dataset:
            raise ValueError("sampler.data_source must be the exact dataset passed to the loader")
        if batch_size is None:
            raise ValueError("batch_size is required with sampler")
        resolved_batch_size = _positive_int(batch_size, field_name="batch_size")
    else:
        if not isinstance(batch_sampler, LatentTokenBatchSampler):
            raise TypeError("batch_sampler must be a LatentTokenBatchSampler")
        if batch_sampler.data_source is not dataset:
            raise ValueError("batch_sampler.data_source must be the exact dataset passed to the loader")
        if batch_size is not None:
            raise ValueError("batch_size must be omitted when batch_sampler is provided")
        if drop_last:
            raise ValueError("drop_last belongs to the token batch sampler tail policy")
        resolved_batch_size = None
    resolved_workers = _non_negative_int(num_workers, field_name="num_workers")
    resolved_seed = _non_negative_int(worker_seed, field_name="worker_seed")
    resolved_snapshot_interval = _positive_int(
        snapshot_every_n_steps,
        field_name="snapshot_every_n_steps",
    )
    for name, value in (
        ("pin_memory", pin_memory),
        ("drop_last", drop_last),
        ("persistent_workers", persistent_workers),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
    if persistent_workers and resolved_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    if prefetch_factor is not None:
        resolved_prefetch = _positive_int(prefetch_factor, field_name="prefetch_factor")
        if resolved_workers == 0:
            raise ValueError("prefetch_factor requires num_workers > 0")
    else:
        resolved_prefetch = None
    resolved_context = None
    if multiprocessing_context is not None:
        resolved_context = str(multiprocessing_context).strip().lower()
        if resolved_workers == 0:
            raise ValueError("multiprocessing_context requires num_workers > 0")
        supported_contexts = set(multiprocessing.get_all_start_methods())
        if resolved_context not in supported_contexts:
            raise ValueError(f"multiprocessing_context must be one of {sorted(supported_contexts)}")

    try:
        import torch
        from torchdata.stateful_dataloader import StatefulDataLoader
    except ModuleNotFoundError as error:
        raise RuntimeError("checkpointable worker loading requires the 'train-core' TorchData dependency") from error

    generator = torch.Generator()
    generator.manual_seed(resolved_seed)
    common = {
        "num_workers": resolved_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "generator": generator,
        "prefetch_factor": resolved_prefetch,
        "persistent_workers": persistent_workers,
        "multiprocessing_context": resolved_context,
        "in_order": True,
        "snapshot_every_n_steps": resolved_snapshot_interval,
    }
    if batch_sampler is not None:
        # TorchData follows PyTorch's custom-batch-sampler contract: the public
        # default batch_size must remain 1 and is normalized to None internally.
        return StatefulDataLoader(dataset, batch_sampler=batch_sampler, **common)
    assert sampler is not None and resolved_batch_size is not None
    return StatefulDataLoader(
        dataset,
        batch_size=resolved_batch_size,
        sampler=sampler,
        drop_last=drop_last,
        **common,
    )


__all__ = ["build_stateful_dataloader"]
