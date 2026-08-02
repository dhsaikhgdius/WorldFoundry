"""Object and tensor collectives for lightweight distributed jobs."""

from __future__ import annotations

import torch
from torch import distributed as dist
from torch.utils import data

from .generic_collectives import get_rank, get_world_size
from .generic_collectives import is_master as is_primary

LOCAL_PROCESS_GROUP = None


def get_local_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    if LOCAL_PROCESS_GROUP is None:
        raise ValueError("LOCAL_PROCESS_GROUP is None")
    return dist.get_rank(group=LOCAL_PROCESS_GROUP)


def synchronize() -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if dist.get_world_size() == 1:
        return
    dist.barrier()


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    if get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(value):
    """Gather picklable Python objects from every rank."""

    from .evaluation_collectives import all_gather as gather

    return gather(value)


def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[key] for key in keys], 0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            values /= world_size
    return {key: value for key, value in zip(keys, values)}


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    return data.SequentialSampler(dataset)


__all__ = [
    "LOCAL_PROCESS_GROUP",
    "all_gather",
    "all_reduce",
    "data_sampler",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "is_primary",
    "reduce_dict",
    "synchronize",
]
