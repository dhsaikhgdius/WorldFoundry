# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os

import torch
import torch.distributed as dist

from .generic_collectives import get_rank, get_world_size  # noqa: F401 - public compatibility exports
from .tensor_collectives import all_gather_concat, all_to_all_concat


def _distributed_ready():
    return dist.is_available() and dist.is_initialized()


def init_distributed_group():
    """r initialize sequence parallel group."""
    if not _distributed_ready():
        dist.init_process_group(backend="nccl")


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = dist.get_world_size(group) if _distributed_ready() else get_world_size()
    if world_size > 1:
        if not kwargs:
            return all_to_all_concat(
                x,
                scatter_dim=scatter_dim,
                gather_dim=gather_dim,
                group=group,
            )
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_to_all_many(tensors, scatter_dim, gather_dim, group=None, **kwargs):
    """Fuse equal-shaped tensors into one all-to-all when memory-safe.

    Q/K/V exchanges otherwise pay three Python, dispatcher, and NCCL launch
    costs per attention layer. Stacking introduces temporary storage, so calls
    larger than ``WORLDFOUNDRY_FUSED_QKV_A2A_MAX_MB`` fall back automatically.
    """

    values = tuple(tensors)
    if len(values) < 2 or not _distributed_ready():
        return values
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return values
    first = values[0]
    compatible = all(
        value.shape == first.shape
        and value.dtype == first.dtype
        and value.device == first.device
        for value in values[1:]
    )
    try:
        max_bytes = max(
            int(float(os.getenv("WORLDFOUNDRY_FUSED_QKV_A2A_MAX_MB", "512") or "512") * 1024**2),
            0,
        )
    except ValueError:
        max_bytes = 512 * 1024**2
    total_bytes = sum(value.numel() * value.element_size() for value in values)
    if not compatible or max_bytes == 0 or total_bytes > max_bytes:
        return tuple(all_to_all(value, scatter_dim, gather_dim, group=group, **kwargs) for value in values)

    packed = torch.stack(values, dim=0)
    exchanged = all_to_all(
        packed,
        scatter_dim=scatter_dim + 1,
        gather_dim=gather_dim + 1,
        group=group,
        **kwargs,
    )
    return exchanged.unbind(dim=0)


def all_gather(tensor):
    world_size = get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def gather_forward(input, dim):
    # skip if world_size == 1
    world_size = get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    return all_gather_concat(input, dim=dim)
