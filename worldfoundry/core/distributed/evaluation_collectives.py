"""Small process-group helpers shared by evaluation runtimes."""

from __future__ import annotations

import math
import os
import pickle
from typing import Any

import torch
from torch import distributed as dist

from .generic_collectives import get_collective_device, get_rank, get_world_size


def print0(*args, **kwargs) -> None:
    if get_rank() == 0:
        print(*args, **kwargs)


def dist_init() -> None:
    """Initialize a default process group from torchrun-style environment variables."""

    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl" if use_cuda else "gloo", init_method="env://")


def all_gather(value: Any) -> list[Any]:
    """Gather tensors or arbitrary picklable values without assuming CUDA 0."""

    world_size = get_world_size()
    if world_size == 1:
        return [value]
    device = get_collective_device()

    original_shape: torch.Size | None = None
    if isinstance(value, torch.Tensor):
        original_shape = value.shape
        tensor = value.detach().reshape(-1).contiguous().to(device)
    else:
        buffer = bytearray(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        tensor = torch.frombuffer(buffer, dtype=torch.uint8).to(device)

    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=device)
    gathered_sizes = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    max_size = max(sizes)

    outputs = [torch.empty(max_size, dtype=tensor.dtype, device=device) for _ in sizes]
    if tensor.numel() < max_size:
        tensor = torch.cat(
            (tensor, torch.empty(max_size - tensor.numel(), dtype=tensor.dtype, device=device)),
            dim=0,
        )
    dist.all_gather(outputs, tensor)

    if original_shape is None:
        return [
            pickle.loads(output.cpu().numpy().tobytes()[:size])
            for size, output in zip(sizes, outputs)
        ]

    trailing_shape = tuple(original_shape[1:])
    trailing_size = math.prod(trailing_shape)
    if trailing_size == 0:
        if any(size != 0 for size in sizes):
            raise ValueError("cannot reconstruct gathered tensors with a zero-sized trailing dimension")
        return [output[:0].reshape((0, *trailing_shape)) for output in outputs]
    if not trailing_shape:
        return [output[:size].reshape(()) if size == 1 else output[:size] for size, output in zip(sizes, outputs)]
    if any(size % trailing_size for size in sizes):
        raise ValueError("gathered tensors do not share compatible trailing dimensions")
    return [
        output[:size].reshape((size // trailing_size, *trailing_shape))
        for size, output in zip(sizes, outputs)
    ]


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def merge_list_of_list(results):
    return [item for sublist in results for item in sublist]


def gather_list_of_dict(results):
    return merge_list_of_list(all_gather(results))


def distribute_list_to_rank(data_list):
    return data_list[get_rank() :: get_world_size()]


__all__ = [
    "all_gather",
    "barrier",
    "dist_init",
    "distribute_list_to_rank",
    "gather_list_of_dict",
    "get_rank",
    "get_world_size",
    "merge_list_of_list",
    "print0",
]
