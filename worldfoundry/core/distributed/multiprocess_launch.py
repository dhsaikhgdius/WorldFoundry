"""Multiprocess launch helpers for torch distributed jobs."""

from __future__ import annotations

import os
import socket

import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from worldfoundry.core.distributed import object_collectives


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def launch(fn, n_gpu_per_machine, n_machine=1, machine_rank=0, dist_url=None, args=()):
    world_size = n_machine * n_gpu_per_machine
    if world_size <= 1:
        fn(*args)
        return

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"

    if dist_url == "auto":
        if n_machine != 1:
            raise ValueError('dist_url="auto" is only supported for single-machine jobs')
        dist_url = f"tcp://127.0.0.1:{find_free_port()}"

    if n_machine > 1 and dist_url and dist_url.startswith("file://"):
        raise ValueError("file:// is not reliable for multi-machine jobs; use tcp://")

    mp.spawn(
        distributed_worker,
        nprocs=n_gpu_per_machine,
        args=(fn, world_size, n_gpu_per_machine, machine_rank, dist_url, args),
        daemon=False,
    )


def distributed_worker(local_rank, fn, world_size, n_gpu_per_machine, machine_rank, dist_url, args):
    if not torch.cuda.is_available():
        raise OSError("CUDA is not available.")

    global_rank = machine_rank * n_gpu_per_machine + local_rank

    if n_gpu_per_machine > torch.cuda.device_count():
        raise ValueError(
            f"specified n_gpu_per_machine is larger than available devices ({torch.cuda.device_count()})"
        )
    # Bind the CUDA device before any NCCL collective: a barrier issued while
    # every local process still points at cuda:0 makes NCCL build multiple
    # communicators on one GPU ("Duplicate GPU detected" or a hang, CC-20).
    torch.cuda.set_device(local_rank)

    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )
    except Exception as exc:
        raise OSError("failed to initialize NCCL groups") from exc

    object_collectives.synchronize()
    if object_collectives.LOCAL_PROCESS_GROUP is not None:
        raise ValueError("LOCAL_PROCESS_GROUP is already initialized")

    n_machine = world_size // n_gpu_per_machine
    for index in range(n_machine):
        ranks = list(range(index * n_gpu_per_machine, (index + 1) * n_gpu_per_machine))
        process_group = dist.new_group(ranks)
        if index == machine_rank:
            object_collectives.LOCAL_PROCESS_GROUP = process_group

    fn(*args)


__all__ = ["distributed_worker", "find_free_port", "launch"]
