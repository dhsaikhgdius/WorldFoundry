"""Torch process-group helpers shared by runtime and training code."""

from __future__ import annotations

import collections
import collections.abc
import ctypes
import functools
import logging
import math
import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Callable, Container, Optional

import torch
import torch.distributed as dist
from torch.distributed import get_process_group_ranks

try:
    import pynvml
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency.
    pynvml = None

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group
    from torch.distributed.utils import (
        _sync_module_states,
        _verify_param_shape_across_processes,
    )

try:
    from worldfoundry.core.distributed.megatron_compat import parallel_state
except Exception:  # pragma: no cover - optional training dependency.

    class _NoParallelState:
        @staticmethod
        def is_initialized() -> bool:
            return False

        @staticmethod
        def get_data_parallel_group(with_context_parallel: bool = False):
            return None

    parallel_state = _NoParallelState()


logger = logging.getLogger(__name__)


def _get_local_cuda_device(local_rank: int) -> int:
    override = os.getenv("WORLDFOUNDRY_GEN3C_LOCAL_CUDA_DEVICE")
    if override is None:
        return local_rank
    try:
        return int(override)
    except ValueError:
        logger.warning(
            "Ignoring invalid WORLDFOUNDRY_GEN3C_LOCAL_CUDA_DEVICE=%r.",
            override,
        )
        return local_rank


def _get_gpu_cpu_affinity(device_idx: int) -> list[int]:
    if pynvml is None:
        return []
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    affinity_elements = math.ceil((os.cpu_count() or 1) / 64)
    affinity_string = ""
    for element in pynvml.nvmlDeviceGetCpuAffinity(handle, affinity_elements):
        affinity_string = f"{element:064b}" + affinity_string
    affinity = [int(value) for value in affinity_string]
    affinity.reverse()
    return [index for index, enabled in enumerate(affinity) if enabled]


def _set_cuda_l2_fetch_granularity() -> None:
    if not torch.cuda.is_available():
        return
    try:
        libcudart = ctypes.CDLL("libcudart.so")
        value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        libcudart.cudaDeviceGetLimit(value, ctypes.c_int(0x05))
    except OSError:
        logger.debug("libcudart.so is unavailable; skipped CUDA device limit setup.")


def init() -> int | None:
    """Initialize a NCCL process group from torchrun-style environment variables."""

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    local_cuda_device = _get_local_cuda_device(local_rank)
    if dist.is_available() and dist.is_initialized():
        return torch.cuda.current_device() if torch.cuda.is_available() else local_rank

    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            affinity = _get_gpu_cpu_affinity(local_cuda_device)
            if affinity:
                os.sched_setaffinity(0, affinity)
        except Exception as exc:
            logger.warning("Failed to set GPU CPU affinity: %s", exc)

    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if dist.is_available():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_cuda_device)
        timeout_seconds = int(os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
        logger.info(
            "Initialized distributed process group with local rank %s, CUDA device %s, and timeout %s.",
            local_rank,
            local_cuda_device,
            timeout_seconds,
        )

    _set_cuda_l2_fetch_granularity()
    logger.info("Running with %s GPUs.", get_world_size())
    return None


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Return this worker's rank, or 0 outside distributed execution."""

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group)
    return 0


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """Return the process-group size, or 1 outside distributed execution."""

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group)
    return 1


def is_rank0() -> bool:
    return get_rank() == 0


def is_local_rank0() -> bool:
    if torch.cuda.is_available():
        return torch.cuda.current_device() == 0
    return int(os.getenv("LOCAL_RANK", 0)) == 0


def device_with_rank(device: str) -> str:
    if device == "cuda":
        return f"cuda:{get_rank()}"
    return device


def rank0_only(func: Callable) -> Callable:
    """Run ``func`` only on global rank 0."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        if is_rank0():
            return func(*args, **kwargs)
        return None

    return wrapper


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def rank0_first(func: Callable) -> Callable:
    """Run ``func`` on rank 0 before all other ranks."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        result = None
        if is_rank0():
            result = func(*args, **kwargs)
        barrier()
        if not is_rank0():
            result = func(*args, **kwargs)
        return result

    return wrapper


def parallel_model_wrapper(config_ddp: Any, model: torch.nn.Module) -> torch.nn.Module | DistributedDataParallel:
    """Wrap a model with DDP when a process group is initialized."""

    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        local_cuda_device = _get_local_cuda_device(local_rank)
        try:
            ddp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        except Exception as exc:
            logger.info("parallel_state not initialized; using the default DDP group: %s", exc)
            ddp_group = None

        model = DistributedDataParallel(
            model,
            device_ids=[local_cuda_device],
            output_device=local_cuda_device,
            find_unused_parameters=config_ddp.find_unused_parameters,
            static_graph=config_ddp.static_graph,
            broadcast_buffers=config_ddp.broadcast_buffers,
            process_group=ddp_group,
        )
    return model


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """DDP wrapper that redirects ``training_step`` through ``forward``."""

    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.show_sync_grad_static_graph_warning = True

    def training_step(self, *args, **kwargs) -> Any:
        original_forward = self.module.forward

        def wrapped_training_step(*_args, **_kwargs):  # noqa: ANN202
            self.module.forward = original_forward
            return self.module.training_step(*_args, **_kwargs)

        self.module.forward = wrapped_training_step
        return self(*args, **kwargs)


@contextmanager
def ddp_sync_grad(model, enabled):
    """Temporarily enable or disable DDP gradient synchronization."""

    assert isinstance(model, torch.nn.Module)
    old_require_backward_grad_sync = None
    if isinstance(model, DistributedDataParallel):
        old_require_backward_grad_sync = model.require_backward_grad_sync
        if model.static_graph and model.require_backward_grad_sync != enabled:
            if model.show_sync_grad_static_graph_warning:
                logger.warning("DDP static_graph=True is incompatible with ddp_sync_grad().")
                model.show_sync_grad_static_graph_warning = False
        else:
            model.require_backward_grad_sync = enabled
    try:
        yield
    finally:
        if isinstance(model, DistributedDataParallel) and old_require_backward_grad_sync is not None:
            model.require_backward_grad_sync = old_require_backward_grad_sync


def collate_batches(data_batches: list[dict[str, torch.Tensor]]) -> torch.Tensor | dict[str, torch.Tensor]:
    """Gather validation batches from all ranks in original rank order."""

    if isinstance(data_batches[0], torch.Tensor):
        data_concat = torch.cat(data_batches, dim=0)  # type: ignore[arg-type]
        if get_world_size() == 1:
            return data_concat
        max_num_local_samples = torch.tensor(len(data_concat), device=data_concat.device)
        dist.all_reduce(max_num_local_samples, op=dist.ReduceOp.MAX)
        max_num_local_samples_value = int(max_num_local_samples.item())
        if len(data_concat) < max_num_local_samples_value:
            assert len(data_concat) + 1 == max_num_local_samples_value
            dummy = torch.empty_like(data_concat[:1])
            data_concat = torch.cat([data_concat, dummy], dim=0)
            dummy_count = torch.tensor(1, device=data_concat.device)
        else:
            dummy_count = torch.tensor(0, device=data_concat.device)
        dist.all_reduce(dummy_count, op=dist.ReduceOp.SUM)
        dummy_count_value = int(dummy_count.item())
        gathered = all_gather_tensor(data_concat.contiguous())
        data_collate = torch.stack(gathered, dim=1).flatten(start_dim=0, end_dim=1)
        if dummy_count_value > 0:
            data_collate = data_collate[:-dummy_count_value]
    elif isinstance(data_batches[0], collections.abc.Mapping):
        data_collate = {}
        for key in data_batches[0].keys():
            data_collate[key] = collate_batches([data[key] for data in data_batches])  # type: ignore[index]
    else:
        raise TypeError(f"Unsupported batch type: {type(data_batches[0])!r}")
    return data_collate


@torch.no_grad()
def all_gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    if get_world_size() == 1:
        return [tensor]
    tensor_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def broadcast(tensor, src, group=None, async_op=False):
    if get_world_size() < 2:
        return tensor
    dist.broadcast(tensor, src=src, group=group, async_op=async_op)
    return tensor


def dist_reduce_tensor(tensor, rank=0, reduce="mean"):
    if get_world_size() < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=rank)
        if get_rank() == rank:
            if reduce == "mean":
                tensor /= get_world_size()
            elif reduce != "sum":
                raise NotImplementedError(f"Unsupported reduce mode: {reduce}")
    return tensor


def sync_model_states(
    model: torch.nn.Module,
    process_group: Optional[dist.ProcessGroup] = None,
    src: int = 0,
    params_and_buffers_to_ignore: Optional[Container[str]] = None,
    broadcast_buffers: bool = True,
) -> None:
    """Broadcast model parameters and buffers from ``src`` to the process group."""

    if not dist.is_available() or not dist.is_initialized():
        return
    if process_group is None:
        process_group = _get_default_group()
    if not params_and_buffers_to_ignore:
        params_and_buffers_to_ignore = set()

    logger.info(
        "Synchronizing model states from rank %s to process group %s.",
        src,
        get_process_group_ranks(process_group),
    )

    modules_and_parameters = [
        (module, parameter)
        for module_name, module in model.named_modules()
        for parameter in [
            param
            for param_name, param in module.named_parameters(recurse=False)
            if f"{module_name}.{param_name}" not in params_and_buffers_to_ignore
        ]
    ]

    memo = set()
    modules_and_parameters = [
        (module, parameter)
        for module, parameter in modules_and_parameters
        if parameter not in memo and not memo.add(parameter)  # type: ignore[func-returns-value]
    ]
    parameters = [parameter for _, parameter in modules_and_parameters]
    if not parameters:
        return

    _verify_param_shape_across_processes(process_group, parameters)
    _sync_module_states(
        module=model,
        process_group=process_group,
        broadcast_bucket_size=int(250 * 1024 * 1024),
        src=src,
        params_and_buffers_to_ignore=params_and_buffers_to_ignore,
        broadcast_buffers=broadcast_buffers,
    )


__all__ = [
    "DistributedDataParallel",
    "all_gather_tensor",
    "barrier",
    "broadcast",
    "collate_batches",
    "ddp_sync_grad",
    "device_with_rank",
    "dist_reduce_tensor",
    "get_rank",
    "get_world_size",
    "init",
    "is_local_rank0",
    "is_rank0",
    "parallel_model_wrapper",
    "rank0_first",
    "rank0_only",
    "sync_model_states",
]
