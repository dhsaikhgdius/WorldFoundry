# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import gc
import logging
import os
import shutil

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

#: Environment variables that indicate this process was launched by a
#: distributed launcher (torchrun/torch.distributed.launch or a scheduler
#: that pre-populates the rendezvous contract). ``MASTER_PORT`` alone is
#: deliberately excluded: several tools set a default port preemptively
#: without implying a multi-process launch.
_DISTRIBUTED_LAUNCH_ENV_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "TORCHELASTIC_RUN_ID",
)


def _distributed_launch_indicators() -> dict[str, str]:
    """Return the subset of distributed-launcher env vars present in this process."""

    return {name: os.environ[name] for name in _DISTRIBUTED_LAUNCH_ENV_VARS if name in os.environ}


def get_collective_device(group=None) -> torch.device:
    """Return the device required by the active process-group backend.

    NCCL collectives must use the CUDA device selected for this process. CPU
    backends such as Gloo use CPU tensors. Keeping this decision in one place
    prevents bare ``cuda`` allocations from silently landing on GPU 0.
    """

    if dist.is_available() and dist.is_initialized():
        backend = str(dist.get_backend(group)).lower()
        if "nccl" not in backend:
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def is_distributed():
    return get_world_size() > 1


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    return local_rank


def is_master():
    return get_rank() == 0


def is_local_master():
    return get_local_rank() == 0


def get_local_proc_group(group_size=8):
    world_size = get_world_size()
    if world_size <= group_size or group_size == 1:
        return None
    assert world_size % group_size == 0, (
        f"world size ({world_size}) should be evenly divided by group size ({group_size})."
    )
    process_groups = getattr(get_local_proc_group, "process_groups", dict())
    if group_size not in process_groups:
        num_groups = dist.get_world_size() // group_size
        groups = [list(range(i * group_size, (i + 1) * group_size)) for i in range(num_groups)]
        process_groups.update({group_size: [torch.distributed.new_group(group) for group in groups]})
        get_local_proc_group.process_groups = process_groups

    group_idx = get_rank() // group_size
    process_groups = get_local_proc_group.process_groups.get(group_size)[group_idx]
    return process_groups


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def dist_init() -> None:
    """Initialize the default NCCL process group, or fall back to single-process mode.

    Failure semantics:
    - If the environment carries distributed-launcher indicators (``RANK``,
      ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR``, ``TORCHELASTIC_RUN_ID``),
      an initialization failure re-raises. Masquerading a failed rank as a
      standalone ``RANK=0/WORLD_SIZE=1`` process would skip every collective,
      write rank-0 output paths, and leave the surviving ranks hanging at their
      first collective until the NCCL timeout — a data-corruption path.
    - If no such indicator is present (plain ``python script.py``), the
      single-process fallback is legitimate and is kept.
    """

    if is_dist_initialized():
        return
    try:
        torch.distributed.init_process_group(backend="nccl")
        if not torch.distributed.is_initialized():
            raise RuntimeError("torch.distributed.init_process_group returned but the process group is not initialized")
    except Exception as exc:
        indicators = _distributed_launch_indicators()
        if indicators:
            rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(indicators.items()))
            logger.error(
                "Distributed process-group initialization failed in a distributed launch environment (%s). "
                "Refusing the single-process fallback; failing fast instead: %s",
                rendered,
                exc,
            )
            raise
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        logger.warning(
            "No distributed launch environment detected; continuing as a single process "
            "(process-group init failed with: %s)",
            exc,
        )


def is_dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_dist_rank() -> int:
    return get_rank()


def get_dist_size() -> int:
    return get_world_size()


def get_dist_local_rank() -> int:
    return get_local_rank()


def dist_barrier() -> None:
    synchronize()


def sync_tensor(tensor, reduce="mean"):
    if not is_dist_initialized():
        return tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor([tensor], device=get_collective_device())
    tensor_list = [torch.empty_like(tensor) for _ in range(get_world_size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return sum(tensor_list) / len(tensor_list)
    if reduce == "sum":
        return sum(tensor_list)
    if reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    if reduce == "root":
        return tensor_list[0]
    return tensor_list


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    from .evaluation_collectives import all_gather as gather

    return gather(data)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def broadcast(data, **kwargs):
    if get_world_size() == 1:
        return data
    data = [data]
    dist.broadcast_object_list(data, **kwargs)
    return data[0]


def all_gather_cpu(result_part, tmpdir=None, collect_by_master=True):
    import mmcv
    from mmcv.runner import get_dist_info

    rank, world_size = get_dist_info()
    if tmpdir is None:
        tmpdir = "./tmp"
    if rank == 0:
        mmcv.mkdir_or_exist(tmpdir)
    synchronize()
    # dump the part result to the dir
    mmcv.dump(result_part, os.path.join(tmpdir, f"part_{rank}.pkl"))
    synchronize()
    # collect all parts
    if collect_by_master and rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        results = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f"part_{i}.pkl")
            results.append(mmcv.load(part_file))
    if not collect_by_master:
        synchronize()
    # remove tmp dir
    if rank == 0:
        shutil.rmtree(tmpdir)
    return results


def all_gather_tensor(tensor, group_size=None, group=None):
    if group_size is None:
        group_size = get_world_size()
    if group_size == 1:
        output = [tensor]
    else:
        output = [torch.zeros_like(tensor) for _ in range(group_size)]
        dist.all_gather(output, tensor, group=group)
    return output


def gather_difflen_tensor(feat, num_samples_list, concat=True, group=None, group_size=None):
    world_size = get_world_size()
    if world_size == 1:
        if not concat:
            return [feat]
        return feat
    num_samples, *feat_dim = feat.size()
    # padding to max number of samples
    feat_padding = feat.new_zeros((max(num_samples_list), *feat_dim))
    feat_padding[:num_samples] = feat
    # gather
    feat_gather = all_gather_tensor(feat_padding, group=group, group_size=group_size)
    for r, num in enumerate(num_samples_list):
        feat_gather[r] = feat_gather[r][:num]
    if concat:
        feat_gather = torch.cat(feat_gather)
    return feat_gather


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        num_samples = torch.tensor(input.size(0), dtype=torch.long, device=input.device)
        ctx.num_samples_list = all_gather_tensor(num_samples)
        output = gather_difflen_tensor(input, ctx.num_samples_list, concat=False)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):  # tuple(output)'s grad
        (input,) = ctx.saved_tensors
        num_samples_list = ctx.num_samples_list
        rank = get_rank()
        start, end = sum(num_samples_list[:rank]), sum(num_samples_list[: rank + 1])
        grads = torch.cat(grads)
        if is_distributed():
            dist.all_reduce(grads)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[start:end]
        return grad_out, None, None


class GatherLayerWithGroup(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input, group, group_size):
        ctx.save_for_backward(input)
        ctx.group_size = group_size
        output = all_gather_tensor(input, group=group, group_size=group_size)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):  # tuple(output)'s grad
        (input,) = ctx.saved_tensors
        grads = torch.stack(grads)
        if is_distributed():
            dist.all_reduce(grads)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank() % ctx.group_size]
        return grad_out, None, None


def gather_layer_with_group(data, group=None, group_size=None):
    if group_size is None:
        group_size = get_world_size()
    output = GatherLayerWithGroup.apply(data, group, group_size)
    return output


def flush():
    gc.collect()
    torch.cuda.empty_cache()
