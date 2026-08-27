"""Sequence-parallel process-group state and collectives (Hunyuan-style).

This module holds one of the three parallel-topology singletons that
currently coexist under ``core/distributed`` (tracked as XC-10 in
``plan/code_review/12_cross_cutting.md``):

- :mod:`worldfoundry.core.distributed.context_parallel_util` — the
  data-parallel x context-parallel device-mesh state;
- this module — ``nccl_info`` + ``_SEQUENCE_PARALLEL_GROUPS``, consumed by
  the ``pipelines/hunyuan_world`` runtimes and the optional xFuser
  compatibility layer at the bottom of the file;
- :mod:`worldfoundry.core.distributed.sequence_parallel.parallel_state` —
  the vLLM-style ``GroupCoordinator`` singletons used by the training
  engine, with a full init/destroy lifecycle.

``sequence_parallel.parallel_state`` is the most complete of the three and
is the intended single source of truth; the state kept here is slated to
become a read-only view of it in a follow-up consolidation. Until then, do
not initialize sequence parallelism through more than one of these modules
in the same process — the singletons do not observe each other.
"""

import datetime
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn import functional as F

from worldfoundry.core.distributed.device_mesh_collectives import all_to_all_tensor

logger = logging.getLogger(__name__)


class SequenceParallelInfo:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.group = None
        self.sp_size = 1
        self.global_rank = 0
        self.rank_within_group = 0
        self.group_id = 0


nccl_info = SequenceParallelInfo()
_SEQUENCE_PARALLEL_STATE = False
_SEQUENCE_PARALLEL_GROUPS: dict[str, dist.ProcessGroup] = {}
_COLLECTIVE_SHAPE_CACHE: "OrderedDict[tuple[object, ...], tuple[tuple[int, ...], ...]]" = OrderedDict()
_COLLECTIVE_SHAPE_CACHE_LOCK = threading.Lock()
_COLLECTIVE_SHAPE_CACHE_LIMIT = 128


def _resolve_group(group: Optional[dist.ProcessGroup] = None):
    return group if group is not None else get_sequence_parallel_group()


def _rank_in_group(group: Optional[dist.ProcessGroup] = None):
    if group is None:
        return dist.get_rank()
    return dist.get_group_rank(group, dist.get_rank())


def _collective_shape_cache_enabled() -> bool:
    # Dynamic video resolutions can leave one rank's local shape unchanged
    # while another rank changes. Cache only when the deployment explicitly
    # promises a fixed-shape workload.
    return os.getenv("WORLDFOUNDRY_CACHE_COLLECTIVE_SHAPES", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _gather_shapes_cached(
    local_shape: torch.Size | tuple[int, ...],
    group: Optional[dist.ProcessGroup],
    device: torch.device,
) -> tuple[tuple[int, ...], ...]:
    """Gather shape metadata on-device, optionally caching fixed workloads."""

    shape = tuple(int(value) for value in local_shape)
    world_size = dist.get_world_size(group)
    key = (id(group), world_size, _rank_in_group(group), shape)
    if _collective_shape_cache_enabled():
        with _COLLECTIVE_SHAPE_CACHE_LOCK:
            cached = _COLLECTIVE_SHAPE_CACHE.get(key)
            if cached is not None:
                _COLLECTIVE_SHAPE_CACHE.move_to_end(key)
                return cached

    local = torch.tensor(shape, dtype=torch.int64, device=device)
    if hasattr(dist, "all_gather_into_tensor"):
        gathered_tensor = local.new_empty(world_size * local.numel())
        dist.all_gather_into_tensor(gathered_tensor, local, group=group)
        gathered_tensor = gathered_tensor.reshape(world_size, local.numel())
    else:
        gathered_list = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered_list, local, group=group)
        gathered_tensor = torch.stack(gathered_list, dim=0)
    result = tuple(tuple(int(value) for value in item) for item in gathered_tensor.cpu().tolist())
    if _collective_shape_cache_enabled():
        with _COLLECTIVE_SHAPE_CACHE_LOCK:
            _COLLECTIVE_SHAPE_CACHE[key] = result
            _COLLECTIVE_SHAPE_CACHE.move_to_end(key)
            while len(_COLLECTIVE_SHAPE_CACHE) > _COLLECTIVE_SHAPE_CACHE_LIMIT:
                _COLLECTIVE_SHAPE_CACHE.popitem(last=False)
    return result


def clear_collective_shape_cache() -> None:
    with _COLLECTIVE_SHAPE_CACHE_LOCK:
        _COLLECTIVE_SHAPE_CACHE.clear()


def reset_sequence_parallel_state() -> None:
    """Clear this module's sequence-parallel state (e.g. between unit tests).

    Does not destroy torch.distributed process groups; it only drops the
    references held by this module. ``nccl_info`` keeps its object identity
    because callers import the object directly.
    """
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = False
    _SEQUENCE_PARALLEL_GROUPS.clear()
    nccl_info.reset()
    clear_collective_shape_cache()


def initialize_sequence_parallel_group(sp_size: int) -> None:
    clear_collective_shape_cache()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % sp_size == 0, "world_size must be divisible by sequence_parallel_size"

    nccl_info.sp_size = sp_size
    nccl_info.global_rank = rank
    num_sequence_parallel_groups = world_size // sp_size
    logger.info(
        "[rank %s] initializing %s sequence-parallel group(s) of size %s",
        rank,
        num_sequence_parallel_groups,
        sp_size,
    )
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sp_size, (i + 1) * sp_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            set_sequence_parallel_group(group)
            nccl_info.rank_within_group = rank - i * sp_size
            nccl_info.group_id = i


def set_sequence_parallel_group(group: dist.ProcessGroup) -> None:
    clear_collective_shape_cache()
    _SEQUENCE_PARALLEL_GROUPS["sequence"] = group
    nccl_info.group = group
    nccl_info.sp_size = dist.get_world_size(group)
    nccl_info.rank_within_group = dist.get_rank(group)


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    return _SEQUENCE_PARALLEL_GROUPS.get("sequence", nccl_info.group)


def initialize_sequence_parallel_state(sequence_parallel_size: int):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        _SEQUENCE_PARALLEL_STATE = False
        nccl_info.sp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_group = 0
        nccl_info.group_id = int(os.getenv("RANK", "0"))


def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE


def initialize_distributed(seed):
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", rank))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    # Set defaults for distributed env vars required by env:// init method
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if world_size == 1 and not dist.is_initialized():
        # Single-GPU mode: skip full init, use default device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL sequence parallelism requires CUDA, but CUDA is unavailable.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=2**31 - 1),
            world_size=world_size,
            rank=rank,
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initialize_sequence_parallel_state(world_size)


def broadcast(input_: torch.Tensor, group: Optional[dist.ProcessGroup] = None):
    group = _resolve_group(group)
    src = dist.get_global_rank(group, 0) if group is not None else 0
    dist.broadcast(input_, src=src, group=group)
    return input_.contiguous()


def _all_to_all_4D(input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    group = _resolve_group(group)
    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        seq_lens = tuple(shape[0] for shape in _gather_shapes_cached((input.shape[1],), group, input.device))
        # uneven
        if seq_lens[-1] != seq_lens[0]:
            assert seq_lens[0] > seq_lens[-1]
            gap = seq_lens[0] - seq_lens[-1]
            if _rank_in_group(group) == seq_world_size - 1:
                assert input.shape[1] == seq_lens[-1]
                input = F.pad(input, (0, 0, 0, 0, 0, gap))
        else:
            gap = 0

        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        assert hc % seq_world_size == 0, f"Invalid size: {hc}, which should be divisible by {seq_world_size}"
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs).transpose(0, 2).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        if gap > 0:
            output = output[:, :-gap]

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape

        hc = shard_hc * seq_world_size
        if seqlen % seq_world_size != 0:
            new_seqlen = (seqlen // seq_world_size + 1) * seq_world_size
            gap = new_seqlen - seqlen
            input = F.pad(input, (0, 0, 0, 0, 0, gap))
            bs, seqlen, shard_hc, hs = input.shape
        else:
            gap = 0

        assert seqlen % seq_world_size == 0

        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)->
        # (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        if gap > 0 and _rank_in_group(group) == seq_world_size - 1:
            output = output[:, :-gap]

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _all_to_all_4D(input, scatter_idx, gather_idx, group=group)

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx),
            None,
            None,
        )


def all_to_all_4D(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    group = _resolve_group(group)
    return SeqAllToAll4D.apply(group, input_, scatter_dim, gather_dim)


def all_to_all_4d_many(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
    scatter_dim: int = 2,
    gather_dim: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Fuse equal-shaped 4D Q/K/V exchanges along their batch dimension."""

    values = tuple(tensors)
    if len(values) < 2:
        return values
    first = values[0]
    compatible = all(
        value.ndim == 4
        and value.shape == first.shape
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
        return tuple(all_to_all_4D(value, group, scatter_dim, gather_dim) for value in values)
    packed = torch.cat(values, dim=0)
    exchanged = all_to_all_4D(packed, group, scatter_dim, gather_dim)
    return exchanged.split(first.shape[0], dim=0)


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = all_to_all_tensor(input_, ctx.world_size, process_group, scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = all_to_all_tensor(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    group = _resolve_group(group)
    return _AllToAll.apply(input_, group, scatter_dim, gather_dim)


class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)

        sizes = _gather_shapes_cached(input_.shape, group, input_.device)

        ctx.gathered_dim_sizes = tuple(shape[dim] for shape in sizes)

        tensor_list = [torch.empty(sizes[i], dtype=input_.dtype, device=input_.device) for i in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        rank = _rank_in_group(group)
        dim = ctx.dim

        grad_input_list = torch.split(grad_output, ctx.gathered_dim_sizes, dim=dim)
        grad_input = grad_input_list[rank]

        return grad_input, None, None


def all_gather(input_: torch.Tensor, dim: int = 1, group=None):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim, _resolve_group(group))


def _split(input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    group = _resolve_group(group)
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size})"
    )
    output_list = torch.split(input_, dim_size // world_size, dim=dim)
    return output_list[rank].contiguous()


def _gather(input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    group = _resolve_group(group)
    world_size = dist.get_world_size(group)
    input_ = input_.contiguous()
    output_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(output_list, input_, group=group)
    return torch.cat(output_list, dim=dim).contiguous()


class _SplitForwardGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]):
        ctx.dim = dim
        ctx.group = group
        return _split(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _gather(grad_output, ctx.dim, ctx.group), None, None


class _GatherForwardSplitBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]):
        ctx.dim = dim
        ctx.group = group
        return _gather(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _split(grad_output, ctx.dim, ctx.group), None, None


def split_forward_gather_backward(
    input_: torch.Tensor,
    dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    return _SplitForwardGatherBackward.apply(input_, dim, _resolve_group(group))


def gather_forward_split_backward(
    input_: torch.Tensor,
    dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    return _GatherForwardSplitBackward.apply(input_, dim, _resolve_group(group))


# Optional xFuser compatibility used by official Wan-family runtimes.  The
# native sequence-parallel collectives above remain the canonical
# implementation for WorldFoundry models.
try:
    import importlib.util as _importlib_util

    if _importlib_util.find_spec("paifuser") is not None:
        from paifuser.xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
            get_world_group,
            init_distributed_environment,
            initialize_model_parallel,
            model_parallel_is_initialized,
        )
        from paifuser.xfuser.core.long_ctx_attention import xFuserLongContextAttention
    else:
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
            get_world_group,
            init_distributed_environment,
            initialize_model_parallel,
            model_parallel_is_initialized,
        )
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
except Exception:
    logger.debug("xFuser/paifuser unavailable; xFuser compatibility helpers disabled", exc_info=True)
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    xFuserLongContextAttention = None
    get_sp_group = None
    get_world_group = None
    init_distributed_environment = None
    initialize_model_parallel = None
    model_parallel_is_initialized = None


def set_multi_gpus_devices(
    ulysses_degree: int,
    ring_degree: int,
    classifier_free_guidance_degree: int = 1,
) -> torch.device:
    """Initialize an optional xFuser mesh and return its process-local device."""

    if ulysses_degree > 1 or ring_degree > 1 or classifier_free_guidance_degree > 1:
        if get_sp_group is None:
            raise RuntimeError("xFuser is required for the requested parallel degrees")
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        expected_world_size = (
            int(ring_degree)
            * int(ulysses_degree)
            * int(classifier_free_guidance_degree)
        )
        if dist.get_world_size() != expected_world_size:
            raise ValueError(
                f"world size {dist.get_world_size()} does not match requested "
                f"parallel degree {expected_world_size}"
            )
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=ring_degree * ulysses_degree,
            classifier_free_guidance_degree=classifier_free_guidance_degree,
            ring_degree=ring_degree,
            ulysses_degree=ulysses_degree,
        )
        return torch.device(f"cuda:{get_world_group().local_rank}")
    return torch.device("cuda", torch.cuda.current_device())


def sequence_parallel_chunk(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Return the xFuser-owned sequence shard, or the input in local mode."""

    if (
        get_sequence_parallel_world_size is None
        or model_parallel_is_initialized is None
        or not model_parallel_is_initialized()
    ):
        return x
    world_size = int(get_sequence_parallel_world_size())
    if world_size <= 1:
        return x
    if x.size(dim) % world_size:
        raise ValueError(
            f"dimension {dim} ({x.size(dim)}) must be divisible by "
            f"sequence-parallel world size {world_size}"
        )
    return torch.chunk(x, world_size, dim=dim)[get_sequence_parallel_rank()]


def sequence_parallel_all_gather(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Gather xFuser sequence shards, or return the input in local mode."""

    if (
        get_sequence_parallel_world_size is None
        or model_parallel_is_initialized is None
        or not model_parallel_is_initialized()
    ):
        return x
    if int(get_sequence_parallel_world_size()) <= 1:
        return x
    return get_sp_group().all_gather(x, dim=dim)
