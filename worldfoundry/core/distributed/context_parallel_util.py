from __future__ import annotations

import logging
from dataclasses import dataclass, fields

import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed.device_mesh import init_device_mesh

from worldfoundry.core.distributed.device_mesh_collectives import all_to_all_tensor

logger = logging.getLogger(__name__)


@dataclass
class ContextParallelState:
    """Process-wide context-parallel topology (single source of truth, XC-10).

    Historically each field lived as a lowercase module-level global mutated
    through ``global`` statements. They are now grouped here; legacy
    module-attribute reads (``context_parallel_util.cp_size`` or
    ``from ... import cp_size``) keep working through the module-level
    ``__getattr__`` below.
    """

    dp_size: int | None = None
    cp_size: int | None = None
    dp_group: dist.ProcessGroup | None = None
    cp_group: dist.ProcessGroup | None = None
    cp_stream: torch.cuda.Stream | None = None
    dp_ranks: list[int] | None = None
    cp_ranks: list[int] | None = None
    dp_rank: int | None = None
    cp_rank: int | None = None

    def reset(self) -> None:
        for field in fields(self):
            setattr(self, field.name, None)


_CP_STATE = ContextParallelState()

# Names formerly exposed as lowercase module globals, kept readable for
# backwards compatibility via PEP 562 module __getattr__.
_LEGACY_STATE_ATTRS = tuple(field.name for field in fields(ContextParallelState))


def __getattr__(name: str):
    if name in _LEGACY_STATE_ATTRS:
        return getattr(_CP_STATE, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_context_parallel_state() -> ContextParallelState:
    """Return the mutable singleton holding the context-parallel topology."""
    return _CP_STATE


def reset_context_parallel() -> None:
    """Clear the context-parallel state (e.g. between unit tests).

    Does not destroy torch.distributed process groups; it only drops the
    references held by this module.
    """
    _CP_STATE.reset()


def init_context_parallel(context_parallel_size: int = 1, global_rank: int = 0, world_size: int = 1):
    if world_size % context_parallel_size != 0:
        raise RuntimeError(f"world_size {world_size} must be multiple of context_parallel_size {context_parallel_size}")

    state = _CP_STATE
    state.cp_size = context_parallel_size
    state.dp_size = world_size // context_parallel_size

    logger.info(
        "[rank %s] init_device_mesh [dp_size x cp_size]: [%s x %s]", global_rank, state.dp_size, state.cp_size
    )
    mesh_2d = init_device_mesh("cuda", (state.dp_size, state.cp_size), mesh_dim_names=("dp", "cp"))
    logger.info("[rank %s] mesh_2d: %s", global_rank, mesh_2d)

    state.dp_group = mesh_2d.get_group(mesh_dim="dp")
    state.cp_group = mesh_2d.get_group(mesh_dim="cp")
    state.dp_ranks = torch.distributed.get_process_group_ranks(state.dp_group)
    state.cp_ranks = torch.distributed.get_process_group_ranks(state.cp_group)
    state.dp_rank = dist.get_rank(group=state.dp_group)
    state.cp_rank = dist.get_rank(group=state.cp_group)

    current_global_rank = torch.distributed.get_rank()
    logger.info(
        "[rank %s] [dp_rank, cp_rank]: [%s, %s], dp_ranks: %s, cp_ranks: %s",
        current_global_rank,
        state.dp_rank,
        state.cp_rank,
        state.dp_ranks,
        state.cp_ranks,
    )


def get_cp_size():
    return _CP_STATE.cp_size


def get_dp_size():
    return _CP_STATE.dp_size


def get_cp_stream():
    if _CP_STATE.cp_stream is None:
        _CP_STATE.cp_stream = torch.cuda.Stream()
    return _CP_STATE.cp_stream


def get_dp_group():
    return _CP_STATE.dp_group


def get_cp_group():
    return _CP_STATE.cp_group


def get_dp_rank():
    return _CP_STATE.dp_rank


def get_cp_rank():
    return _CP_STATE.cp_rank


def get_cp_rank_list():
    if _CP_STATE.cp_ranks is None:
        _CP_STATE.cp_ranks = torch.distributed.get_process_group_ranks(_CP_STATE.cp_group)
    return _CP_STATE.cp_ranks


def cp_broadcast(tensor, cp_index=0):
    cp_ranks = get_cp_rank_list()
    torch.distributed.broadcast(tensor, cp_ranks[cp_index], group=_CP_STATE.cp_group)


def cp_broadcast_objects(tensor):
    raise NotImplementedError("cp_broadcast_objects method is not yet implemented")


def split_tensor_in_cp(input, seq_dim):
    cp_size = _CP_STATE.cp_size

    seq_size = input.shape[seq_dim]
    if seq_size % cp_size != 0:
        raise RuntimeError(f"seq_length {seq_size} in dim {seq_dim} must be multiple of cp_size {cp_size}")

    split_seq_size = seq_size // cp_size
    tensor_splits = input.split(split_seq_size, dim=seq_dim)
    return tensor_splits[get_cp_rank()]


def split_tensor_in_cp_2d(input, dim_hw, split_hw):
    cp_size = _CP_STATE.cp_size

    dim_h, dim_w = dim_hw
    split_h, split_w = split_hw
    if cp_size != split_h * split_w:
        raise RuntimeError(f"cp_size {cp_size} must equal split_h * split_w ({split_h} * {split_w})")

    seq_size_h = input.shape[dim_h]
    seq_size_w = input.shape[dim_w]
    if seq_size_h % split_h != 0:
        raise RuntimeError(f"seq_size_h {seq_size_h} in dim_h {dim_h} must be multiple of split_h {split_h}")
    if seq_size_w % split_w != 0:
        raise RuntimeError(f"seq_size_w {seq_size_w} in dim_w {dim_w} must be multiple of split_w {split_w}")

    split_seq_size_h = seq_size_h // split_h
    split_seq_size_w = seq_size_w // split_w

    tensor_splits = []
    for tensor_split_h in input.split(split_seq_size_h, dim=dim_h):
        tensor_splits.extend(tensor_split_h.split(split_seq_size_w, dim=dim_w))

    return tensor_splits[get_cp_rank()]


class GatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, process_group, seq_dim, frames):
        ctx.cp_group = process_group
        ctx.seq_dim = seq_dim
        ctx.frames = frames
        ctx.cp_size = get_cp_size()

        input = rearrange(input, "B (T S) C -> B T S C", T=frames)
        with torch.no_grad():
            input = input.contiguous()
            output_tensors = [torch.zeros_like(input) for _ in range(ctx.cp_size)]
            dist.all_gather(output_tensors, input, group=ctx.cp_group)
            output_tensor = torch.cat(output_tensors, dim=seq_dim)

        return rearrange(output_tensor, "B T S C -> B (T S) C", T=frames)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            grad_output = grad_output * ctx.cp_size
            grad_output = rearrange(grad_output, "B (T S) C -> B T S C", T=ctx.frames)
            grad_input = split_tensor_in_cp(grad_output, ctx.seq_dim)
            grad_input = rearrange(grad_input, "B T S C -> B (T S) C", T=ctx.frames)

        return grad_input, None, None, None


class SplitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, process_group, seq_dim):
        ctx.cp_group = process_group
        ctx.seq_dim = seq_dim
        ctx.cp_size = get_cp_size()
        return split_tensor_in_cp(input, ctx.seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            grad_output = grad_output / ctx.cp_size
            output_tensors = [torch.zeros_like(grad_output) for _ in range(ctx.cp_size)]
            dist.all_gather(output_tensors, grad_output, group=ctx.cp_group)
            grad_input = torch.cat(output_tensors, dim=ctx.seq_dim)

        return grad_input, None, None


class GatherFunction2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, process_group, seq_dim_hw, shape, split_hw):
        ctx.cp_group = process_group
        ctx.seq_dim_hw = seq_dim_hw
        ctx.split_hw = split_hw
        ctx.shape = shape
        ctx.cp_size = get_cp_size()

        t, h, w = shape
        dim_h, dim_w = seq_dim_hw
        split_h, split_w = split_hw
        if h % split_h != 0 or w % split_w != 0:
            raise RuntimeError(f"shape {(t, h, w)} is not divisible by split_hw {split_hw}")
        if t * (h // split_h) * (w // split_w) != input.shape[1]:
            raise RuntimeError("input sequence length does not match shape and split_hw")

        input = rearrange(input, "B (T H W) C -> B T H W C", T=t, H=h // split_h, W=w // split_w)
        with torch.no_grad():
            input = input.contiguous()
            output_tensors = [torch.zeros_like(input) for _ in range(ctx.cp_size)]
            dist.all_gather(output_tensors, input, group=ctx.cp_group)
            output_tensors_hs = []
            if ctx.cp_size % split_w != 0:
                raise RuntimeError(f"cp_size {ctx.cp_size} must be divisible by split_w {split_w}")
            for i in range(ctx.cp_size // split_w):
                output_tensors_hs.append(torch.cat(output_tensors[i * split_w : (i + 1) * split_w], dim=dim_w))
            output_tensor = torch.cat(output_tensors_hs, dim=dim_h)

        return rearrange(output_tensor, "B T H W C -> B (T H W) C")

    @staticmethod
    def backward(ctx, grad_output):
        t, h, w = ctx.shape
        with torch.no_grad():
            grad_output = grad_output * ctx.cp_size
            grad_output = rearrange(grad_output, "B (T H W) C -> B T H W C", T=t, H=h, W=w)
            grad_input = split_tensor_in_cp_2d(grad_output, ctx.seq_dim_hw, ctx.split_hw)
            grad_input = rearrange(grad_input, "B T H W C -> B (T H W) C")

        return grad_input, None, None, None, None


class SplitFunction2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, process_group, seq_dim_hw, split_hw):
        ctx.cp_group = process_group
        ctx.seq_dim_hw = seq_dim_hw
        ctx.split_hw = split_hw
        ctx.cp_size = get_cp_size()
        return split_tensor_in_cp_2d(input, ctx.seq_dim_hw, split_hw)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            grad_output = grad_output / ctx.cp_size
            output_tensors = [torch.zeros_like(grad_output) for _ in range(ctx.cp_size)]
            dist.all_gather(output_tensors, grad_output, group=ctx.cp_group)

            split_h, split_w = ctx.split_hw
            dim_h, dim_w = ctx.seq_dim_hw
            if ctx.cp_size % split_w != 0:
                raise RuntimeError(f"cp_size {ctx.cp_size} must be divisible by split_w {split_w}")
            output_tensors_hs = []
            for i in range(ctx.cp_size // split_w):
                output_tensors_hs.append(torch.cat(output_tensors[i * split_w : (i + 1) * split_w], dim=dim_w))
            grad_input = torch.cat(output_tensors_hs, dim=dim_h)

        return grad_input, None, None, None


def gather_cp(input, frames):
    cp_process_group = get_cp_group()
    return GatherFunction.apply(input, cp_process_group, 2, frames)


def split_cp(input, seq_dim):
    cp_process_group = get_cp_group()
    return SplitFunction.apply(input, cp_process_group, seq_dim)


def gather_cp_2d(input, shape, split_hw):
    cp_process_group = get_cp_group()
    return GatherFunction2D.apply(input, cp_process_group, (2, 3), shape, split_hw)


def split_cp_2d(input, seq_dim_hw, split_hw):
    cp_process_group = get_cp_group()
    return SplitFunction2D.apply(input, cp_process_group, seq_dim_hw, split_hw)


class ReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, process_group):
        ctx.cp_group = process_group
        output = input.detach().clone()
        dist.all_reduce(output, group=ctx.cp_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.detach().clone(), None


class ReplicateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, process_group):
        ctx.cp_group = process_group
        return input.detach().clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.detach().clone()
        dist.all_reduce(grad_input, group=ctx.cp_group)
        return grad_input, None


def reduce_cp(partial_sum, partial_square_sum):
    cp_process_group = get_cp_group()
    all_sum = ReduceFunction.apply(partial_sum, cp_process_group)
    all_square_sum = ReduceFunction.apply(partial_square_sum, cp_process_group)
    return all_sum, all_square_sum


def replicate_cp(all_mean, all_var):
    cp_process_group = get_cp_group()
    all_mean = ReplicateFunction.apply(all_mean, cp_process_group)
    all_var = ReplicateFunction.apply(all_var, cp_process_group)
    return all_mean, all_var


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(process_group)
        return all_to_all_tensor(input_, world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return return_grad, None, None, None


def all_to_all_with_pad(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    scatter_pad: int = 0,
    gather_pad: int = 0,
):
    if scatter_pad > 0:
        pad_shape = list(input_.shape)
        pad_shape[scatter_dim] = scatter_pad
        pad_tensor = torch.zeros(pad_shape, device=input_.device, dtype=input_.dtype)
        input_ = torch.cat([input_, pad_tensor], dim=scatter_dim)

    world_size = dist.get_world_size(process_group)
    if input_.shape[scatter_dim] % world_size != 0:
        raise RuntimeError(
            f"Dimension to scatter ({input_.shape[scatter_dim]}) is not divisible by world size ({world_size})"
        )
    input_ = _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)

    if gather_pad > 0:
        input_ = input_.narrow(gather_dim, 0, input_.size(gather_dim) - gather_pad)

    return input_


def dynamic_switch(x, scatter_dim, gather_dim):
    return all_to_all_with_pad(
        x,
        get_cp_group(),
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
        scatter_pad=0,
        gather_pad=0,
    )


def get_optimal_split(size):
    factors = []
    for i in range(1, int(size**0.5) + 1):
        if size % i == 0:
            factors.append([i, size // i])
    return min(factors, key=lambda x: abs(x[0] - x[1]))
