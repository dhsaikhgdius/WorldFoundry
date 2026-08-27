"""Distributed metric logging and synchronization helpers.

Library-path diagnostics in this module go through ``logging``. The one
intentional exception is :meth:`MetricLogger.log_every`, whose progress lines
are emitted via ``builtins.print`` so that the rank-filtered wrapper installed
by :func:`setup_for_distributed` keeps them master-rank-only.
"""

from __future__ import annotations

import builtins
import datetime
import os
import pickle
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from logging import getLogger

import torch
import torch.distributed as dist

from .generic_collectives import (
    get_collective_device,
    get_rank,
    get_world_size,
)
from .generic_collectives import (
    is_dist_initialized as is_dist_avail_and_initialized,
)
from .generic_collectives import (
    is_master as is_main_process,
)

logger = getLogger(__name__)

# Original builtins.print, captured the first time setup_for_distributed patches it.
_ORIGINAL_BUILTINS_PRINT = None


def setup_for_distributed(is_master) -> None:
    """Disable plain print on non-master ranks unless forced.

    This monkey-patches ``builtins.print`` process-wide (including third-party
    libraries). Use :func:`restore_builtins_print` or
    :func:`builtins_print_unpatched` to undo it. Note the historical quirk
    inherited from the MAE codebase: with ``world_size > 8`` every rank prints
    (``force`` is implied); this behavior is preserved as-is.
    """

    global _ORIGINAL_BUILTINS_PRINT

    builtin_print = builtins.print
    if getattr(builtin_print, "_worldfoundry_rank_filtered_print", False):
        # Re-entrant setup: wrap the original once instead of chaining wrappers
        # (chaining would duplicate timestamps and capture stale master flags).
        builtin_print = _ORIGINAL_BUILTINS_PRINT
    if _ORIGINAL_BUILTINS_PRINT is None:
        _ORIGINAL_BUILTINS_PRINT = builtin_print

    def print(*args, **kwargs):  # noqa: A001
        force = kwargs.pop("force", False)
        force = force or get_world_size() > 8
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print(f"[{now}] ", end="")
            builtin_print(*args, **kwargs)

    setattr(print, "_worldfoundry_rank_filtered_print", True)
    builtins.print = print
    logger.info(
        "Patched builtins.print process-wide with a rank-filtered wrapper (is_master=%s); "
        "call worldfoundry.core.distributed.metric_sync.restore_builtins_print() to undo.",
        bool(is_master),
    )


def restore_builtins_print() -> bool:
    """Restore the original ``builtins.print``; returns True if a patch was removed."""

    if _ORIGINAL_BUILTINS_PRINT is None or not getattr(builtins.print, "_worldfoundry_rank_filtered_print", False):
        return False
    builtins.print = _ORIGINAL_BUILTINS_PRINT
    logger.info("Restored the original builtins.print.")
    return True


@contextmanager
def builtins_print_unpatched():
    """Temporarily restore the original ``builtins.print`` within a scope."""

    patched_print = builtins.print if getattr(builtins.print, "_worldfoundry_rank_filtered_print", False) else None
    restore_builtins_print()
    try:
        yield
    finally:
        if patched_print is not None:
            builtins.print = patched_print


def init_distributed(port=37124, rank_and_world_size=(None, None)):
    rank, world_size = rank_and_world_size
    gpu = None
    dist_url = "env://"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(port))
    logger.info("Using port %s", os.environ["MASTER_PORT"])

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            gpu = int(os.environ["LOCAL_RANK"])
        except Exception as exc:
            raise RuntimeError(
                "RANK/WORLD_SIZE are set but the torchrun environment is incomplete or invalid "
                f"(RANK={os.environ.get('RANK')!r}, WORLD_SIZE={os.environ.get('WORLD_SIZE')!r}, "
                f"LOCAL_RANK={os.environ.get('LOCAL_RANK')!r}): {exc}"
            ) from exc
    elif "SLURM_PROCID" in os.environ:
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            gpu = rank % max(torch.cuda.device_count(), 1)
            os.environ["MASTER_ADDR"] = os.environ.get("HOSTNAME", "127.0.0.1")
        except Exception as exc:
            raise RuntimeError(
                "SLURM_PROCID is set but the SLURM environment is incomplete or invalid "
                f"(SLURM_NTASKS={os.environ.get('SLURM_NTASKS')!r}, "
                f"SLURM_PROCID={os.environ.get('SLURM_PROCID')!r}): {exc}"
            ) from exc
    else:
        rank = 0
        world_size = 1
        gpu = 0
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    if rank is None or world_size is None or gpu is None:
        raise RuntimeError(
            "init_distributed could not determine rank/world_size/local device; "
            "pass rank_and_world_size explicitly or launch via torchrun/SLURM."
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        init_method=dist_url,
    )
    return world_size, rank, gpu, True


class SmoothedValue:
    """Track a series of values and expose smoothed statistics."""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        tensor = torch.tensor(
            [self.count, self.total],
            dtype=torch.float64,
            device=get_collective_device(),
        )
        dist.barrier()
        dist.all_reduce(tensor)
        values = tensor.tolist()
        self.count = int(values[0])
        self.total = values[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert isinstance(value, (float, int))
            self.meters[key].update(value)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """Yield from ``iterable`` while printing periodic progress lines.

        Progress output intentionally uses ``builtins.print`` (not ``logging``)
        so the rank-filtered wrapper installed by :func:`setup_for_distributed`
        restricts it to the master rank; do not convert these to logger calls.
        """
        index = 0
        header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        mb = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if index % print_freq == 0 or index == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - index)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            index,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / mb,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            index,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            index += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")
        self.update(total_time=total_time)


def sync_fid_loss_fns(fid_loss_fn, device="cuda"):
    """Synchronize FID metric objects across all distributed ranks."""

    if not is_dist_avail_and_initialized():
        return fid_loss_fn

    serialized_fid_loss_fn = pickle.dumps(fid_loss_fn)
    gathered_fid_loss_fn = [None] * dist.get_world_size()

    dist.barrier()
    dist.all_gather_object(gathered_fid_loss_fn, serialized_fid_loss_fn)

    from torcheval.metrics import FrechetInceptionDistance

    final_fid_loss_fn = {sec: FrechetInceptionDistance(feature_dim=2048).to(device) for sec in [1, 2, 4, 8, 16]}
    for serialized_rank_metrics in gathered_fid_loss_fn:
        rank_metrics = pickle.loads(serialized_rank_metrics)
        for sec in [1, 2, 4, 8, 16]:
            final_fid_loss_fn[sec].merge_state([rank_metrics[sec]])

    return final_fid_loss_fn


__all__ = [
    "MetricLogger",
    "SmoothedValue",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_dist_avail_and_initialized",
    "is_main_process",
    "setup_for_distributed",
    "sync_fid_loss_fns",
]
