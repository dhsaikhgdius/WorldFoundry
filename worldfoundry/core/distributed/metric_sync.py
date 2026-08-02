"""Distributed metric logging and synchronization helpers."""

from __future__ import annotations

import builtins
import datetime
import os
import pickle
import time
from collections import defaultdict, deque
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

logger = getLogger()


def setup_for_distributed(is_master) -> None:
    """Disable plain print on non-master ranks unless forced."""

    builtin_print = builtins.print

    def print(*args, **kwargs):  # noqa: A001
        force = kwargs.pop("force", False)
        force = force or get_world_size() > 8
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print(f"[{now}] ", end="")
            builtin_print(*args, **kwargs)

    builtins.print = print


def init_distributed(port=37124, rank_and_world_size=(None, None)):
    rank, world_size = rank_and_world_size
    dist_url = "env://"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(port))
    print("Using port", os.environ["MASTER_PORT"])

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            gpu = int(os.environ["LOCAL_RANK"])
        except Exception:
            logger.info("torchrun env vars not set")
    elif "SLURM_PROCID" in os.environ:
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            gpu = rank % torch.cuda.device_count()
            os.environ["MASTER_ADDR"] = os.environ.get("HOSTNAME", "127.0.0.1")
        except Exception:
            logger.info("SLURM vars not set")
    else:
        rank = 0
        world_size = 1
        gpu = 0
        os.environ["MASTER_ADDR"] = "127.0.0.1"

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
