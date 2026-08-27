"""CPU-only tests for the XC-10 slice B fix: sequence_parallel_runtime hygiene.

Covers the local (sp_size == 1) initialization path, group registration via
set_sequence_parallel_group with mocked torch.distributed queries, and the
new reset_sequence_parallel_state() teardown. No CUDA or process groups are
required.
"""

from __future__ import annotations

import pytest
import torch.distributed as dist

from worldfoundry.core.distributed import sequence_parallel_runtime as sp_runtime


@pytest.fixture(autouse=True)
def clean_sp_state():
    sp_runtime.reset_sequence_parallel_state()
    yield
    sp_runtime.reset_sequence_parallel_state()


def test_local_mode_initialization(monkeypatch):
    monkeypatch.setenv("RANK", "3")

    sp_runtime.initialize_sequence_parallel_state(1)

    assert sp_runtime.get_sequence_parallel_state() is False
    assert sp_runtime.nccl_info.sp_size == 1
    assert sp_runtime.nccl_info.global_rank == 3
    assert sp_runtime.nccl_info.group_id == 3


def test_set_group_updates_state_and_reset_clears_it(monkeypatch):
    fake_group = object()
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1)

    info = sp_runtime.nccl_info
    sp_runtime.set_sequence_parallel_group(fake_group)

    assert sp_runtime.get_sequence_parallel_group() is fake_group
    assert info.group is fake_group
    assert info.sp_size == 2
    assert info.rank_within_group == 1

    sp_runtime.reset_sequence_parallel_state()

    # nccl_info keeps its identity: callers import the object directly.
    assert sp_runtime.nccl_info is info
    assert sp_runtime.get_sequence_parallel_group() is None
    assert sp_runtime.get_sequence_parallel_state() is False
    assert info.group is None
    assert info.sp_size == 1
    assert info.rank_within_group == 0


def test_reset_clears_collective_shape_cache():
    with sp_runtime._COLLECTIVE_SHAPE_CACHE_LOCK:
        sp_runtime._COLLECTIVE_SHAPE_CACHE[("key",)] = ((1, 2),)

    sp_runtime.reset_sequence_parallel_state()

    assert not sp_runtime._COLLECTIVE_SHAPE_CACHE
