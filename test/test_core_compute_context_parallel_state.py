"""CPU-only tests for the XC-10 slice A fix: context_parallel_util state hygiene.

The nine lowercase module globals now live in a single ContextParallelState
object. These tests exercise init -> read -> reset without CUDA by mocking
init_device_mesh and the torch.distributed rank/group queries.
"""

from __future__ import annotations

import pytest
import torch

from worldfoundry.core.distributed import context_parallel_util as cp_util


class _FakeMesh:
    def __init__(self):
        self.groups = {"dp": object(), "cp": object()}

    def get_group(self, mesh_dim):
        return self.groups[mesh_dim]


@pytest.fixture(autouse=True)
def clean_cp_state():
    cp_util.reset_context_parallel()
    yield
    cp_util.reset_context_parallel()


def _install_fake_distributed(monkeypatch, mesh, cp_ranks=(0, 1), dp_ranks=(0,)):
    monkeypatch.setattr(cp_util, "init_device_mesh", lambda *args, **kwargs: mesh)

    def fake_get_process_group_ranks(group):
        return list(cp_ranks) if group is mesh.groups["cp"] else list(dp_ranks)

    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", fake_get_process_group_ranks)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)


def test_init_populates_state_and_getters(monkeypatch):
    mesh = _FakeMesh()
    _install_fake_distributed(monkeypatch, mesh)

    cp_util.init_context_parallel(context_parallel_size=2, global_rank=0, world_size=2)

    assert cp_util.get_cp_size() == 2
    assert cp_util.get_dp_size() == 1
    assert cp_util.get_cp_group() is mesh.groups["cp"]
    assert cp_util.get_dp_group() is mesh.groups["dp"]
    assert cp_util.get_cp_rank() == 0
    assert cp_util.get_dp_rank() == 0
    assert cp_util.get_cp_rank_list() == [0, 1]


def test_legacy_module_attribute_reads_track_state(monkeypatch):
    mesh = _FakeMesh()
    _install_fake_distributed(monkeypatch, mesh)

    assert cp_util.cp_size is None
    assert cp_util.dp_group is None

    cp_util.init_context_parallel(context_parallel_size=2, global_rank=0, world_size=2)

    assert cp_util.cp_size == 2
    assert cp_util.dp_size == 1
    assert cp_util.cp_group is mesh.groups["cp"]
    assert cp_util.dp_group is mesh.groups["dp"]
    assert cp_util.cp_ranks == [0, 1]
    assert cp_util.dp_ranks == [0]
    assert cp_util.cp_rank == 0
    assert cp_util.dp_rank == 0


def test_reset_clears_state(monkeypatch):
    mesh = _FakeMesh()
    _install_fake_distributed(monkeypatch, mesh)
    cp_util.init_context_parallel(context_parallel_size=2, global_rank=0, world_size=2)
    state = cp_util.get_context_parallel_state()
    assert state.cp_size == 2

    cp_util.reset_context_parallel()

    # Same singleton object, all fields cleared.
    assert cp_util.get_context_parallel_state() is state
    assert cp_util.get_cp_size() is None
    assert cp_util.get_dp_size() is None
    assert cp_util.get_cp_group() is None
    assert cp_util.get_dp_group() is None
    assert cp_util.get_cp_rank() is None
    assert cp_util.get_dp_rank() is None
    assert cp_util.cp_size is None
    assert cp_util.cp_ranks is None
    assert state.cp_stream is None


def test_init_rejects_indivisible_world_size():
    with pytest.raises(RuntimeError, match="must be multiple of context_parallel_size"):
        cp_util.init_context_parallel(context_parallel_size=3, global_rank=0, world_size=4)


def test_unknown_module_attribute_raises():
    with pytest.raises(AttributeError, match="no attribute 'not_a_state_field'"):
        cp_util.not_a_state_field


def test_init_logs_instead_of_printing(monkeypatch, capsys, caplog):
    mesh = _FakeMesh()
    _install_fake_distributed(monkeypatch, mesh)

    with caplog.at_level("INFO", logger=cp_util.__name__):
        cp_util.init_context_parallel(context_parallel_size=2, global_rank=0, world_size=2)

    assert capsys.readouterr().out == ""
    assert any("init_device_mesh" in message for message in caplog.messages)
