"""Tests for Studio workspace max-jobs resolution."""

from __future__ import annotations

import pytest

from worldfoundry.studio.launch_config import resolve_workspace_max_jobs


def test_resolve_workspace_max_jobs_honors_env() -> None:
    assert resolve_workspace_max_jobs(environ={"WORLDFOUNDRY_WORKSPACE_MAX_JOBS": "5"}, device_count=8) == 5


def test_resolve_workspace_max_jobs_adapts_to_gpu_count() -> None:
    assert resolve_workspace_max_jobs(environ={}, device_count=4) == 4
    assert resolve_workspace_max_jobs(environ={}, device_count=32, cap=16) == 16


def test_resolve_workspace_max_jobs_cpu_fallback() -> None:
    assert resolve_workspace_max_jobs(environ={}, device_count=0) == 1


def test_resolve_workspace_max_jobs_invalid_env() -> None:
    with pytest.raises(ValueError, match="WORLDFOUNDRY_WORKSPACE_MAX_JOBS"):
        resolve_workspace_max_jobs(environ={"WORLDFOUNDRY_WORKSPACE_MAX_JOBS": "nope"}, device_count=2)
