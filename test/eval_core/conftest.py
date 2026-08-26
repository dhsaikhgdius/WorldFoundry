from __future__ import annotations

import pytest

from test.eval_core.factories import (
    write_benchmark_manifest,
    write_json_document,
    write_model_manifest,
    write_targets_manifest,
    write_zoo_manifest,
)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(pytest.mark.fast_eval_core)


@pytest.fixture
def json_document_factory():
    """Return :func:`write_json_document` for eval_core tests."""

    return write_json_document


@pytest.fixture
def targets_manifest_factory():
    """Return :func:`write_targets_manifest` for VLA/VA/WAM scripts."""

    return write_targets_manifest


@pytest.fixture
def model_manifest_factory():
    """Return :func:`write_model_manifest` for model-zoo scripts."""

    return write_model_manifest


@pytest.fixture
def benchmark_manifest_factory():
    """Return :func:`write_benchmark_manifest` for benchmark-zoo scripts."""

    return write_benchmark_manifest


@pytest.fixture
def zoo_manifest_factory():
    """Return :func:`write_zoo_manifest` for arbitrary zoo filenames."""

    return write_zoo_manifest
