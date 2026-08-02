"""Contract tests for the model-neutral native viewer boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from .backend_loader import DUMMY_BACKEND, MODEL_BACKENDS, backend_class
from .server_base import InferenceModel

EXPLORER_ROOT = Path(__file__).resolve().parents[1]


def test_dynamic_backend_is_an_inference_model() -> None:
	assert issubclass(backend_class(DUMMY_BACKEND), InferenceModel)


def test_gen3c_model_id_selects_the_camera_native_backend() -> None:
	assert backend_class(model_id="gen3c").__name__ == "Gen3CModel"
	assert "gen3c" in MODEL_BACKENDS


@pytest.mark.parametrize("spec", ("missing_separator", ":Class", "module:"))
def test_dynamic_backend_rejects_invalid_specs(spec: str) -> None:
	with pytest.raises(ValueError, match="python.module:BackendClass"):
		backend_class(spec)


def test_user_facing_native_sources_are_model_neutral() -> None:
	paths = [
		EXPLORER_ROOT / "CMakeLists.txt",
		EXPLORER_ROOT / "api" / "client.py",
		*(EXPLORER_ROOT / "src").glob("*"),
		*(EXPLORER_ROOT / "include").rglob("*"),
	]
	offenders = [
		path.relative_to(EXPLORER_ROOT).as_posix()
		for path in paths
		if path.is_file() and "lyra2" in path.read_text(errors="ignore").lower()
	]
	assert offenders == []
