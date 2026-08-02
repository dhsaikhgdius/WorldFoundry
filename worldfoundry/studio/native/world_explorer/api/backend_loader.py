"""Resolve a model backend without coupling the native viewer to one runtime."""

from __future__ import annotations

import inspect
import os
from importlib import import_module
from typing import Type

from .server_base import InferenceModel

BACKEND_ENV = "WORLDFOUNDRY_EXPLORER_BACKEND"
MODEL_ID_ENV = "WORLDFOUNDRY_EXPLORER_MODEL_ID"
DEFAULT_BACKEND = (
	"worldfoundry.studio.native.world_explorer.api.server_lyra:LyraModel"
)
DUMMY_BACKEND = (
	"worldfoundry.studio.native.world_explorer.api.server_lyra:DummyLyraModel"
)
MODEL_BACKENDS = {
	"gen3c": "worldfoundry.studio.native.world_explorer.api.server_gen3c:Gen3CModel",
	"gen3c-cosmos-7b": "worldfoundry.studio.native.world_explorer.api.server_gen3c:Gen3CModel",
	"lyra": DEFAULT_BACKEND,
	"lyra-2": DEFAULT_BACKEND,
	"lyra2": DEFAULT_BACKEND,
}


def backend_class(
	spec: str | None = None,
	*,
	model_id: str | None = None,
) -> Type[InferenceModel]:
	"""Load ``module:Class`` and validate the native explorer backend contract."""
	selected_model = (model_id or os.environ.get(MODEL_ID_ENV) or "").strip().lower()
	resolved = (
		spec
		or os.environ.get(BACKEND_ENV)
		or MODEL_BACKENDS.get(selected_model)
		or DEFAULT_BACKEND
	).strip()
	module_name, separator, class_name = resolved.partition(":")
	if not separator or not module_name or not class_name:
		raise ValueError(
			f"{BACKEND_ENV} must use the form 'python.module:BackendClass'; got {resolved!r}."
		)
	candidate = getattr(import_module(module_name), class_name)
	if not inspect.isclass(candidate) or not issubclass(candidate, InferenceModel):
		raise TypeError(f"{resolved!r} must resolve to an InferenceModel subclass.")
	return candidate


__all__ = [
	"BACKEND_ENV",
	"DEFAULT_BACKEND",
	"DUMMY_BACKEND",
	"MODEL_BACKENDS",
	"MODEL_ID_ENV",
	"backend_class",
]
