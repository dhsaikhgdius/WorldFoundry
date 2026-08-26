"""Lazy public facade for pipeline runner contracts and data-backed bindings.

Each public name is mapped to the submodule that defines it.  Accessing any
exported symbol triggers a lazy ``import_module`` so that heavyweight
sub-modules (YAML parsing, plugin discovery, etc.) are only loaded when
actually needed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ── Lazy import map ─────────────────────────────────────────────────────
# Maps each exported symbol name to the dotted module path that defines it.
_EXPORT_MODULES: dict[str, str] = {
    "ACTION_CONTROL_KEYS": "worldfoundry.evaluation.models.pipelines.invocation",
    "CONSUMED_INPUT_KEYS": "worldfoundry.evaluation.models.pipelines.invocation",
    "DEFAULT_PIPELINE_ALIASES_ROOT": "worldfoundry.evaluation.models.pipelines.aliases",
    "DEFAULT_PIPELINE_BINDINGS_ROOT": "worldfoundry.evaluation.models.pipelines.bindings",
    "ENTRY_POINT_GROUP": "worldfoundry.evaluation.models.pipelines.discovery",
    "ENV_VAR": "worldfoundry.evaluation.models.pipelines.discovery",
    "IMAGE_INPUT_KEYS": "worldfoundry.evaluation.models.pipelines.invocation",
    "PipelineAliasGroup": "worldfoundry.evaluation.models.pipelines.aliases",
    "PipelineAliasRegistry": "worldfoundry.evaluation.models.pipelines.aliases",
    "PipelineBinding": "worldfoundry.evaluation.models.pipelines.bindings",
    "PipelineBindingRegistry": "worldfoundry.evaluation.models.pipelines.bindings",
    "PipelineInvocation": "worldfoundry.evaluation.models.pipelines.invocation",
    "PipelineLifecycleContext": "worldfoundry.evaluation.models.pipelines.lifecycle",
    "PipelineResultContext": "worldfoundry.evaluation.models.pipelines.results",
    "PipelineRunnerSpec": "worldfoundry.evaluation.models.pipelines.loading",
    "PipelineRuntimeProfile": "worldfoundry.evaluation.models.pipelines.lifecycle",
    "PipelineRoute": "worldfoundry.evaluation.models.pipelines.bindings",
    "TEXT_INPUT_KEYS": "worldfoundry.evaluation.models.pipelines.invocation",
    "VIDEO_INPUT_KEYS": "worldfoundry.evaluation.models.pipelines.invocation",
    "WorldFoundryPipelineInvocationProtocol": "worldfoundry.evaluation.models.pipelines.lifecycle",
    "WorldFoundryPipelineLifecycle": "worldfoundry.evaluation.models.pipelines.lifecycle",
    "WorldFoundryPipelineProtocol": "worldfoundry.evaluation.models.pipelines.lifecycle",
    "build_alias_mapping": "worldfoundry.evaluation.models.pipelines.aliases",
    "build_pipeline_invocation": "worldfoundry.evaluation.models.pipelines.invocation",
    "build_pipeline_runner_spec": "worldfoundry.evaluation.models.pipelines.loading",
    "call_pipeline_from_pretrained": "worldfoundry.evaluation.models.pipelines.loading",
    "clear_pipeline_binding_discovery_cache": "worldfoundry.evaluation.models.pipelines.discovery",
    "discover_pipeline_bindings": "worldfoundry.evaluation.models.pipelines.discovery",
    "failed_generation_result": "worldfoundry.evaluation.models.pipelines.results",
    "first_text": "worldfoundry.evaluation.models.pipelines.bindings",
    "generation_result_from_pipeline": "worldfoundry.evaluation.models.pipelines.results",
    "import_pipeline_target": "worldfoundry.evaluation.models.pipelines.loading",
    "invoke_pipeline": "worldfoundry.evaluation.models.pipelines.invocation",
    "load_named_pipeline": "worldfoundry.evaluation.models.pipelines.loading",
    "load_pipeline_alias_groups": "worldfoundry.evaluation.models.pipelines.aliases",
    "load_pipeline_alias_registry": "worldfoundry.evaluation.models.pipelines.aliases",
    "load_pipeline_binding": "worldfoundry.evaluation.models.pipelines.bindings",
    "load_pipeline_binding_registry": "worldfoundry.evaluation.models.pipelines.bindings",
    "load_pipeline_binding_registry_with_plugins": "worldfoundry.evaluation.models.pipelines.bindings",
    "load_pipeline_bindings": "worldfoundry.evaluation.models.pipelines.bindings",
    "load_pipeline_from_config": "worldfoundry.evaluation.models.pipelines.loading",
    "load_pipeline_from_spec": "worldfoundry.evaluation.models.pipelines.loading",
    "merge_pipeline_binding_plugins": "worldfoundry.evaluation.models.pipelines.bindings",
    "pipeline_metadata": "worldfoundry.evaluation.models.pipelines.results",
    "pipeline_result_error": "worldfoundry.evaluation.models.pipelines.results",
    "pipeline_result_status": "worldfoundry.evaluation.models.pipelines.results",
    "pipeline_route_from_config": "worldfoundry.evaluation.models.pipelines.loading",
    "pipeline_target_from_config": "worldfoundry.evaluation.models.pipelines.loading",
    "request_output_dir": "worldfoundry.evaluation.models.pipelines.invocation",
    "resolve_pipeline_route": "worldfoundry.evaluation.models.pipelines.bindings",
    "resolve_hfd_cached_repo": "worldfoundry.evaluation.models.pipelines.loading",
    "resolve_model_path_entry": "worldfoundry.evaluation.models.pipelines.loading",
    "resolve_optional_model_path_entry": "worldfoundry.evaluation.models.pipelines.loading",
    "resolve_pipeline_binding": "worldfoundry.evaluation.models.pipelines.bindings",
    "runtime_profile_execution_metadata": "worldfoundry.evaluation.models.pipelines.bindings",
    "runtime_profile_id": "worldfoundry.evaluation.models.pipelines.bindings",
    "sample_output_path": "worldfoundry.evaluation.models.pipelines.invocation",
}

__all__ = sorted(_EXPORT_MODULES)


# ── Lazy attribute resolution ──────────────────────────────────────────


def __getattr__(name: str) -> Any:
    """Lazily import and resolve submodules and shortcut exports on access."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the list of all exported attributes."""
    return sorted({*globals(), *__all__})
