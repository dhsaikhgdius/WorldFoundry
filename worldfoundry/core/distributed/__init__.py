"""Distributed tensor helpers shared by optimized runtime modules.

CM-01: every submodule here imports ``torch`` at module top level, so eager
re-exports made ``import worldfoundry.core.distributed.logging`` (and thereby
every CLI startup that configured logging) pay the full torch import. The
re-exports below are resolved lazily via PEP 562 ``__getattr__`` — the same
pattern as :mod:`worldfoundry.evaluation` — so importing one submodule no
longer drags in the whole distributed stack.
"""

from __future__ import annotations

from importlib import import_module

_EXPORT_MAP = {
    # .context_parallel
    "broadcast": (".context_parallel", "broadcast"),
    "broadcast_split_tensor": (".context_parallel", "broadcast_split_tensor"),
    "cat_outputs_cp": (".context_parallel", "cat_outputs_cp"),
    "cat_outputs_cp_object_list": (".context_parallel", "cat_outputs_cp_object_list"),
    "cat_outputs_cp_with_grad": (".context_parallel", "cat_outputs_cp_with_grad"),
    "find_split": (".context_parallel", "find_split"),
    "robust_broadcast": (".context_parallel", "robust_broadcast"),
    "split_inputs_cp": (".context_parallel", "split_inputs_cp"),
    "split_inputs_cp_object_list": (".context_parallel", "split_inputs_cp_object_list"),
    # .device_mesh_collectives
    "DTensorFastEmaModelUpdater": (".device_mesh_collectives", "DTensorFastEmaModelUpdater"),
    "FastEmaModelUpdater": (".device_mesh_collectives", "FastEmaModelUpdater"),
    "broadcast_dtensor_model_states": (".device_mesh_collectives", "broadcast_dtensor_model_states"),
    "get_local_tensor_if_DTensor": (".device_mesh_collectives", "get_local_tensor_if_DTensor"),
    "get_local_tensor_if_dtensor": (".device_mesh_collectives", "get_local_tensor_if_dtensor"),
    # .generic_collectives
    "get_global_rank": (".generic_collectives", "get_rank"),
    "get_world_size": (".generic_collectives", "get_world_size"),
    "is_distributed_initialized": (".generic_collectives", "is_dist_initialized"),
    # .inference_runtime
    "dist_init": (".inference_runtime", "dist_init"),
    "get_distributed_device": (".inference_runtime", "get_device"),
    "is_last_rank": (".inference_runtime", "is_last_rank"),
    "is_last_tp_cp_rank": (".inference_runtime", "is_last_tp_cp_rank"),
    # .logging
    "print_per_rank": (".logging", "print_per_rank"),
    "print_rank_0": (".logging", "print_rank_0"),
    # .model_parallel_groups
    "destroy_model_parallel": (".model_parallel_groups", "destroy_model_parallel"),
    "get_cp_group": (".model_parallel_groups", "get_cp_group"),
    "get_cp_rank": (".model_parallel_groups", "get_cp_rank"),
    "get_cp_world_size": (".model_parallel_groups", "get_cp_world_size"),
    "get_dp_group": (".model_parallel_groups", "get_dp_group"),
    "get_dp_group_gloo": (".model_parallel_groups", "get_dp_group_gloo"),
    "get_dp_rank": (".model_parallel_groups", "get_dp_rank"),
    "get_dp_world_size": (".model_parallel_groups", "get_dp_world_size"),
    "get_model_parallel_group": (".model_parallel_groups", "get_model_parallel_group"),
    "get_pipeline_model_parallel_first_rank": (
        ".model_parallel_groups",
        "get_pipeline_model_parallel_first_rank",
    ),
    "get_pipeline_model_parallel_last_rank": (
        ".model_parallel_groups",
        "get_pipeline_model_parallel_last_rank",
    ),
    "get_pipeline_model_parallel_next_rank": (
        ".model_parallel_groups",
        "get_pipeline_model_parallel_next_rank",
    ),
    "get_pipeline_model_parallel_prev_rank": (
        ".model_parallel_groups",
        "get_pipeline_model_parallel_prev_rank",
    ),
    "get_pp_group": (".model_parallel_groups", "get_pp_group"),
    "get_pp_rank": (".model_parallel_groups", "get_pp_rank"),
    "get_pp_world_size": (".model_parallel_groups", "get_pp_world_size"),
    "get_tensor_model_parallel_last_rank": (
        ".model_parallel_groups",
        "get_tensor_model_parallel_last_rank",
    ),
    "get_tensor_model_parallel_ranks": (".model_parallel_groups", "get_tensor_model_parallel_ranks"),
    "get_tensor_model_parallel_src_rank": (
        ".model_parallel_groups",
        "get_tensor_model_parallel_src_rank",
    ),
    "get_tp_group": (".model_parallel_groups", "get_tp_group"),
    "get_tp_rank": (".model_parallel_groups", "get_tp_rank"),
    "get_tp_world_size": (".model_parallel_groups", "get_tp_world_size"),
    "initialize_model_parallel": (".model_parallel_groups", "initialize_model_parallel"),
    "model_parallel_is_initialized": (".model_parallel_groups", "model_parallel_is_initialized"),
    # .pipeline_parallel
    "PPScheduler": (".pipeline_parallel", "PPScheduler"),
    "init_pp_scheduler": (".pipeline_parallel", "init_pp_scheduler"),
    "pp_scheduler": (".pipeline_parallel", "pp_scheduler"),
    # .rank_orchestration
    "DistributedOpSpec": (".rank_orchestration", "DistributedOpSpec"),
    "PayloadBus": (".rank_orchestration", "PayloadBus"),
    "RankCoordinator": (".rank_orchestration", "RankCoordinator"),
    "SignalBus": (".rank_orchestration", "SignalBus"),
    "distributed_op": (".rank_orchestration", "distributed_op"),
}

__all__ = sorted(_EXPORT_MAP)


def __getattr__(name: str):
    """Lazily resolve re-exported symbols and sibling submodules."""
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    try:
        module = import_module(f".{name}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name == f"{__name__}.{name}":
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
        raise
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Return the list of all exported and public attributes."""
    return sorted({*globals(), *__all__})
