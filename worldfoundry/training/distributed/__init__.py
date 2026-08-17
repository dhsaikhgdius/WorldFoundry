"""Lazy public API for process groups, meshes, and FSDP2."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "FSDP2_APPLICATION_SCHEMA": ".fsdp",
    "FSDP2Application": ".fsdp",
    "apply_fsdp2": ".fsdp",
    "apply_fsdp2_frozen_reference": ".fsdp",
    "PARALLEL_PLAN_SCHEMA": ".parallel",
    "DistributedTrainingContext": ".parallel",
    "ParallelPlan": ".parallel",
    "FlowTrajectoryShardRequest": ".flow_rollout",
    "FlowTrajectoryShardResult": ".flow_rollout",
    "RayFlowSamplerConfig": ".flow_rollout",
    "RayFlowTrajectorySampler": ".flow_rollout",
    "RayFlowTrajectoryWorker": ".flow_rollout",
    "attach_ray_flow_policy_rollout": ".flow_rollout",
    "partition_complete_flow_groups": ".flow_rollout",
    "DeviceLease": ".ray_runtime",
    "DevicePoolPlanner": ".ray_runtime",
    "RayDevicePool": ".ray_runtime",
    "RayDevicePoolConfig": ".ray_runtime",
    "RayRoleWorker": ".ray_runtime",
    "RayWorkerContext": ".ray_runtime",
    "RayWorkerGroup": ".ray_runtime",
    "RolloutPlacement": ".ray_runtime",
    "RayPostTrainingRuntime": ".rollout_runtime",
    "RayPostTrainingRuntimeConfig": ".rollout_runtime",
    "ray_runtime_config_from_rollout_spec": ".rollout_runtime",
    "TrainerBinding": ".rollout_runtime",
    "ModuleWeightReceiver": ".weight_sync",
    "NativeWeightSynchronizer": ".weight_sync",
    "WeightBucket": ".weight_sync",
    "WeightKind": ".weight_sync",
    "WeightSyncReport": ".weight_sync",
    "WeightUpdateHeader": ".weight_sync",
    "WeightUpdateReceiver": ".weight_sync",
    "build_weight_buckets": ".weight_sync",
    "materialize_weight_tensors": ".weight_sync",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
