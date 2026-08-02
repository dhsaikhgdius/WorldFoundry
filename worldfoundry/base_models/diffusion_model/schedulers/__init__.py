"""WorldFoundry canonical numerical schedulers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Cosmos3FlowUniPCScheduler": ".cosmos3",
    "FlowDPMSolverMultistepScheduler": ".flow_dpm",
    "FlowMatchEulerScheduler": ".flow_match",
    "FlowUniPCMultistepScheduler": ".flow_unipc",
    "HunyuanVideoFlowMatchEulerScheduler": ".hunyuan_video",
    "InferenceFlowMatchScheduler": ".wan",
    "KarrasX0AB2Scheduler": ".karras_x0",
    "KarrasX0EulerScheduler": ".karras_x0",
    "LTXFixedEulerScheduler": ".ltx",
    "SanaSCMScheduler": ".sana",
    "SanaFlowDPMScheduler": ".sana",
    "SanaStreamingEulerScheduler": ".sana",
    "StepVideoFlowScheduler": ".step_video",
    "T2VTurboLCMScheduler": ".t2v_turbo",
    "WanFlowMatchEulerScheduler": ".wan",
    "WanFlowUniPCScheduler": ".wan",
    "build_cosmos3_flow_unipc_scheduler": ".cosmos3",
    "build_hunyuan_video_flow_match_scheduler": ".hunyuan_video",
    "build_karras_x0_ab2_scheduler": ".karras_x0",
    "build_karras_x0_euler_scheduler": ".karras_x0",
    "build_ltx_fixed_euler_scheduler": ".ltx",
    "build_sana_flow_match_scheduler": ".sana",
    "build_sana_flow_dpm_scheduler": ".sana",
    "build_sana_scm_scheduler": ".sana",
    "build_sana_streaming_euler_scheduler": ".sana",
    "build_step_video_flow_scheduler": ".step_video",
    "build_t2v_turbo_lcm_scheduler": ".t2v_turbo",
    "build_wan_flow_match_euler_scheduler": ".wan",
    "build_wan_flow_unipc_scheduler": ".wan",
    "build_wan_sigmas": ".wan",
    "get_sampling_sigmas": ".flow_dpm",
    "retrieve_timesteps": ".flow_dpm",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
