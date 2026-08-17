"""Native training objectives.

The facade resolves symbols lazily (mirroring ``training/__init__`` and
``training/models/__init__``) so importing torch-free flow-matching math such
as ``flow_shift_sigmas`` does not pull in ``classic_diffusion``'s top-level
torch dependency.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ClassicDiffusionConfig": ".classic_diffusion",
    "ClassicDiffusionObjective": ".classic_diffusion",
    "DiffusionRegressionLoss": ".classic_diffusion",
    "dynamic_latent_scale": ".classic_diffusion",
    "lvdm_linear_beta_schedule": ".classic_diffusion",
    "lvdm_short_objective": ".classic_diffusion",
    "rescale_betas_to_zero_terminal_snr": ".classic_diffusion",
    "FlowMatchingConfig": ".flow_matching",
    "FlowMatchingLoss": ".flow_matching",
    "FlowMatchingObjective": ".flow_matching",
    "flow_clean_from_velocity": ".flow_matching",
    "flow_interpolate": ".flow_matching",
    "flow_matching_denominator": ".flow_matching",
    "flow_matching_mse": ".flow_matching",
    "flow_noise_from_velocity": ".flow_matching",
    "flow_shift_sigmas": ".flow_matching",
    "flow_velocity_target": ".flow_matching",
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
