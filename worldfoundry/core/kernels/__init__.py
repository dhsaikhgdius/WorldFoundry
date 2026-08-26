"""In-tree accelerator kernels with portable PyTorch fallbacks.

Heavy submodules (diffusion / moe / capabilities) import torch. Keep this
package ``__init__`` lazy so optional native-kernel inspection can import
``worldfoundry.core.kernels.native_provider`` on CPU hosts without pulling
torch at package import time.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "clear_kernel_dispatch_cache",
    "group_norm_silu",
    "hidden_qk_rmsnorm_rope_3d",
    "KernelDeviceProfile",
    "kernel_device_profile",
    "kernel_dispatch_report",
    "layer_norm_scale_shift",
    "native_provider",
    "native_provider_status",
    "NativeProviderStatus",
    "NativeProviderUnavailable",
    "qk_rmsnorm_rope",
    "residual_gate_add",
    "rms_norm_scale_shift",
    "routed_swiglu_moe_pytorch",
    "routed_swiglu_moe",
    "routed_swiglu_moe_triton",
    "silu_and_mul",
    "silu_mul",
]

_DIFFUSION_EXPORTS = frozenset(
    {
        "clear_kernel_dispatch_cache",
        "group_norm_silu",
        "hidden_qk_rmsnorm_rope_3d",
        "kernel_dispatch_report",
        "layer_norm_scale_shift",
        "qk_rmsnorm_rope",
        "residual_gate_add",
        "rms_norm_scale_shift",
        "silu_and_mul",
        "silu_mul",
    }
)
_CAPABILITY_EXPORTS = frozenset({"KernelDeviceProfile", "kernel_device_profile"})
_MOE_EXPORTS = frozenset({"routed_swiglu_moe", "routed_swiglu_moe_pytorch"})
_NATIVE_EXPORTS = frozenset(
    {
        "native_provider",
        "native_provider_status",
        "NativeProviderStatus",
        "NativeProviderUnavailable",
    }
)


def __getattr__(name: str) -> Any:
    import importlib

    if name in _CAPABILITY_EXPORTS:
        module = importlib.import_module("worldfoundry.core.kernels.capabilities")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _DIFFUSION_EXPORTS:
        module = importlib.import_module("worldfoundry.core.kernels.diffusion")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _MOE_EXPORTS:
        module = importlib.import_module("worldfoundry.core.kernels.moe")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "routed_swiglu_moe_triton":
        module = importlib.import_module("worldfoundry.core.kernels.triton_moe")
        value = module.routed_swiglu_moe_triton
        globals()[name] = value
        return value
    if name == "native_provider":
        module = importlib.import_module("worldfoundry.core.kernels.native_provider")
        globals()[name] = module
        return module
    if name in _NATIVE_EXPORTS:
        module = importlib.import_module("worldfoundry.core.kernels.native_provider")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
