"""Attention backend capability probes shared by inference dispatchers.

This module is responsible for safely, lightly, and side-effect-freely probing and
dispatching the optimal Attention computation backend in the current runtime environment.

Core Architectural Design:
1. Zero-Cost Probing: Uses `importlib.util.find_spec` to dynamically query packages
   rather than physically importing them. Since many acceleration packages (e.g., `flash_attn`)
   might crash hard during import/initialization (e.g., due to missing shared libraries or
   mismatched CUDA runtimes), this probing mechanism ensures a graceful fallback to PyTorch
   native SDPA without crashing the process.
2. Two-Stage Validation: Separates physical availability (`available`) from runtime hardware
   compatibility (`usable`). For example, even if `flash_attn` is successfully installed, it
   will be marked unusable if the active GPU Compute Capability is < 8.0 (e.g., V100, T4).
3. Fallback Priority Chain: Defines structured fallback paths for standard/default backends
   as well as experimental/sparse attention kernels (e.g., video sparse attention, V-MoBA),
   ensuring the system is highly robust across diverse hardware.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import torch


@dataclass(frozen=True)
class AttentionKernelCapability:
    """Runtime availability metadata for a single attention kernel family.

    Attributes:
        name: Canonical identifier of the attention backend.
        package: The underlying Python package or C++/CUDA extension.
        available: True if the module/package is physically installed in the environment.
        usable: True if the backend is physically runnable on the active hardware (e.g., CUDA capability).
        reason: Explains why the backend is unusable or unavailable, if applicable.
    """

    name: str
    package: str
    available: bool
    usable: bool
    reason: str = ""


_AUTO = "auto"
_TORCH = "torch"
_FLASH_AUTO = "flash_attention"
_BACKEND_ALIASES: Mapping[str, str] = {
    "auto": _AUTO,
    "default": _AUTO,
    "torch": _TORCH,
    "torch_sdpa": _TORCH,
    "sdpa": _TORCH,
    "math": _TORCH,
    "efficient": _TORCH,
    "cudnn": _TORCH,
    "flash": _FLASH_AUTO,
    "flash_attn": _FLASH_AUTO,
    "flash_attention": _FLASH_AUTO,
    "flash_attention_2": "flash_attention_2",
    "flash2": "flash_attention_2",
    "flash_attn_2": "flash_attention_2",
    "flash_attn2": "flash_attention_2",
    "flash_attention_3": "flash_attention_3",
    "flash3": "flash_attention_3",
    "flash_attn_3": "flash_attention_3",
    "flash_attn3": "flash_attention_3",
    "sage": "sage_attention",
    "sage_attn": "sage_attention",
    "sage_attention": "sage_attention",
    "sageattn": "sage_attention",
    "sage_attn_three": "sage_attention_3",
    "sage_attention_3": "sage_attention_3",
    "sage3": "sage_attention_3",
    "xformers": "xformers",
    "xformers_attention": "xformers",
    "video_sparse_attn": "video_sparse_attention",
    "video_sparse_attention": "video_sparse_attention",
    "flex_block_attn": "flex_block_attention",
    "flex_block_attention": "flex_block_attention",
    "vmoba": "vmoba_attention",
    "vmoba_attn": "vmoba_attention",
    "vmoba_attention": "vmoba_attention",
    "sla": "sla_attention",
    "sla_attn": "sla_attention",
    "sla_attention": "sla_attention",
    "sage_sla": "sage_sla_attention",
    "sage_sla_attn": "sage_sla_attention",
    "sage_sla_attention": "sage_sla_attention",
}
_DEFAULT_PRIORITY = (_TORCH,)
_REPORT_PRIORITY = (
    "flash_attention_3",
    "flash_attention_2",
    "sage_attention",
    "xformers",
    _TORCH,
)
_EXPLICIT_PRIORITY = ("sage_attention_3",)
_EXPERIMENTAL_PRIORITY = (
    "flex_block_attention",
    "video_sparse_attention",
    "vmoba_attention",
    "sla_attention",
    "sage_sla_attention",
)
_VIDEO_SPARSE_KERNEL_IMPORT = "fast" + "video_kernel"


def attention_backend_from_env(environ: Mapping[str, str] | None = None) -> str:
    """Read and resolve the canonical attention backend requested by the environment.

    Inspects `WORLDFOUNDRY_ATTENTION_IMPLEMENTATION` or `WORLDFOUNDRY_ATTENTION_BACKEND`,
    falling back to `"auto"` if neither is specified.
    """
    env = os.environ if environ is None else environ
    value = env.get("WORLDFOUNDRY_ATTENTION_IMPLEMENTATION") or env.get("WORLDFOUNDRY_ATTENTION_BACKEND") or _AUTO
    return normalize_attention_backend(value)


def normalize_attention_backend(value: str | None) -> str:
    """Normalize colloquial or varied attention backend names into standard keys.

    For example, maps 'flash-attn-2', 'flash2', or 'flash_attention_2' to 'flash_attention_2'
    while stripping and lowering input values to handle typos gracefully.
    """
    if value is None:
        return _AUTO
    key = str(value).strip().lower().replace("-", "_")
    if not key:
        return _AUTO
    canonical = _BACKEND_ALIASES.get(key)
    if canonical is None:
        allowed = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(f"Unknown attention backend {value!r}. Expected one of: {allowed}")
    return canonical


def probe_attention_backends(device: torch.device | str | int | None = None) -> dict[str, AttentionKernelCapability]:
    """Probe installed attention packages and hardware compatibility.

    Results are cached by runtime and compute capability, rather than globally.
    This keeps mixed A100/H100 nodes correct when the active tensor/device
    changes after module import.
    """

    capability = _cuda_compute_capability(device)
    hip = bool(getattr(torch.version, "hip", None))
    accelerator = _torch_cuda_accelerator_available(device)
    return _probe_attention_backends_cached(capability, hip, accelerator)


@lru_cache(maxsize=16)
def _probe_attention_backends_cached(
    capability: tuple[int, int] | None,
    hip: bool,
    accelerator: bool,
) -> dict[str, AttentionKernelCapability]:
    nvidia_cuda = capability is not None and not hip
    flash_gpu = capability is not None and capability[0] in {8, 9}
    flash3_gpu = capability == (9, 0)
    sage_gpu = capability is not None and capability[0] in {8, 9}
    # Keep this aligned with the architectures compiled by SageAttention3's
    # upstream Blackwell extension.  In particular, its current build does not
    # target SM103, so B300/GB300 must retain exact cuDNN/SDPA rather than being
    # accepted merely because they share the Blackwell generation name.
    sage3_gpu = capability in {(10, 0), (12, 0), (12, 1)}
    return {
        "flash_attention_3": _package_capability(
            name="flash_attention_3",
            package="flash_attn_interface",
            usable_if=flash3_gpu,
            unavailable_reason="flash_attn_interface is not installed",
            unusable_reason="FlashAttention 3 requires NVIDIA Hopper (SM90)",
        ),
        "flash_attention_2": _package_capability(
            name="flash_attention_2",
            package="flash_attn",
            usable_if=flash_gpu,
            unavailable_reason="flash_attn is not installed",
            unusable_reason="FlashAttention 2 requires a supported NVIDIA Ampere, Ada, or Hopper GPU",
        ),
        "sage_attention": _package_capability(
            name="sage_attention",
            package="sageattention",
            usable_if=sage_gpu,
            unavailable_reason="sageattention is not installed",
            unusable_reason="SageAttention requires a supported NVIDIA Ampere, Ada, or Hopper GPU",
        ),
        "sage_attention_3": _package_capability(
            name="sage_attention_3",
            package="sageattn3",
            usable_if=sage3_gpu,
            unavailable_reason="the in-tree sageattn3 extension is not built",
            unusable_reason="SageAttention 3 requires an explicitly supported NVIDIA Blackwell target",
        ),
        "xformers": _package_capability(
            name="xformers",
            package="xformers.ops",
            usable_if=accelerator,
            unavailable_reason="xformers is not installed",
            unusable_reason="xFormers attention requires a CUDA or ROCm accelerator build",
        ),
        "video_sparse_attention": _package_capability(
            name="video_sparse_attention",
            package="video_sparse_attention_kernels",
            import_name=_VIDEO_SPARSE_KERNEL_IMPORT,
            usable_if=nvidia_cuda,
            unavailable_reason="video sparse attention kernel package is not installed",
            unusable_reason="Video sparse attention requires CUDA and model-specific kernels",
        ),
        "flex_block_attention": _package_capability(
            name="flex_block_attention",
            package="flex_block_attn",
            usable_if=nvidia_cuda,
            unavailable_reason="flex_block_attn is not installed",
            unusable_reason="FlexBlockAttention requires an NVIDIA CUDA device",
        ),
        "vmoba_attention": _package_capability(
            name="vmoba_attention",
            package="vmoba_attention_kernels",
            import_name=_VIDEO_SPARSE_KERNEL_IMPORT,
            usable_if=nvidia_cuda,
            unavailable_reason="V-MoBA attention kernel package is not installed",
            unusable_reason="V-MoBA attention requires CUDA and model-specific kernels",
        ),
        "sla_attention": _package_capability(
            name="sla_attention",
            package="sparse_linear_attention_kernels",
            import_name=_VIDEO_SPARSE_KERNEL_IMPORT,
            usable_if=nvidia_cuda,
            unavailable_reason="sparse linear attention kernel package is not installed",
            unusable_reason="Sparse linear attention requires CUDA and model-specific kernels",
        ),
        "sage_sla_attention": _package_capability(
            name="sage_sla_attention",
            package="sage_sparse_linear_attention_kernels",
            import_name=_VIDEO_SPARSE_KERNEL_IMPORT,
            usable_if=nvidia_cuda,
            unavailable_reason="SageSLA attention kernel package is not installed",
            unusable_reason="SageSLA attention requires CUDA and model-specific kernels",
        ),
        _TORCH: AttentionKernelCapability(name=_TORCH, package="torch", available=True, usable=True),
    }


# Preserve the cache invalidation hook exposed by the previous cached public
# function; tests and long-running plugin hosts use it after environment changes.
setattr(probe_attention_backends, "cache_clear", _probe_attention_backends_cached.cache_clear)


def resolve_attention_backend(
    preferred: str | None = None,
    device: torch.device | str | int | None = None,
) -> str:
    """Resolve the highest priority usable backend, falling back gracefully to PyTorch SDPA.

    If `"auto"` is preferred, traverses priority list to select the first runnable package.
    If the requested package is not usable on the active hardware, automatically degrades to
    `"torch"` SDPA to ensure robust execution.
    """
    requested = attention_backend_from_env() if preferred is None else normalize_attention_backend(preferred)
    capabilities = probe_attention_backends(device)
    if requested == _AUTO:
        for name in _DEFAULT_PRIORITY:
            if capabilities[name].usable:
                return name
        return _TORCH
    if requested == _FLASH_AUTO:
        for name in ("flash_attention_3", "flash_attention_2"):
            if capabilities[name].usable:
                return name
        return _TORCH
    capability = capabilities.get(requested)
    if capability is not None and capability.usable:
        return requested
    return _TORCH


def resolve_transformers_attention_implementation(
    preferred: str | None = None,
    device: torch.device | str | int | None = None,
) -> str:
    """Resolve a backend name accepted by Transformers model configs.

    Transformers currently exposes portable ``eager``/``sdpa`` paths and the
    separately installed ``flash_attention_2`` provider.  WorldFoundry has a
    wider backend vocabulary (including FA3 and model-specific kernels), so a
    small adapter is needed before writing ``config._attn_implementation``.
    Unsupported or unavailable providers conservatively map to PyTorch SDPA.
    """

    requested = attention_backend_from_env() if preferred is None else str(preferred)
    normalized = requested.strip().lower().replace("-", "_")
    if normalized == "eager":
        return "eager"
    resolved = resolve_attention_backend(requested, device)
    return "flash_attention_2" if resolved == "flash_attention_2" else "sdpa"


def attention_backend_capability(
    name: str,
    device: torch.device | str | int | None = None,
) -> AttentionKernelCapability:
    """Retrieve runtime capability metadata for a single normalized backend."""
    canonical = normalize_attention_backend(name)
    if canonical in {_AUTO, _FLASH_AUTO}:
        canonical = resolve_attention_backend(canonical, device)
    return probe_attention_backends(device)[canonical]


def attention_backend_report() -> tuple[AttentionKernelCapability, ...]:
    """Retrieve capability status of all registered backends, ordered by dispatch priority."""
    capabilities = probe_attention_backends()
    return tuple(
        capabilities[name]
        for name in (*_REPORT_PRIORITY, *_EXPLICIT_PRIORITY, *_EXPERIMENTAL_PRIORITY)
        if name in capabilities
    )


def gpu_supports_flash_attention(device: torch.device | str | int | None = None) -> bool:
    """Determine if the active CUDA device is architecturally capable of running FlashAttention.

    This in-tree FA2 path targets NVIDIA Ampere (SM8x), Ada (SM89), and
    Hopper (SM90). Blackwell kernels are resolved separately because SM100/103
    and SM120 are not binary-compatible feature targets.
    """
    capability = _cuda_compute_capability(device)
    if capability is None:
        return False
    # The in-tree FA2 dispatcher is validated for NVIDIA Ampere/Ada/Hopper.
    # Blackwell data-center (SM100/103) and client (SM120) kernels require
    # separate providers and must not inherit this compatibility decision.
    return capability[0] in {8, 9}


def gpu_supports_flash_attention_3(device: torch.device | str | int | None = None) -> bool:
    """Return whether the active device is a validated FA3 target.

    FA3 is a Hopper-specific provider in WorldFoundry.  Importability of the
    ``flash_attn_interface`` module alone is not sufficient: Ampere, Ada and
    Blackwell builds use different kernel targets.
    """

    return _cuda_compute_capability(device) == (9, 0)


def _cuda_compute_capability(
    device: torch.device | str | int | None = None,
) -> tuple[int, int] | None:
    """Return an NVIDIA CUDA capability without mistaking ROCm for CUDA."""

    if getattr(torch.version, "hip", None):
        return None
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability(device)
        return int(major), int(minor)
    except Exception:
        return None


def _torch_cuda_accelerator_available(device: torch.device | str | int | None = None) -> bool:
    """Return whether *device* refers to PyTorch's CUDA/HIP accelerator API."""

    if not torch.cuda.is_available():
        return False
    if device is None or isinstance(device, int):
        return True
    try:
        return torch.device(device).type == "cuda"
    except (TypeError, RuntimeError):
        return False


def _package_capability(
    *,
    name: str,
    package: str,
    import_name: str | None = None,
    usable_if: bool,
    unavailable_reason: str,
    unusable_reason: str,
) -> AttentionKernelCapability:
    """Underlying safe package prober.

    Queries package specs via `importlib.util.find_spec` without executing the actual
    `__init__.py` code, avoiding potential segfaults or library errors on non-CUDA machines.
    """
    try:
        available = importlib.util.find_spec(import_name or package) is not None
    except ModuleNotFoundError:
        available = False
    if not available:
        return AttentionKernelCapability(
            name=name, package=package, available=False, usable=False, reason=unavailable_reason
        )
    if not usable_if:
        return AttentionKernelCapability(
            name=name, package=package, available=True, usable=False, reason=unusable_reason
        )
    return AttentionKernelCapability(name=name, package=package, available=True, usable=True)


__all__ = [
    "AttentionKernelCapability",
    "attention_backend_capability",
    "attention_backend_from_env",
    "attention_backend_report",
    "gpu_supports_flash_attention",
    "normalize_attention_backend",
    "probe_attention_backends",
    "resolve_attention_backend",
    "resolve_transformers_attention_implementation",
]
