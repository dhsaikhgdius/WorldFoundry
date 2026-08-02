"""Public diffusion operators with accelerator selection and PyTorch fallback."""

from __future__ import annotations

import os
from functools import lru_cache

import torch
import torch.nn.functional as F

from worldfoundry.core.kernels.capabilities import (
    QUALIFIED_TRITON_CAPABILITIES,
    default_kernel_thresholds,
    detected_kernel_device_profiles,
    kernel_device_profile,
    triton_tensor_eligible,
)
from worldfoundry.core.kernels.registry import (
    KERNEL_REGISTRY,
    clear_kernel_dispatch_cache,
    kernel_autotune_enabled,
)
from worldfoundry.core.kernels.registry import (
    kernel_dispatch_report as _registry_dispatch_report,
)

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# Bound each double-precision RoPE activation slab in the portable fallback.
# A LingBot full sequence contains more than 316 million hidden elements; an
# unchunked ``torch.stack`` for one rotated Q/K tensor alone is 2.36 GiB.  The
# fallback keeps the reference float64 arithmetic, but limits the selected
# activation slab to roughly 128 MiB before intermediate expressions.
_HIDDEN_ROPE_FALLBACK_CHUNK_ELEMENTS = 16 * 1024 * 1024


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(int(os.getenv(name, str(default)) or default), 0)
    except ValueError:
        return default


def _positive_override(value: str | None, default: int) -> int:
    try:
        return max(int(value if value is not None else default), 0)
    except ValueError:
        return default


# Below the device-profile thresholds, eager PyTorch's vendor/pointwise path is
# usually faster than paying custom dispatch and launch costs. The thresholds
# are architecture-specific conservative defaults; throughput mode measures
# exact device/dtype/shape signatures and retains the faster implementation.
_TORCH_BACKENDS = {"torch", "pytorch", "native", "off", "disabled"}


def _requested_kernel_backend() -> str:
    return os.getenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto").strip().casefold() or "auto"


def kernel_dispatch_report() -> dict[str, object]:
    """Return kernel candidates, quarantine state, and size thresholds."""

    report = _registry_dispatch_report()
    report["devices"] = [profile.to_dict() for profile in detected_kernel_device_profiles()]
    report["threshold_overrides"] = {
        "residual_gate_min_elements": os.getenv("WORLDFOUNDRY_RESIDUAL_GATE_MIN_ELEMENTS"),
        "adaln_min_elements": os.getenv("WORLDFOUNDRY_ADALN_MIN_ELEMENTS"),
        "gated_activation_min_elements": os.getenv("WORLDFOUNDRY_GATED_ACTIVATION_MIN_ELEMENTS"),
        "group_norm_silu_min_elements": os.getenv("WORLDFOUNDRY_GROUP_NORM_SILU_MIN_ELEMENTS"),
    }
    return report


def _kernel_threshold(tensor: torch.Tensor, op: str, *related: torch.Tensor) -> int:
    compiling = torch.compiler.is_compiling()
    if not compiling:
        override_name = {
            "residual_gate_add": "WORLDFOUNDRY_RESIDUAL_GATE_MIN_ELEMENTS",
            "layer_norm_scale_shift": "WORLDFOUNDRY_ADALN_MIN_ELEMENTS",
            "silu_mul": "WORLDFOUNDRY_GATED_ACTIVATION_MIN_ELEMENTS",
            "group_norm_silu": "WORLDFOUNDRY_GROUP_NORM_SILU_MIN_ELEMENTS",
        }.get(op)
        if override_name is None:
            raise KeyError(f"unknown kernel threshold operator: {op}")
        return _eager_kernel_threshold_cached(
            str(tensor.device),
            tensor.dtype,
            op,
            tuple(item.dtype for item in related),
            os.getenv(override_name),
        )

    if compiling:
        capability = torch.cuda.get_device_capability(tensor.device) if tensor.is_cuda else None
        residual_default, adaln_default = default_kernel_thresholds(capability)
    else:  # pragma: no cover - eager execution returns through the cache above.
        raise AssertionError("unreachable")
    if op == "residual_gate_add":
        if capability == (8, 0):
            dtypes = (tensor.dtype, *(item.dtype for item in related))
            promoted = dtypes[0]
            for dtype in dtypes[1:]:
                promoted = torch.promote_types(promoted, dtype)
            if all(dtype == torch.float16 for dtype in dtypes):
                residual_default = 64 * 1024
            elif promoted == torch.float32:
                residual_default = 4 * 1024 * 1024
            else:
                residual_default = 8 * 1024 * 1024
        return _positive_env_int(
            "WORLDFOUNDRY_RESIDUAL_GATE_MIN_ELEMENTS",
            residual_default,
        )
    if op == "layer_norm_scale_shift":
        return _positive_env_int("WORLDFOUNDRY_ADALN_MIN_ELEMENTS", adaln_default)
    if op == "silu_mul":
        mib = 1024 * 1024
        if capability == (8, 0) and tensor.dtype == torch.float32:
            activation_default = 4 * mib
        elif capability is not None and capability[0] == 8:
            activation_default = 8 * mib
        elif capability is not None and capability[0] in {10, 11, 12}:
            activation_default = 32 * mib
        else:
            activation_default = 16 * mib
        return _positive_env_int("WORLDFOUNDRY_GATED_ACTIVATION_MIN_ELEMENTS", activation_default)
    if op == "group_norm_silu":
        mib = 1024 * 1024
        if capability is not None and capability[0] == 8:
            group_norm_default = mib
        elif capability is not None and capability[0] in {10, 11, 12}:
            group_norm_default = 4 * mib
        else:
            group_norm_default = 2 * mib
        return _positive_env_int("WORLDFOUNDRY_GROUP_NORM_SILU_MIN_ELEMENTS", group_norm_default)
    raise KeyError(f"unknown kernel threshold operator: {op}")


@lru_cache(maxsize=256)
def _eager_kernel_threshold_cached(
    device: str,
    dtype: torch.dtype,
    op: str,
    related_dtypes: tuple[torch.dtype, ...],
    override: str | None,
) -> int:
    profile = kernel_device_profile(device)
    capability = profile.compute_capability
    if op == "residual_gate_add":
        default = profile.residual_gate_min_elements
        if capability == (8, 0):
            dtypes = (dtype, *related_dtypes)
            promoted = dtypes[0]
            for related_dtype in dtypes[1:]:
                promoted = torch.promote_types(promoted, related_dtype)
            if all(item == torch.float16 for item in dtypes):
                default = 64 * 1024
            elif promoted == torch.float32:
                default = 4 * 1024 * 1024
            else:
                default = 8 * 1024 * 1024
        return _positive_override(override, default)
    if op == "layer_norm_scale_shift":
        return _positive_override(override, profile.adaln_min_elements)
    if op == "silu_mul":
        mib = 1024 * 1024
        if capability == (8, 0) and dtype == torch.float32:
            default = 4 * mib
        elif capability is not None and capability[0] == 8:
            default = 8 * mib
        elif capability is not None and capability[0] in {10, 11, 12}:
            default = 32 * mib
        else:
            default = 16 * mib
        return _positive_override(override, default)
    if op == "group_norm_silu":
        mib = 1024 * 1024
        if capability is not None and capability[0] == 8:
            default = mib
        elif capability is not None and capability[0] in {10, 11, 12}:
            default = 4 * mib
        else:
            default = 2 * mib
        return _positive_override(override, default)
    raise KeyError(f"unknown kernel threshold operator: {op}")


def _triton_device_supported(tensor: torch.Tensor) -> bool:
    if not torch.compiler.is_compiling():
        eligible = triton_tensor_eligible(tensor)
        if eligible:
            _configure_triton_cache_once()
        return eligible
    if not tensor.is_cuda or bool(getattr(torch.version, "hip", None)):
        return False
    capability = torch.cuda.get_device_capability(tensor.device)
    allow_untested = os.getenv("WORLDFOUNDRY_ALLOW_UNTESTED_GPU_KERNELS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if capability not in QUALIFIED_TRITON_CAPABILITIES and not allow_untested:
        return False
    return tensor.dtype != torch.bfloat16 or capability >= (8, 0)


@lru_cache(maxsize=1)
def _configure_triton_cache_once() -> None:
    """Configure the persistent cache before the first eager Triton import."""

    try:
        from worldfoundry.runtime.compile_cache import configure_persistent_compile_cache

        configure_persistent_compile_cache(namespace="in-tree-kernels")
    except (ImportError, OSError):
        # Cache setup must never make the exact PyTorch fallback unavailable.
        return


def _inference_only(*tensors: torch.Tensor) -> bool:
    return not (torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors))


def _same_cuda_device(*tensors: torch.Tensor) -> bool:
    return len(tensors) > 0 and all(tensor.is_cuda and tensor.device == tensors[0].device for tensor in tensors)


def _broadcastable_to(value: torch.Tensor, target: torch.Tensor) -> bool:
    try:
        return torch.broadcast_shapes(value.shape, target.shape) == target.shape
    except RuntimeError:
        return False


def _regular_contiguous(tensor: torch.Tensor) -> bool:
    return tensor.is_contiguous() and tensor.stride(-1) == 1


def _workload_signature(*tensors: torch.Tensor, extra: tuple[object, ...] = ()) -> tuple[object, ...]:
    first = tensors[0]
    capability: tuple[int, int] | str = "cpu"
    if first.is_cuda:
        try:
            capability = torch.cuda.get_device_capability(first.device)
        except (AssertionError, RuntimeError):
            capability = "unknown"
    return (
        str(first.device),
        capability,
        bool(torch.is_grad_enabled()),
        os.getenv("WORLDFOUNDRY_ALLOW_UNTESTED_GPU_KERNELS", "").strip().casefold(),
        *(
            (
                str(tensor.device),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
                bool(tensor.requires_grad),
            )
            for tensor in tensors
        ),
        *extra,
    )


def _silu_mul_torch(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * value


def _eligible_silu_mul(gate: torch.Tensor, value: torch.Tensor) -> bool:
    return (
        _inference_only(gate, value)
        and _same_cuda_device(gate, value)
        and all(_triton_device_supported(tensor) for tensor in (gate, value))
        and gate.dtype in _FLOAT_DTYPES
        and value.dtype == gate.dtype
        and gate.shape == value.shape
        and gate.numel() > 0
        and _regular_contiguous(gate)
        and _regular_contiguous(value)
    )


def _silu_mul_triton(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    from worldfoundry.core.kernels.triton_diffusion import silu_mul as implementation

    return implementation(gate, value)


def silu_mul(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Return ``silu(gate) * value`` with an eager inference fast path."""

    requested = _requested_kernel_backend()
    if (
        torch.compiler.is_compiling()
        or requested in _TORCH_BACKENDS
        or (
            requested == "auto" and not kernel_autotune_enabled() and gate.numel() < _kernel_threshold(gate, "silu_mul")
        )
    ):
        return _silu_mul_torch(gate, value)
    return KERNEL_REGISTRY.dispatch(
        "silu_mul",
        _silu_mul_torch,
        gate,
        value,
        signature=_workload_signature(gate, value),
    )


def _silu_and_mul_torch(input: torch.Tensor) -> torch.Tensor:
    gate, value = input.chunk(2, dim=-1)
    return F.silu(gate) * value


def _eligible_silu_and_mul(input: torch.Tensor) -> bool:
    return (
        _inference_only(input)
        and _triton_device_supported(input)
        and input.dtype in _FLOAT_DTYPES
        and input.ndim >= 1
        and input.numel() > 0
        and input.shape[-1] > 0
        and input.shape[-1] % 2 == 0
        and _regular_contiguous(input)
    )


def _silu_and_mul_triton(input: torch.Tensor) -> torch.Tensor:
    from worldfoundry.core.kernels.triton_diffusion import silu_and_mul as implementation

    return implementation(input)


def silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """Split the last dimension and return ``silu(first) * second``."""

    requested = _requested_kernel_backend()
    output_elements = input.numel() // 2
    if (
        torch.compiler.is_compiling()
        or requested in _TORCH_BACKENDS
        or (
            requested == "auto"
            and not kernel_autotune_enabled()
            and output_elements < _kernel_threshold(input, "silu_mul")
        )
    ):
        return _silu_and_mul_torch(input)
    return KERNEL_REGISTRY.dispatch(
        "silu_and_mul",
        _silu_and_mul_torch,
        input,
        signature=_workload_signature(input),
    )


def _group_norm_silu_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    return F.silu(F.group_norm(input, int(num_groups), weight=weight, bias=bias, eps=float(eps)))


def _eligible_group_norm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> bool:
    del eps
    return (
        _inference_only(input, weight, bias)
        and _same_cuda_device(input, weight, bias)
        and all(_triton_device_supported(tensor) for tensor in (input, weight, bias))
        and input.dtype in _FLOAT_DTYPES
        and weight.dtype == input.dtype
        and bias.dtype == input.dtype
        and 2 <= input.ndim <= 5
        and input.numel() > 0
        and int(num_groups) > 0
        and input.shape[1] % int(num_groups) == 0
        and input.is_contiguous()
        and weight.is_contiguous()
        and bias.is_contiguous()
        and weight.shape == bias.shape == (input.shape[1],)
    )


def _group_norm_silu_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    from worldfoundry.core.kernels.triton_group_norm_silu import group_norm_silu as implementation

    return implementation(input, weight, bias, int(num_groups), float(eps))


def group_norm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fuse affine GroupNorm and SiLU for contiguous NCHW/NCDHW tensors.

    The eager accelerator is adapted from SGLang's Apache-2.0 Triton kernel;
    training, CPU, unsupported layouts, and compiled graphs use exact PyTorch.
    """

    args = (input, weight, bias, int(num_groups), float(eps))
    requested = _requested_kernel_backend()
    if (
        torch.compiler.is_compiling()
        or requested in _TORCH_BACKENDS
        or (
            requested == "auto"
            and not kernel_autotune_enabled()
            and input.numel() < _kernel_threshold(input, "group_norm_silu")
        )
    ):
        return _group_norm_silu_torch(*args)
    return KERNEL_REGISTRY.dispatch(
        "group_norm_silu",
        _group_norm_silu_torch,
        *args,
        signature=_workload_signature(input, weight, bias, extra=(int(num_groups), float(eps))),
    )


def _residual_gate_torch(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return residual + update * gate


def _eligible_residual_gate(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> bool:
    return (
        _inference_only(residual, update, gate)
        and _same_cuda_device(residual, update, gate)
        and all(_triton_device_supported(tensor) for tensor in (residual, update, gate))
        and residual.dtype in _FLOAT_DTYPES
        and update.dtype in _FLOAT_DTYPES
        and gate.dtype in _FLOAT_DTYPES
        and 2 <= residual.ndim <= 5
        and residual.numel() > 0
        and residual.shape == update.shape
        and 0 < residual.shape[-1] <= 8192
        and _regular_contiguous(residual)
        and _regular_contiguous(update)
        and _broadcastable_to(gate, residual)
    )


def _residual_gate_triton(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    from worldfoundry.core.kernels.triton_diffusion import residual_gate as implementation

    return implementation(residual, update, gate)


def residual_gate_add(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Return ``residual + update * gate`` with broadcast-aware fusion."""

    # Inductor already emits one pointwise kernel for this expression and can
    # fuse it with adjacent compiled work. Keep the explicit Triton kernel for
    # eager execution, where PyTorch otherwise launches multiply and add
    # separately. An explicit backend override remains available for profiling.
    requested = _requested_kernel_backend()
    if (
        torch.compiler.is_compiling()
        or requested in _TORCH_BACKENDS
        or (
            requested == "auto"
            and not kernel_autotune_enabled()
            and (residual.numel() < _kernel_threshold(residual, "residual_gate_add", update, gate))
        )
    ):
        return _residual_gate_torch(residual, update, gate)
    return KERNEL_REGISTRY.dispatch(
        "residual_gate_add",
        _residual_gate_torch,
        residual,
        update,
        gate,
        signature=_workload_signature(residual, update, gate),
    )


def _layer_norm_scale_shift_torch(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    upcast: bool,
) -> torch.Tensor:
    normalized_input = x.float() if upcast else x
    normalized = F.layer_norm(normalized_input, (x.shape[-1],), weight=None, bias=None, eps=eps)
    if upcast:
        normalized = normalized.to(x.dtype)
    return normalized * (1 + scale) + shift


def _eligible_layer_norm_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    upcast: bool,
) -> bool:
    del eps, upcast
    return (
        _inference_only(x, scale, shift)
        and _same_cuda_device(x, scale, shift)
        and all(_triton_device_supported(tensor) for tensor in (x, scale, shift))
        and x.dtype in _FLOAT_DTYPES
        and scale.dtype in _FLOAT_DTYPES
        and shift.dtype in _FLOAT_DTYPES
        and 2 <= x.ndim <= 5
        and x.numel() > 0
        and 0 < x.shape[-1] <= 8192
        and _regular_contiguous(x)
        and _broadcastable_to(scale, x)
        and _broadcastable_to(shift, x)
    )


def _layer_norm_scale_shift_triton(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    upcast: bool,
) -> torch.Tensor:
    del upcast
    from worldfoundry.core.kernels.triton_diffusion import layer_norm_scale_shift as implementation

    return implementation(x, scale, shift, eps)


def layer_norm_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
    *,
    upcast: bool = False,
) -> torch.Tensor:
    """Fuse affine-free LayerNorm with AdaLN scale and shift."""

    requested = _requested_kernel_backend()
    if (
        torch.compiler.is_compiling()
        or requested in _TORCH_BACKENDS
        or (
            requested == "auto"
            and not kernel_autotune_enabled()
            and x.numel() < _kernel_threshold(x, "layer_norm_scale_shift")
        )
    ):
        return _layer_norm_scale_shift_torch(x, scale, shift, eps, upcast)
    return KERNEL_REGISTRY.dispatch(
        "layer_norm_scale_shift",
        _layer_norm_scale_shift_torch,
        x,
        scale,
        shift,
        eps,
        upcast,
        signature=_workload_signature(x, scale, shift, extra=(float(eps), bool(upcast))),
    )


def _rms_norm_scale_shift_torch(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps) * (1 + scale) + shift


def _eligible_rms_norm_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> bool:
    del eps
    tensors = (x, scale, shift) if weight is None else (x, weight, scale, shift)
    return (
        _inference_only(*tensors)
        and _same_cuda_device(*tensors)
        and all(_triton_device_supported(tensor) for tensor in tensors)
        and all(tensor.dtype in _FLOAT_DTYPES for tensor in tensors)
        and (weight is None or weight.dtype == x.dtype)
        and 2 <= x.ndim <= 5
        and x.numel() > 0
        and 0 < x.shape[-1] <= 8192
        and _regular_contiguous(x)
        and (weight is None or weight.shape == (x.shape[-1],))
        and _broadcastable_to(scale, x)
        and _broadcastable_to(shift, x)
    )


def _rms_norm_scale_shift_triton(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from worldfoundry.core.kernels.triton_diffusion import rms_norm_scale_shift as implementation

    return implementation(x, weight, scale, shift, eps)


def rms_norm_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
    *,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse RMSNorm with broadcast scale/shift modulation."""

    requested = _requested_kernel_backend()
    if (
        torch.compiler.is_compiling()
        or requested in _TORCH_BACKENDS
        or (
            requested == "auto"
            and not kernel_autotune_enabled()
            and x.numel() < _kernel_threshold(x, "layer_norm_scale_shift")
        )
    ):
        return _rms_norm_scale_shift_torch(x, weight, scale, shift, eps)
    tensors = (x, scale, shift) if weight is None else (x, weight, scale, shift)
    return KERNEL_REGISTRY.dispatch(
        "rms_norm_scale_shift",
        _rms_norm_scale_shift_torch,
        x,
        weight,
        scale,
        shift,
        eps,
        signature=_workload_signature(*tensors, extra=(float(eps), weight is not None)),
    )


def _qk_rmsnorm_rope_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
    rope_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Match the Wan/Gamma RMSNorm contract exactly: accumulate in fp32,
    # round the normalized activation back to its projection dtype, then
    # apply the learned weight. ``F.rms_norm`` has device-dependent mixed-
    # weight promotion behavior and is therefore not the semantic reference.
    def normalize(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        normed = value.float() * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + eps)
        return normed.to(value.dtype) * weight

    q = normalize(q, q_weight)
    k = normalize(k, k_weight)
    output_dtype = q.dtype
    if rope_fp32:
        q = q.float()
        k = k.float()
    half = q.shape[-1] // 2
    selected_freqs = freqs[: q.shape[1], 0, 0, :half]
    if selected_freqs.is_complex():
        cosine = selected_freqs.real.float()[None, :, None, :]
        sine = selected_freqs.imag.float()[None, :, None, :]
    else:
        angles = selected_freqs.float()
        cosine = angles.cos()[None, :, None, :]
        sine = angles.sin()[None, :, None, :]
    q_a, q_b = q[..., :half], q[..., half:]
    k_a, k_b = k[..., :half], k[..., half:]
    q = torch.cat((q_a * cosine - q_b * sine, q_b * cosine + q_a * sine), dim=-1)
    k = torch.cat((k_a * cosine - k_b * sine, k_b * cosine + k_a * sine), dim=-1)
    return q.to(output_dtype), k.to(output_dtype)


def _hidden_qk_rmsnorm_rope_3d_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    grid_size: tuple[int, int, int],
    eps: float,
    sequence_offset: int,
    start_frame: int,
    valid_tokens: int,
    head_start: int,
    head_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = q.shape[-1]
    if freqs.is_complex():
        cosine_table = freqs.real
        sine_table = freqs.imag
    else:
        # PyTorch has no complex-bfloat16 dtype, and ComplexHalf support is
        # intentionally incomplete. Promote only the frequency table; output
        # activations still return in their original dtype.
        real_freqs = freqs if freqs.dtype in {torch.float32, torch.float64} else freqs.float()
        cosine_table = real_freqs[..., 0]
        sine_table = real_freqs[..., 1]
    head_dim = hidden_size // num_heads

    def normalize(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        value_float = value.float()
        normed = value_float * torch.rsqrt(value_float.square().mean(dim=-1, keepdim=True) + eps)
        return normed.to(value.dtype) * weight

    q = normalize(q, q_weight)
    k = normalize(k, k_weight)
    pairs = head_dim // 2
    temporal_pairs = pairs - 2 * (pairs // 3)
    height_pairs = pairs // 3
    _, height, width = grid_size
    positions = torch.arange(q.shape[1], device=q.device, dtype=torch.long) + int(sequence_offset)
    plane = int(height) * int(width)
    temporal_position = int(start_frame) + positions // plane
    spatial_position = positions % plane
    height_position = spatial_position // int(width)
    width_position = spatial_position % int(width)
    pair_index = torch.arange(pairs, device=q.device, dtype=torch.long)
    rope_position = torch.where(
        pair_index[None] < temporal_pairs,
        temporal_position[:, None],
        torch.where(
            pair_index[None] < temporal_pairs + height_pairs,
            height_position[:, None],
            width_position[:, None],
        ),
    )
    frequency_columns = pair_index[None].expand_as(rope_position)
    selected_cosine = cosine_table[rope_position, frequency_columns]
    selected_sine = sine_table[rope_position, frequency_columns]
    valid = positions < int(valid_tokens)
    selected_cosine = torch.where(valid[:, None], selected_cosine, torch.ones_like(selected_cosine))
    selected_sine = torch.where(valid[:, None], selected_sine, torch.zeros_like(selected_sine))
    selected_cosine = selected_cosine[None, :, None, :].double()
    selected_sine = selected_sine[None, :, None, :].double()

    def rotate(value: torch.Tensor) -> torch.Tensor:
        output = value.clone()
        batch, sequence = value.shape[:2]
        selected_heads = int(head_end) - int(head_start)
        value_heads = value.view(batch, sequence, int(num_heads), head_dim)
        output_heads = output.view(batch, sequence, int(num_heads), head_dim)

        # Usually all selected heads fit in one slab and only sequence is
        # chunked.  The head loop also keeps the bound valid for unusually
        # wide models or large batches without changing the public API.
        heads_per_chunk = max(
            1,
            min(
                selected_heads,
                _HIDDEN_ROPE_FALLBACK_CHUNK_ELEMENTS // max(batch * head_dim, 1),
            ),
        )
        for relative_head_start in range(0, selected_heads, heads_per_chunk):
            relative_head_end = min(relative_head_start + heads_per_chunk, selected_heads)
            chunk_head_start = int(head_start) + relative_head_start
            chunk_head_end = int(head_start) + relative_head_end
            chunk_heads = chunk_head_end - chunk_head_start
            sequence_per_chunk = max(
                1,
                _HIDDEN_ROPE_FALLBACK_CHUNK_ELEMENTS // max(batch * chunk_heads * head_dim, 1),
            )
            for sequence_start in range(0, sequence, sequence_per_chunk):
                sequence_end = min(sequence_start + sequence_per_chunk, sequence)
                selected = value_heads[:, sequence_start:sequence_end, chunk_head_start:chunk_head_end, :].view(
                    batch, sequence_end - sequence_start, chunk_heads, pairs, 2
                )
                destination = output_heads[:, sequence_start:sequence_end, chunk_head_start:chunk_head_end, :].view(
                    batch, sequence_end - sequence_start, chunk_heads, pairs, 2
                )
                cosine = selected_cosine[:, sequence_start:sequence_end]
                sine = selected_sine[:, sequence_start:sequence_end]
                even = selected[..., 0].double()
                odd = selected[..., 1].double()
                destination[..., 0].copy_(even * cosine - odd * sine)
                destination[..., 1].copy_(odd * cosine + even * sine)
        return output

    # Rebind sequentially so the normalized Q input can be released before K
    # allocates its output clone and rotary work buffers.
    q = rotate(q)
    k = rotate(k)
    return q, k


def _eligible_hidden_qk_rmsnorm_rope_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    grid_size: tuple[int, int, int],
    eps: float,
    sequence_offset: int,
    start_frame: int,
    valid_tokens: int,
    head_start: int,
    head_end: int,
) -> bool:
    del eps
    hidden_size = q.shape[-1] if q.ndim == 3 else 0
    head_dim = hidden_size // int(num_heads) if num_heads else 0
    frames, height, width = (int(value) for value in grid_size)
    return (
        _inference_only(q, k, q_weight, k_weight, freqs)
        and _same_cuda_device(q, k, q_weight, k_weight, freqs)
        and all(_triton_device_supported(tensor) for tensor in (q, k, q_weight, k_weight, freqs))
        and q.dtype in _FLOAT_DTYPES
        and k.dtype == q.dtype
        and q_weight.dtype == q.dtype
        and k_weight.dtype == q.dtype
        and q.ndim == 3
        and k.shape == q.shape
        and q.numel() > 0
        and _regular_contiguous(q)
        and _regular_contiguous(k)
        and hidden_size <= 4096
        and hidden_size % int(num_heads) == 0
        and head_dim >= 6
        and head_dim % 2 == 0
        and q_weight.shape == k_weight.shape == (hidden_size,)
        and ((freqs.ndim == 2 and freqs.is_complex()) or (freqs.ndim == 3 and freqs.shape[-1] == 2))
        and freqs.shape[1] == head_dim // 2
        and frames > 0
        and height > 0
        and width > 0
        and int(start_frame) + frames <= freqs.shape[0]
        and max(height, width) <= freqs.shape[0]
        and 0 <= int(sequence_offset)
        and 0 <= int(valid_tokens) <= frames * height * width
        and 0 <= int(head_start) < int(head_end) <= int(num_heads)
    )


def _hidden_qk_rmsnorm_rope_3d_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    grid_size: tuple[int, int, int],
    eps: float,
    sequence_offset: int,
    start_frame: int,
    valid_tokens: int,
    head_start: int,
    head_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from worldfoundry.core.kernels.triton_diffusion import hidden_qk_rmsnorm_rope_3d as implementation

    return implementation(
        q,
        k,
        q_weight,
        k_weight,
        freqs,
        num_heads=num_heads,
        grid_size=grid_size,
        eps=eps,
        sequence_offset=sequence_offset,
        start_frame=start_frame,
        valid_tokens=valid_tokens,
        head_start=head_start,
        head_end=head_end,
    )


def hidden_qk_rmsnorm_rope_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    *,
    num_heads: int,
    grid_size: tuple[int, int, int],
    eps: float = 1e-6,
    sequence_offset: int = 0,
    start_frame: int = 0,
    valid_tokens: int | None = None,
    head_start: int = 0,
    head_end: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse full-hidden RMSNorm and packed interleaved 3D RoPE.

    This is a model-neutral video-transformer primitive. ``q`` and ``k`` use
    ``[batch, sequence, hidden]`` layout, while ``freqs`` contains packed
    complex frequency tables shared by the temporal/height/width sections.
    Sequence and head ranges make the same operator usable before or after
    sequence/head parallel sharding.
    """

    resolved_valid = int(grid_size[0]) * int(grid_size[1]) * int(grid_size[2]) if valid_tokens is None else valid_tokens
    resolved_head_end = int(num_heads) if head_end is None else head_end
    args = (
        q,
        k,
        q_weight,
        k_weight,
        freqs,
        int(num_heads),
        tuple(int(value) for value in grid_size),
        float(eps),
        int(sequence_offset),
        int(start_frame),
        int(resolved_valid),
        int(head_start),
        int(resolved_head_end),
    )
    if torch.compiler.is_compiling() or _requested_kernel_backend() in _TORCH_BACKENDS:
        return _hidden_qk_rmsnorm_rope_3d_torch(*args)
    return KERNEL_REGISTRY.dispatch(
        "hidden_qk_rmsnorm_rope_3d",
        _hidden_qk_rmsnorm_rope_3d_torch,
        *args,
        signature=_workload_signature(
            q,
            k,
            q_weight,
            k_weight,
            freqs,
            extra=args[5:],
        ),
    )


def _eligible_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
    rope_fp32: bool,
) -> bool:
    del eps, rope_fp32
    return (
        _inference_only(q, k, q_weight, k_weight, freqs)
        and _same_cuda_device(q, k, q_weight, k_weight, freqs)
        and all(_triton_device_supported(tensor) for tensor in (q, k, q_weight, k_weight, freqs))
        and q.dtype in _FLOAT_DTYPES
        and k.dtype == q.dtype
        and q_weight.dtype == q.dtype
        and k_weight.dtype == q.dtype
        and freqs.dtype in _FLOAT_DTYPES
        and q.ndim == 4
        and q.numel() > 0
        and k.shape == q.shape
        and q.shape[-1] in {32, 64, 80, 96, 128, 160, 192, 256}
        and q.shape[-1] % 2 == 0
        and _regular_contiguous(q)
        and _regular_contiguous(k)
        and q_weight.ndim == k_weight.ndim == 1
        and q_weight.numel() == k_weight.numel() == q.shape[-1]
        and freqs.ndim == 4
        and freqs.shape[0] == q.shape[1]
        and freqs.shape[1:3] == (1, 1)
        and freqs.shape[-1] == q.shape[-1]
        and freqs.stride(-1) == 1
    )


def _qk_rmsnorm_rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
    rope_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from worldfoundry.core.kernels.triton_diffusion import qk_rmsnorm_rope as implementation

    return implementation(q, k, q_weight, k_weight, freqs, eps, rope_fp32)


def qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float = 1e-6,
    *,
    rope_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Q/K RMSNorm and non-interleaved RoPE for ``[B, S, H, D]``.

    Real ``freqs`` contain full-width rotation angles. Complex ``freqs`` may
    contain the equivalent half-width cosine/sine pairs and use the exact
    PyTorch path until a benchmark justifies another specialized kernel.
    """

    if torch.compiler.is_compiling() or _requested_kernel_backend() in _TORCH_BACKENDS:
        return _qk_rmsnorm_rope_torch(q, k, q_weight, k_weight, freqs, eps, rope_fp32)
    return KERNEL_REGISTRY.dispatch(
        "qk_rmsnorm_rope",
        _qk_rmsnorm_rope_torch,
        q,
        k,
        q_weight,
        k_weight,
        freqs,
        eps,
        rope_fp32,
        signature=_workload_signature(q, k, q_weight, k_weight, freqs, extra=(float(eps), bool(rope_fp32))),
    )


@lru_cache(maxsize=1)
def _register_kernels() -> None:
    KERNEL_REGISTRY.register(
        "group_norm_silu",
        backend="triton",
        name="triton_group_norm_silu",
        priority=100,
        autotune=True,
        implementation=_group_norm_silu_triton,
        predicate=_eligible_group_norm_silu,
    )
    KERNEL_REGISTRY.register(
        "silu_and_mul",
        backend="triton",
        name="triton_silu_and_mul",
        priority=100,
        autotune=True,
        implementation=_silu_and_mul_triton,
        predicate=_eligible_silu_and_mul,
    )
    KERNEL_REGISTRY.register(
        "silu_mul",
        backend="triton",
        name="triton_silu_mul",
        priority=100,
        autotune=True,
        implementation=_silu_mul_triton,
        predicate=_eligible_silu_mul,
    )
    KERNEL_REGISTRY.register(
        "residual_gate_add",
        backend="triton",
        name="triton_residual_gate_add",
        priority=100,
        autotune=True,
        implementation=_residual_gate_triton,
        predicate=_eligible_residual_gate,
    )
    KERNEL_REGISTRY.register(
        "layer_norm_scale_shift",
        backend="triton",
        name="triton_layer_norm_scale_shift",
        priority=100,
        autotune=True,
        implementation=_layer_norm_scale_shift_triton,
        predicate=_eligible_layer_norm_scale_shift,
    )
    KERNEL_REGISTRY.register(
        "rms_norm_scale_shift",
        backend="triton",
        name="triton_rms_norm_scale_shift",
        priority=100,
        autotune=True,
        implementation=_rms_norm_scale_shift_triton,
        predicate=_eligible_rms_norm_scale_shift,
    )
    KERNEL_REGISTRY.register(
        "qk_rmsnorm_rope",
        backend="triton",
        name="triton_qk_rmsnorm_rope",
        priority=100,
        implementation=_qk_rmsnorm_rope_triton,
        predicate=_eligible_qk_rmsnorm_rope,
    )
    KERNEL_REGISTRY.register(
        "hidden_qk_rmsnorm_rope_3d",
        backend="triton",
        name="triton_hidden_qk_rmsnorm_rope_3d",
        priority=100,
        implementation=_hidden_qk_rmsnorm_rope_3d_triton,
        predicate=_eligible_hidden_qk_rmsnorm_rope_3d,
    )


_register_kernels()

__all__ = [
    "clear_kernel_dispatch_cache",
    "group_norm_silu",
    "hidden_qk_rmsnorm_rope_3d",
    "kernel_dispatch_report",
    "layer_norm_scale_shift",
    "qk_rmsnorm_rope",
    "rms_norm_scale_shift",
    "residual_gate_add",
    "silu_and_mul",
    "silu_mul",
]
