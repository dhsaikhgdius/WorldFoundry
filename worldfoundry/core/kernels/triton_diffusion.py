"""Portable Triton kernels used by diffusion and world-model blocks.

These kernels are authored in-tree and target regular CUDA tensors.  They do
not rely on FlashAttention, xFormers, SGLang, vLLM, or another source checkout.
"""

from __future__ import annotations

import torch

from worldfoundry.runtime.compile_cache import configure_persistent_compile_cache

configure_persistent_compile_cache(namespace="diffusion-triton")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _silu_mul_kernel(
    gate_ptr,
    value_ptr,
    out_ptr,
    elements,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(value_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output = gate * tl.sigmoid(gate) * value
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def _packed_silu_mul_kernel(
    input_ptr,
    out_ptr,
    output_elements,
    half_features: tl.constexpr,
    block: tl.constexpr,
):
    output_offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = output_offsets < output_elements
    row = output_offsets // half_features
    feature = output_offsets % half_features
    input_base = row * (2 * half_features) + feature
    gate = tl.load(input_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(input_ptr + input_base + half_features, mask=mask, other=0.0).to(tl.float32)
    output = gate * tl.sigmoid(gate) * value
    tl.store(out_ptr + output_offsets, output, mask=mask)


@triton.jit
def _broadcast_row_offset(row, d0, d1, d2, d3, s0, s1, s2, s3):
    i3 = row % d3
    row = row // d3
    i2 = row % d2
    row = row // d2
    i1 = row % d1
    i0 = row // d1
    return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3


@triton.jit
def _residual_gate_kernel(
    residual_ptr,
    update_ptr,
    gate_ptr,
    out_ptr,
    rows,
    features: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    gs0: tl.constexpr,
    gs1: tl.constexpr,
    gs2: tl.constexpr,
    gs3: tl.constexpr,
    gs4: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = (row < rows) & (cols < features)
    offsets = row * features + cols
    gate_base = _broadcast_row_offset(row, d0, d1, d2, d3, gs0, gs1, gs2, gs3)

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + gate_base + cols * gs4, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, residual + update * gate, mask=mask)


@triton.jit
def _layer_norm_scale_shift_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    rows,
    eps,
    features: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    ss3: tl.constexpr,
    ss4: tl.constexpr,
    hs0: tl.constexpr,
    hs1: tl.constexpr,
    hs2: tl.constexpr,
    hs3: tl.constexpr,
    hs4: tl.constexpr,
    round_modulation: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = (row < rows) & (cols < features)
    offsets = row * features + cols
    x_input = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x_input.to(tl.float32)
    mean = tl.sum(x, axis=0) / features
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / features
    # LayerNorm returns the input dtype before the following AdaLN
    # arithmetic. Preserve that rounding point for mixed fp16/bf16 + fp32
    # modulation as used by Wan/LingBot.
    normed = (centered * tl.rsqrt(variance + eps)).to(x_input.dtype)

    scale_base = _broadcast_row_offset(row, d0, d1, d2, d3, ss0, ss1, ss2, ss3)
    shift_base = _broadcast_row_offset(row, d0, d1, d2, d3, hs0, hs1, hs2, hs3)
    scale_input = tl.load(scale_ptr + scale_base + cols * ss4, mask=mask, other=0.0)
    shift_input = tl.load(shift_ptr + shift_base + cols * hs4, mask=mask, other=0.0)
    # Preserve PyTorch's eager expression boundaries while keeping one launch:
    # ``1 + scale`` and the following multiply each round when their promoted
    # dtype is fp16/bf16. Mixed low-precision or fp32 modulation stays fp32.
    scale_factor = (1.0 + scale_input.to(tl.float32)).to(scale_input.dtype)
    modulated = normed.to(tl.float32) * scale_factor.to(tl.float32)
    if round_modulation:
        modulated = modulated.to(x_input.dtype)
    output = modulated.to(tl.float32) + shift_input.to(tl.float32)
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def _rms_norm_scale_shift_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    rows,
    eps,
    features: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    ss3: tl.constexpr,
    ss4: tl.constexpr,
    hs0: tl.constexpr,
    hs1: tl.constexpr,
    hs2: tl.constexpr,
    hs3: tl.constexpr,
    hs4: tl.constexpr,
    has_weight: tl.constexpr,
    round_modulation: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = (row < rows) & (cols < features)
    offsets = row * features + cols
    x_input = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x_input.to(tl.float32)
    variance = tl.sum(x * x, axis=0) / features
    normed = x * tl.rsqrt(variance + eps)
    if has_weight:
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed *= weight
    # torch.rms_norm applies its learned weight in the accumulation dtype and
    # rounds the completed norm once. Rounding before the weight can introduce
    # 0.1-level BF16 errors after AdaLN modulation on common DiT activations.
    normed = normed.to(x_input.dtype)

    scale_base = _broadcast_row_offset(row, d0, d1, d2, d3, ss0, ss1, ss2, ss3)
    shift_base = _broadcast_row_offset(row, d0, d1, d2, d3, hs0, hs1, hs2, hs3)
    scale_input = tl.load(scale_ptr + scale_base + cols * ss4, mask=mask, other=0.0)
    shift_input = tl.load(shift_ptr + shift_base + cols * hs4, mask=mask, other=0.0)
    scale_factor = (1.0 + scale_input.to(tl.float32)).to(scale_input.dtype)
    modulated = normed.to(tl.float32) * scale_factor.to(tl.float32)
    if round_modulation:
        modulated = modulated.to(x_input.dtype)
    output = modulated.to(tl.float32) + shift_input.to(tl.float32)
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def _qk_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    freqs_ptr,
    q_stride_s,
    q_stride_h,
    k_stride_s,
    k_stride_h,
    freq_stride_s,
    freq_stride_d,
    sequence,
    heads,
    eps,
    rope_fp32: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // (sequence * heads)
    sequence_head = row % (sequence * heads)
    seq = sequence_head // heads
    head = sequence_head % heads
    cols = tl.arange(0, block)
    mask = cols < head_dim
    q_offsets = batch * sequence * q_stride_s + seq * q_stride_s + head * q_stride_h + cols
    k_offsets = batch * sequence * k_stride_s + seq * k_stride_s + head * k_stride_h + cols

    q_input = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)
    k_input = tl.load(k_ptr + k_offsets, mask=mask, other=0.0)
    q_float = q_input.to(tl.float32)
    k_float = k_input.to(tl.float32)
    q_rstd = tl.rsqrt(tl.sum(q_float * q_float, axis=0) / head_dim + eps)
    k_rstd = tl.rsqrt(tl.sum(k_float * k_float, axis=0) / head_dim + eps)
    half = head_dim // 2
    pair_cols = cols % half
    pair_offsets = batch * sequence * q_stride_s + seq * q_stride_s + head * q_stride_h + pair_cols
    k_pair_offsets = batch * sequence * k_stride_s + seq * k_stride_s + head * k_stride_h + pair_cols
    q_a = tl.load(q_ptr + pair_offsets, mask=pair_cols < half, other=0.0).to(tl.float32)
    q_b = tl.load(q_ptr + pair_offsets + half, mask=pair_cols < half, other=0.0).to(tl.float32)
    k_a = tl.load(k_ptr + k_pair_offsets, mask=pair_cols < half, other=0.0).to(tl.float32)
    k_b = tl.load(k_ptr + k_pair_offsets + half, mask=pair_cols < half, other=0.0).to(tl.float32)
    q_weight_a = tl.load(q_weight_ptr + pair_cols, mask=pair_cols < half, other=0.0).to(tl.float32)
    q_weight_b = tl.load(q_weight_ptr + pair_cols + half, mask=pair_cols < half, other=0.0).to(tl.float32)
    k_weight_a = tl.load(k_weight_ptr + pair_cols, mask=pair_cols < half, other=0.0).to(tl.float32)
    k_weight_b = tl.load(k_weight_ptr + pair_cols + half, mask=pair_cols < half, other=0.0).to(tl.float32)
    # Match torch RMSNorm's output cast before RoPE.  Wan's fp32 strategy
    # promotes that rounded result again for the rotary arithmetic.
    q_a = (q_a * q_rstd * q_weight_a).to(q_input.dtype)
    q_b = (q_b * q_rstd * q_weight_b).to(q_input.dtype)
    k_a = (k_a * k_rstd * k_weight_a).to(k_input.dtype)
    k_b = (k_b * k_rstd * k_weight_b).to(k_input.dtype)
    if rope_fp32:
        q_a = q_a.to(tl.float32)
        q_b = q_b.to(tl.float32)
        k_a = k_a.to(tl.float32)
        k_b = k_b.to(tl.float32)

    freq = tl.load(
        freqs_ptr + seq * freq_stride_s + pair_cols * freq_stride_d,
        mask=pair_cols < half,
        other=0.0,
    ).to(tl.float32)
    cosine = tl.cos(freq)
    sine = tl.sin(freq)
    rotated_q = tl.where(cols < half, q_a * cosine - q_b * sine, q_b * cosine + q_a * sine)
    rotated_k = tl.where(cols < half, k_a * cosine - k_b * sine, k_b * cosine + k_a * sine)
    tl.store(q_out_ptr + q_offsets, rotated_q, mask=mask)
    tl.store(k_out_ptr + k_offsets, rotated_k, mask=mask)


@triton.jit
def _hidden_qk_rmsnorm_rope_3d_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    freqs_ptr,
    sequence,
    hidden_size: tl.constexpr,
    head_dim: tl.constexpr,
    valid_tokens,
    sequence_offset,
    start_frame,
    height,
    width,
    temporal_pairs: tl.constexpr,
    height_pairs: tl.constexpr,
    store_feature_start: tl.constexpr,
    store_feature_end: tl.constexpr,
    eps,
    freq_stride_position,
    freq_stride_pair,
    freq_stride_component,
    block_hidden: tl.constexpr,
):
    """Fuse full-hidden Q/K RMSNorm with packed interleaved 3D RoPE."""

    row = tl.program_id(0)
    local_token = row % sequence
    global_token = sequence_offset + local_token
    cols = tl.arange(0, block_hidden)
    hidden_mask = cols < hidden_size
    offsets = row * hidden_size + cols
    q_input = tl.load(q_ptr + offsets, mask=hidden_mask, other=0.0)
    k_input = tl.load(k_ptr + offsets, mask=hidden_mask, other=0.0)
    q_float = q_input.to(tl.float32)
    k_float = k_input.to(tl.float32)
    q_rstd = tl.rsqrt(tl.sum(q_float * q_float, axis=0) / hidden_size + eps)
    k_rstd = tl.rsqrt(tl.sum(k_float * k_float, axis=0) / hidden_size + eps)

    pair_index = tl.arange(0, block_hidden // 2)
    even_col = pair_index * 2
    odd_col = even_col + 1
    pair_mask = odd_col < hidden_size
    rotate_mask = pair_mask & (even_col >= store_feature_start) & (odd_col < store_feature_end)
    even_offsets = row * hidden_size + even_col
    odd_offsets = row * hidden_size + odd_col
    q_even_input = tl.load(q_ptr + even_offsets, mask=pair_mask, other=0.0)
    q_odd_input = tl.load(q_ptr + odd_offsets, mask=pair_mask, other=0.0)
    k_even_input = tl.load(k_ptr + even_offsets, mask=pair_mask, other=0.0)
    k_odd_input = tl.load(k_ptr + odd_offsets, mask=pair_mask, other=0.0)
    q_weight_even = tl.load(q_weight_ptr + even_col, mask=pair_mask, other=0.0).to(tl.float32)
    q_weight_odd = tl.load(q_weight_ptr + odd_col, mask=pair_mask, other=0.0).to(tl.float32)
    k_weight_even = tl.load(k_weight_ptr + even_col, mask=pair_mask, other=0.0).to(tl.float32)
    k_weight_odd = tl.load(k_weight_ptr + odd_col, mask=pair_mask, other=0.0).to(tl.float32)

    # WanRMSNorm rounds the normalized value to the projection dtype before
    # multiplying by the learned weight.
    q_even = (q_even_input.to(tl.float32) * q_rstd).to(q_even_input.dtype)
    q_odd = (q_odd_input.to(tl.float32) * q_rstd).to(q_odd_input.dtype)
    k_even = (k_even_input.to(tl.float32) * k_rstd).to(k_even_input.dtype)
    k_odd = (k_odd_input.to(tl.float32) * k_rstd).to(k_odd_input.dtype)
    q_even = (q_even.to(tl.float32) * q_weight_even).to(q_even_input.dtype)
    q_odd = (q_odd.to(tl.float32) * q_weight_odd).to(q_odd_input.dtype)
    k_even = (k_even.to(tl.float32) * k_weight_even).to(k_even_input.dtype)
    k_odd = (k_odd.to(tl.float32) * k_weight_odd).to(k_odd_input.dtype)

    pairs_per_head = head_dim // 2
    pair_in_head = pair_index % pairs_per_head
    spatial_plane = height * width
    temporal_position = start_frame + global_token // spatial_plane
    spatial_position = global_token % spatial_plane
    height_position = spatial_position // width
    width_position = spatial_position % width
    rope_position = tl.where(
        pair_in_head < temporal_pairs,
        temporal_position,
        tl.where(pair_in_head < temporal_pairs + height_pairs, height_position, width_position),
    )
    valid = global_token < valid_tokens
    freq_base = rope_position * freq_stride_position + pair_in_head * freq_stride_pair
    cosine = tl.load(
        freqs_ptr + freq_base,
        mask=pair_mask & valid,
        other=1.0,
    ).to(tl.float32)
    sine = tl.load(
        freqs_ptr + freq_base + freq_stride_component,
        mask=pair_mask & valid,
        other=0.0,
    ).to(tl.float32)
    rotated_q_even = q_even.to(tl.float32) * cosine - q_odd.to(tl.float32) * sine
    rotated_q_odd = q_odd.to(tl.float32) * cosine + q_even.to(tl.float32) * sine
    rotated_k_even = k_even.to(tl.float32) * cosine - k_odd.to(tl.float32) * sine
    rotated_k_odd = k_odd.to(tl.float32) * cosine + k_even.to(tl.float32) * sine
    tl.store(q_out_ptr + even_offsets, tl.where(rotate_mask, rotated_q_even, q_even), mask=pair_mask)
    tl.store(q_out_ptr + odd_offsets, tl.where(rotate_mask, rotated_q_odd, q_odd), mask=pair_mask)
    tl.store(k_out_ptr + even_offsets, tl.where(rotate_mask, rotated_k_even, k_even), mask=pair_mask)
    tl.store(k_out_ptr + odd_offsets, tl.where(rotate_mask, rotated_k_odd, k_odd), mask=pair_mask)


def _padded_outer_shape_and_strides(
    x: torch.Tensor, broadcast: torch.Tensor
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    expanded = broadcast.expand_as(x)
    outer_shape = (1,) * (5 - x.ndim) + tuple(int(dim) for dim in x.shape)
    strides = (0,) * (5 - expanded.ndim) + tuple(int(stride) for stride in expanded.stride())
    return outer_shape[:-1], strides


def _num_warps(block: int) -> int:
    if block <= 256:
        return 4
    return 8


def silu_mul(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Compute ``silu(gate) * value`` in one regular pointwise launch."""

    output = torch.empty_like(gate)
    block = 1024
    _silu_mul_kernel[(triton.cdiv(gate.numel(), block),)](
        gate,
        value,
        output,
        gate.numel(),
        block=block,
        num_warps=4,
    )
    return output


def silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """Split the last dimension and compute ``silu(first) * second``."""

    half_features = int(input.shape[-1] // 2)
    output = torch.empty((*input.shape[:-1], half_features), dtype=input.dtype, device=input.device)
    block = 1024
    _packed_silu_mul_kernel[(triton.cdiv(output.numel(), block),)](
        input,
        output,
        output.numel(),
        half_features=half_features,
        block=block,
        num_warps=4,
    )
    return output


def residual_gate(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    rows = residual.numel() // residual.shape[-1]
    features = int(residual.shape[-1])
    dims, gate_strides = _padded_outer_shape_and_strides(residual, gate)
    block = triton.next_power_of_2(features)
    output_dtype = torch.promote_types(residual.dtype, torch.promote_types(update.dtype, gate.dtype))
    output = torch.empty_like(residual, dtype=output_dtype)
    _residual_gate_kernel[(rows,)](
        residual,
        update,
        gate,
        output,
        rows,
        features=features,
        d0=dims[0],
        d1=dims[1],
        d2=dims[2],
        d3=dims[3],
        gs0=gate_strides[0],
        gs1=gate_strides[1],
        gs2=gate_strides[2],
        gs3=gate_strides[3],
        gs4=gate_strides[4],
        block=block,
        num_warps=_num_warps(block),
    )
    return output


def layer_norm_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    rows = x.numel() // x.shape[-1]
    features = int(x.shape[-1])
    dims, scale_strides = _padded_outer_shape_and_strides(x, scale)
    _, shift_strides = _padded_outer_shape_and_strides(x, shift)
    block = triton.next_power_of_2(features)
    modulation_dtype = torch.promote_types(x.dtype, scale.dtype)
    output_dtype = torch.promote_types(x.dtype, torch.promote_types(scale.dtype, shift.dtype))
    output = torch.empty_like(x, dtype=output_dtype)
    _layer_norm_scale_shift_kernel[(rows,)](
        x,
        scale,
        shift,
        output,
        rows,
        float(eps),
        features=features,
        d0=dims[0],
        d1=dims[1],
        d2=dims[2],
        d3=dims[3],
        ss0=scale_strides[0],
        ss1=scale_strides[1],
        ss2=scale_strides[2],
        ss3=scale_strides[3],
        ss4=scale_strides[4],
        hs0=shift_strides[0],
        hs1=shift_strides[1],
        hs2=shift_strides[2],
        hs3=shift_strides[3],
        hs4=shift_strides[4],
        round_modulation=modulation_dtype in {torch.float16, torch.bfloat16},
        block=block,
        num_warps=_num_warps(block),
    )
    return output


def rms_norm_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    rows = x.numel() // x.shape[-1]
    features = int(x.shape[-1])
    dims, scale_strides = _padded_outer_shape_and_strides(x, scale)
    _, shift_strides = _padded_outer_shape_and_strides(x, shift)
    block = triton.next_power_of_2(features)
    modulation_dtype = torch.promote_types(x.dtype, scale.dtype)
    output_dtype = torch.promote_types(x.dtype, torch.promote_types(scale.dtype, shift.dtype))
    output = torch.empty_like(x, dtype=output_dtype)
    weight_pointer = x if weight is None else weight
    _rms_norm_scale_shift_kernel[(rows,)](
        x,
        weight_pointer,
        scale,
        shift,
        output,
        rows,
        float(eps),
        features=features,
        d0=dims[0],
        d1=dims[1],
        d2=dims[2],
        d3=dims[3],
        ss0=scale_strides[0],
        ss1=scale_strides[1],
        ss2=scale_strides[2],
        ss3=scale_strides[3],
        ss4=scale_strides[4],
        hs0=shift_strides[0],
        hs1=shift_strides[1],
        hs2=shift_strides[2],
        hs3=shift_strides[3],
        hs4=shift_strides[4],
        has_weight=weight is not None,
        round_modulation=modulation_dtype in {torch.float16, torch.bfloat16},
        block=block,
        num_warps=_num_warps(block),
    )
    return output


def qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
    rope_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, sequence, heads, head_dim = q.shape
    block = triton.next_power_of_2(head_dim)
    q_output = torch.empty_like(q)
    k_output = torch.empty_like(k)
    _qk_rmsnorm_rope_kernel[(batch * sequence * heads,)](
        q,
        k,
        q_output,
        k_output,
        q_weight,
        k_weight,
        freqs,
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        freqs.stride(0),
        freqs.stride(3),
        sequence,
        heads,
        float(eps),
        rope_fp32=bool(rope_fp32),
        head_dim=head_dim,
        block=block,
        num_warps=_num_warps(block),
    )
    return q_output, k_output


def hidden_qk_rmsnorm_rope_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    *,
    num_heads: int,
    grid_size: tuple[int, int, int],
    eps: float,
    sequence_offset: int,
    start_frame: int,
    valid_tokens: int,
    head_start: int,
    head_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = int(q.shape[-1])
    head_dim = hidden_size // int(num_heads)
    _, height, width = (int(value) for value in grid_size)
    complex_dim = head_dim // 2
    temporal_pairs = complex_dim - 2 * (complex_dim // 3)
    height_pairs = complex_dim // 3
    real_freqs = torch.view_as_real(freqs) if freqs.is_complex() else freqs
    block_hidden = triton.next_power_of_2(hidden_size)
    q_output = torch.empty_like(q)
    k_output = torch.empty_like(k)
    _hidden_qk_rmsnorm_rope_3d_kernel[(q.shape[0] * q.shape[1],)](
        q,
        k,
        q_output,
        k_output,
        q_weight,
        k_weight,
        real_freqs,
        q.shape[1],
        hidden_size=hidden_size,
        head_dim=head_dim,
        valid_tokens=int(valid_tokens),
        sequence_offset=int(sequence_offset),
        start_frame=int(start_frame),
        height=height,
        width=width,
        temporal_pairs=temporal_pairs,
        height_pairs=height_pairs,
        store_feature_start=int(head_start) * head_dim,
        store_feature_end=int(head_end) * head_dim,
        eps=float(eps),
        freq_stride_position=real_freqs.stride(0),
        freq_stride_pair=real_freqs.stride(1),
        freq_stride_component=real_freqs.stride(2),
        block_hidden=block_hidden,
        num_warps=_num_warps(block_hidden),
    )
    return q_output, k_output


__all__ = [
    "hidden_qk_rmsnorm_rope_3d",
    "layer_norm_scale_shift",
    "qk_rmsnorm_rope",
    "rms_norm_scale_shift",
    "residual_gate",
    "silu_and_mul",
    "silu_mul",
]
