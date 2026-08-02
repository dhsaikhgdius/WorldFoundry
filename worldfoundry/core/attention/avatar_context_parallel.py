# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""SkyReels Avatar-Style Model Context-Parallel Attention Implementations.

This module implements custom sequence/context-parallel attention operators required
by SkyReels/Wan2.1-style Avatar and Video-Action models. It specifically supports:
1. Target-masked cross-attention maps for identity/reference conditioning.
2. Dynamic Rotary Position Embedding (RoPE) modifications to handle variable length spatiotemporal grids.
3. Multi-node Sequence Parallelism using `xFuserLongContextAttention` to distribute the heavy
   causal self-attention calculation across GPUs, enabling high-resolution 3D generation.
"""

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from einops import rearrange
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

from worldfoundry.core.attention.sequence_parallel_rope import pad_freqs
from worldfoundry.core.nn import sinusoidal_embedding_1d


def _calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, mode="mean", attn_bias=None):
    """Calculates cross-attention maps between local visual queries and global reference keys.

    Projects and applies softmax across reference keys, strictly masked by `ref_target_masks`.
    This generates a density map depicting where in the sequence the model focuses on specific target objects.

    Args:
        visual_q: The visual query tensor, typically [B, Heads, Seq_q, Dim].
        ref_k: The reference key tensor, typically [B, Heads, Seq_k, Dim].
        ref_target_masks: Boolean mask tensor separating targeted object tokens from background.
        mode: Reduction mode across the Head dimension ("mean" or "max").
        attn_bias: Optional broadcastable bias matrix added before softmax.

    Returns:
        torch.Tensor: The reduced cross-attention activation maps shaped [NumClasses, Seq_q].
    """
    ref_k = ref_k.to(visual_q.dtype).to(visual_q.device)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    visual_q = visual_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = visual_q @ ref_k.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias

    x_ref_attn_map_source = attn.softmax(-1)
    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(visual_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(visual_q.dtype)

    for ref_target_mask in ref_target_masks:
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attnmap = x_ref_attn_map_source * ref_target_mask
        # Normalize over valid masked targets
        x_ref_attnmap = x_ref_attnmap.sum(-1) / ref_target_mask.sum()
        x_ref_attnmap = x_ref_attnmap.permute(0, 2, 1)
        if mode == "mean":
            x_ref_attnmap = x_ref_attnmap.mean(-1)
        elif mode == "max":
            x_ref_attnmap = x_ref_attnmap.max(-1)
        x_ref_attn_maps.append(x_ref_attnmap)
    return torch.concat(x_ref_attn_maps, dim=0)


def get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks=None, split_num=2, enable_sp=False):
    """Compute per-target reference attention maps for SkyReels-style avatar models.

    Orchestrates the sequence-parallel gathering of reference keys across GPUs and calculates
    attention maps in a chunked manner (to save VRAM on the `Seq_q x Seq_k` intermediate matrix).
    """
    _, n_h, n_w = shape
    if enable_sp:
        # Reconstruct the global reference key from all sequence-parallel workers
        ref_k = get_sp_group().all_gather(ref_k, dim=1)

    x_seqlens = n_h * n_w
    ref_k = ref_k[:, :x_seqlens]
    _, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(class_num, seq_lens, device=visual_q.device, dtype=visual_q.dtype)
    split_chunk = heads // split_num

    for index in range(split_num):
        x_ref_attn_maps_perhead = _calculate_x_ref_attn_map(
            visual_q[:, :, index * split_chunk : (index + 1) * split_chunk, :],
            ref_k[:, :, index * split_chunk : (index + 1) * split_chunk, :],
            ref_target_masks,
        )
        x_ref_attn_maps += x_ref_attn_maps_perhead

    return x_ref_attn_maps / split_num


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """Applies complex Rotary Position Embeddings (RoPE) to multi-dimensional temporal grids.

    Calculates rotary multiplications in double precision (float64) to prevent precision decay
    in extremely long video/3D context windows.

    Args:
        x: Input query/key tensor of shape [B, L, N, C].
        grid_sizes: The corresponding 3D/temporal grid dimensions [B, 3].
        freqs: Frequency bases generated via sinusoidal projection [M, C // 2].

    Returns:
        torch.Tensor: Rotary-embedded tensor of shape [B, L, N, C].
    """
    output = []
    for i, freqs_i_rank in enumerate(freqs):
        # precompute multipliers
        x_i = x[i].to(torch.float64).unflatten(-1, (-1, 2))  # .reshape(s, n, -1, 2)
        x_i_sin = x_i[..., 0] * freqs_i_rank[..., 0] - x_i[..., 1] * freqs_i_rank[..., 1]
        x_i_cos = x_i[..., 0] * freqs_i_rank[..., 1] + x_i[..., 1] * freqs_i_rank[..., 0]
        x_i = torch.stack([x_i_sin, x_i_cos], dim=-1).flatten(2)

        # append to collection
        output.append(x_i)
    return torch.stack(output)


def usp_dit_forward_avatar(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    audio=None,
    ref_target_masks=None,
    audio_mask=None,
    block_offload: bool = False,
):
    """Sequence-parallel Diffusion Transformer (DiT) forward pass customized for Avatar architectures.

    This forward method overrides the standard Wan/SkyReels DiT logic to inject:
    1. Reference cross-attention matrices via target-mask projections.
    2. Multi-node Sequence Parallel (SP) chunking of visual latent grids.
    3. Precision-safe distributed RoPE applications.

    Args:
        x: List of incoming visual latents, each of shape [C, T, H, W].
        t: Diffusion timesteps tensor [B].
        context: List of encoded text embeddings, each of shape [L, C].
        seq_len: Pre-calculated sequence length used for static padding and sequence-parallel splits.
        clip_fea: Reference CLIP features extracted from identity/style source images.
        y: Optional reference latents (e.g., target identity/pose videos) to concatenate.
        audio: Optional aligned audio embeddings for lip-sync or action-sync.
        ref_target_masks: Boolean target masking indicating visual ROI (e.g. face/hands).
        audio_mask: Boolean mask indicating temporal valid audio segments.
        block_offload: If True, performs CPU offloading of DiT blocks to conserve VRAM.

    Returns:
        torch.Tensor: The predicted diffusion noise or clean latent.
    """

    assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    _, _, H, W = x[0].shape
    N_h = H // self.patch_size[1]
    N_w = W // self.patch_size[2]

    if y is not None:
        # Concatenate reference latents alongside the denoising target
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
    x[0] = x[0].to(context[0].dtype)

    # Time embeddings (sinusoidal projection mapping diffusion t -> representation space)
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(self.patch_embedding.weight.dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # Spatial-temporal patch embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    # Pad local sequences to uniform sequence length
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

    # Construct Sequence-Parallel partitioned RoPE frequencies
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    s = x.size(1) // sp_size
    c = self.dim // self.num_heads // 2

    # Divide base frequencies into [T, H, W] axes for 3D RoPE
    freqs = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    freqs_l = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len_current = f * h * w
        # Broadcast separate T, H, W frequencies into the 3D block
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len_current, 1, -1)
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        # Sequence-parallel slice retrieval
        freqs_i_rank = freqs_i[(sp_rank * s) : ((sp_rank + 1) * s), :, :]
        freqs_i_rank = torch.view_as_real(freqs_i_rank)
        freqs_l.append(freqs_i_rank)
    freqs = freqs_l

    # Pad Context (Text) embeddings
    context_lens = None
    context = self.text_embedding(
        torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
    )

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    # get audio token
    audio_cond = audio.to(device=x.device, dtype=x.dtype)
    first_frame_audio_emb_s = audio_cond[:, :1, ...]
    latter_frame_audio_emb = audio_cond[:, 1:, ...]
    latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
    middle_index = self.audio_window // 2
    latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, : middle_index + 1, ...]
    latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
    latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index : middle_index + 1, ...]
    latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_frame_audio_emb_s = torch.concat(
        [latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2
    )
    first_frame_audio_emb_s = first_frame_audio_emb_s.contiguous()
    audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
    human_num = len(audio_embedding)
    audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x.dtype)

    # convert ref_target_masks to token_ref_target_masks
    if ref_target_masks is not None:
        ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32)
        token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(N_h, N_w), mode="nearest")
        token_ref_target_masks = token_ref_target_masks.squeeze(0)
        token_ref_target_masks = token_ref_target_masks > 0
        token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1)
        token_ref_target_masks = token_ref_target_masks.to(x.dtype)

    # Sequence/Context Parallel: Chunk visual latents across the GPU worker group
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    if audio_mask is not None:
        audio_mask = torch.chunk(audio_mask, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    # Assemble kwargs for DiT transformer blocks
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,
        context=context,
        context_lens=context_lens,
        audio_embedding=audio_embedding,
        ref_target_masks=token_ref_target_masks,
        human_num=human_num,
        enable_sp=True,
        audio_mask=audio_mask,
    )

    for block in self.blocks:
        x = block(x, **kwargs)

    # Prediction head mapping DiT latents back to image/video space
    x = self.head(x, e)

    # Sequence/Context Parallel: Gather outputs back from all GPU workers
    x = get_sp_group().all_gather(x, dim=1)

    # Unpatchify to standard video latent shape [C, T, H, W]
    x = self.unpatchify(x, grid_sizes)

    return torch.stack(x)


def usp_attn_forward_avatar(
    self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, ref_target_masks=None, human_num=None
):
    """Custom Attention forward pass for sequence-parallel architectures.

    Orchestrates Rotary Position Embeddings (RoPE), precision casting, and invokes the underlying
    Yunchang/xFuser LongContextAttention primitives for extremely long sequence modeling without
    OOM crashes.
    It also generates the global reference-key attention maps using target-masks.
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Query, Key, Value extraction
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    from yunchang.kernels import AttnType

    attn_type = AttnType.FA

    # Compute actual local/global attention utilizing efficient backend kernels
    x = xFuserLongContextAttention(attn_type=attn_type, use_pack_qkv=True)(
        None, query=half(q), key=half(k), value=half(v), window_size=self.window_size
    )

    # Output projection

    x = x.flatten(2)
    x = self.o(x)
    x_ref_attn_map = None
    if human_num != 1:
        with torch.no_grad():
            x_ref_attn_map = get_attn_map_with_target(
                q.type_as(x), k.type_as(x), grid_sizes[0], ref_target_masks=ref_target_masks, enable_sp=True
            )

    return x, x_ref_attn_map


def optimized_transform_before(x, N_t):
    x = get_sp_group().all_gather(x.to(torch.bfloat16).contiguous(), dim=1)
    # x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)
    S = x.shape[1] // N_t
    x = x.reshape(x.shape[0] * N_t, S, x.shape[2])
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    return x


def optimized_gather_after(x: torch.Tensor, N_t: int) -> torch.Tensor:
    x = get_sp_group().all_gather(x.to(torch.bfloat16).contiguous(), dim=1)
    # x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)
    B = x.shape[0] // N_t
    x = x.reshape(B, N_t * x.shape[1], x.shape[2])
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    return x


def usp_crossattn_multi_forward_avatar(
    self,
    x: torch.Tensor,
    encoder_hidden_states: torch.Tensor,  # 1, 21, 64, C
    shape=None,
    x_ref_attn_map=None,
    human_num=None,
    enable_sp=True,
) -> torch.Tensor:
    N_t = shape[0]
    x = optimized_transform_before(x, N_t)
    x = x if x.is_contiguous() else x.contiguous()

    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    if x_ref_attn_map is not None:
        x_ref_attn_map = get_sp_group().all_gather(
            x_ref_attn_map.to(torch.bfloat16).contiguous(), dim=1
        )  # -> [H, L_global]
        S_per_frame = x_ref_attn_map.shape[1] // N_t
        x_ref_attn_map = x_ref_attn_map.view(x_ref_attn_map.shape[0], N_t, S_per_frame)
        x_ref_attn_map = torch.chunk(x_ref_attn_map, sp_size, dim=2)[sp_rank]  # -> [H, N_t, S_per_frame/sp_size]
        x_ref_attn_map = x_ref_attn_map.reshape(x_ref_attn_map.shape[0], -1).type_as(x)

    x = self.origin_forward(
        x, encoder_hidden_states, shape=shape, x_ref_attn_map=x_ref_attn_map, enable_sp=enable_sp, human_num=human_num
    )
    x = optimized_gather_after(x, N_t)
    return x
