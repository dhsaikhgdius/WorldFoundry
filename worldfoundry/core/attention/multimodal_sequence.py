# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""Dense, masked, sparse, and sequence-parallel multimodal attention."""

from typing import Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F

from worldfoundry.core.attention.backends import resolve_attention_backend
from worldfoundry.core.attention.native import scaled_dot_product_attention
from worldfoundry.core.attention.varlen import masked_attention
from worldfoundry.core.distributed.sequence_mesh_state import get_parallel_state
from worldfoundry.core.distributed.sequence_parallel_runtime import (
    all_gather,
    all_to_all_4D,
)

from .sparse_3d import ssta_3d_attention


@torch.compiler.disable
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    drop_rate: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    attn_mode: str = "flash",
) -> torch.Tensor:
    """
    Compute attention through the shared WorldFoundry attention backends.

    Args:
        q: Query tensor of shape [B, L, H, D]
        k: Key tensor of shape [B, L, H, D]
        v: Value tensor of shape [B, L, H, D]
        drop_rate: Dropout rate for attention weights.
        attn_mask: Optional attention mask of shape [B, L].
        causal: Whether to apply causal masking.
        attn_mode: Attention mode, either "flash" or "torch". Defaults to "flash".

    Returns:
        Output tensor after attention of shape [B, L, H*D]
    """
    attn_mode = resolve_attention_backend(attn_mode, q.device)

    if attn_mode == "torch":
        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = q.transpose(1, 2)  # B * H * L * D
        key = k.transpose(1, 2)    # B * H * L * D
        value = v.transpose(1, 2)  # B * H * L * D
        
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [torch.int64, torch.int32]:
                assert attn_mask.max() <= 1 and attn_mask.min() >= 0, 'Integer attention mask must be between 0 and 1 for torch attention.'
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError('Float attention mask is not implemented for torch attention.')
            attn_mask1 = einops.rearrange(attn_mask, 'b l -> b 1 l 1')
            attn_mask2 = einops.rearrange(attn_mask1, 'b 1 l 1 -> b 1 1 l')
            attn_mask = attn_mask1 & attn_mask2
        
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
        
        # transpose back
        x = x.transpose(1, 2)  # B * L * H * D
        b, s, h, d = x.shape
        out = x.reshape(b, s, -1)
        return out
    elif attn_mode in {"flash_attention_2", "flash_attention_3"}:
        x = masked_attention(
            q,
            k,
            v,
            query_mask=attn_mask,
            key_mask=attn_mask,
            causal=causal,
            dropout_p=drop_rate,
            version=3 if attn_mode == "flash_attention_3" else 2,
        )
        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)
        return out
    else:
        raise NotImplementedError(f"Unsupported attention backend: {attn_mode}")


@torch.compiler.disable
def parallel_attention(q, k, v, img_q_len, img_kv_len, 
                       attn_mode=None, text_mask=None, 
                       attn_param=None,
                       block_idx=None,
                       ):
    """Parallel attention.

    Args:
        q: The q.
        k: The k.
        v: The v.
        img_q_len: The img q len.
        img_kv_len: The img kv len.
        attn_mode: The attn mode.
        text_mask: The text mask.
        attn_param: The attn param.
        block_idx: The block idx.
    """
    return sequence_parallel_attention(q, k, v, img_q_len, img_kv_len, attn_mode, text_mask, attn_param=attn_param, block_idx=block_idx)


def sequence_parallel_attention(q, k, v, 
                                img_q_len, img_kv_len, 
                                attn_mode=None, text_mask=None,
                                attn_param=None,
                                block_idx=None,
                                ):
    """Sequence parallel attention.

    Args:
        q: The q.
        k: The k.
        v: The v.
        img_q_len: The img q len.
        img_kv_len: The img kv len.
        attn_mode: The attn mode.
        text_mask: The text mask.
        attn_param: The attn param.
        block_idx: The block idx.
    """
    assert attn_mode is not None
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled

    if enable_sp:
        sp_group = parallel_dims.sp_group
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank
    
    if enable_sp:
        # batch_size, seq_len, attn_heads, head_dim
        query = all_to_all_4D(query, sp_group, scatter_dim=2, gather_dim=1)
        key = all_to_all_4D(key, sp_group, scatter_dim=2, gather_dim=1)
        value = all_to_all_4D(value, sp_group, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            """Shrink head.

            Args:
                encoder_state: The encoder state.
                dim: The dim.
            """
            local_heads = encoder_state.shape[dim] // sp_size
            return encoder_state.narrow(
                dim, sp_rank * local_heads, local_heads
            )

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    attn_mode = resolve_attention_backend(attn_mode, query.device)
    
    if attn_mode == "sage_attention":
        from sageattention import sageattn
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        hidden_states = sageattn(query, key, value, tensor_layout="NHD", is_causal=False)
    elif attn_mode == "torch":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            attn_mask = None

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [torch.int64, torch.int32]:
                assert attn_mask.max() <= 1 and attn_mask.min() >= 0, 'Integer attention mask must be between 0 and 1 for torch attention.'
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError('Float attention mask is not implemented for torch attention.')
            
        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = query.transpose(1, 2)  # B * Head_num * length * dim
        key = key.transpose(1, 2)      # B * Head_num * length * dim
        value = value.transpose(1, 2)  # B * Head_num * length * dim

        pair_mask = None
        if attn_mask is not None:
            pair_mask = attn_mask[:, None, :, None] & attn_mask[:, None, None, :]
        hidden_states = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=pair_mask,
        )
        
        # transpose back
        hidden_states = hidden_states.transpose(1, 2)
        
    elif attn_mode == "flash_attention_2":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = masked_attention(
            query,
            key,
            value,
            query_mask=attn_mask,
            key_mask=attn_mask,
            version=2,
        )
        
    elif attn_mode == "flash_attention_3":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = masked_attention(
            query,
            key,
            value,
            query_mask=attn_mask,
            key_mask=attn_mask,
            version=3,
        )

    elif attn_mode == "flex_block_attention":
        sparse_type = attn_param["attn_sparse_type"]  # sta/block_attn/ssta
        ssta_threshold = attn_param["ssta_threshold"]
        ssta_lambda = attn_param["ssta_lambda"]
        ssta_sampling_type = attn_param["ssta_sampling_type"]
        ssta_adaptive_pool = attn_param["ssta_adaptive_pool"]

        attn_pad_type = attn_param["attn_pad_type"]  # repeat/zero
        attn_use_text_mask = attn_param["attn_use_text_mask"]
        attn_mask_share_within_head = attn_param["attn_mask_share_within_head"]

        ssta_topk = attn_param["ssta_topk"]
        thw = attn_param["thw"]
        tile_size = attn_param["tile_size"]
        win_size = attn_param["win_size"][0].copy()

        def get_image_tile(tile_size):
            """Get image tile.

            Args:
                tile_size: The tile size.
            """
            block_size = np.prod(tile_size)
            if block_size == 384:
                tile_size = (1, 16, 24)
            elif block_size == 128:
                tile_size = (1, 16, 8)
            elif block_size == 64:
                tile_size = (1, 8, 8)
            elif block_size == 16:
                tile_size = (1, 4, 4)
            else:
                raise ValueError(f"Error tile_size {tile_size}, only support in [16, 64, 128, 384]")
            return tile_size

        if thw[0] == 1:
            tile_size = get_image_tile(tile_size)
            win_size = [1, 1, 1]
        elif thw[0] <= 31: # 16fps: 5 * 16 / 4 + 1 = 21; 24fps: 5 * 24 / 4 + 1 = 31
            ssta_topk = ssta_topk // 2

        # Concatenate and permute query, key, value to (B, H, S, D)
        query = torch.cat([query, encoder_query], dim=1).permute(0, 2, 1, 3)
        key = torch.cat([key, encoder_key], dim=1).permute(0, 2, 1, 3)
        value = torch.cat([value, encoder_value], dim=1).permute(0, 2, 1, 3)

        assert (
            query.shape[-1] == 128
        ), "The last dimension of query, key and value must be 128 for FlexBlockAttention."

        hidden_states = ssta_3d_attention(
            query,
            key,
            value,
            thw,
            topk=ssta_topk,
            tile_thw=tile_size,
            kernel_thw=win_size,
            text_len=encoder_sequence_length,
            sparse_type=sparse_type,
            threshold=ssta_threshold,
            lambda_=ssta_lambda,
            pad_type=attn_pad_type,
            text_mask=text_mask if attn_use_text_mask else None,
            sampling_type=ssta_sampling_type,
            adaptive_pool=ssta_adaptive_pool,
            mask_share_within_head=attn_mask_share_within_head,
        )
        hidden_states, sparse_ratio = hidden_states
        hidden_states = hidden_states.permute(0, 2, 1, 3)

    else:
        raise NotImplementedError(
            f"Unsupported attention mode: {attn_mode}"
        )


    if enable_sp:
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes((sequence_length, encoder_sequence_length), dim=1)
        hidden_states = all_to_all_4D(hidden_states, sp_group, scatter_dim=1, gather_dim=2)
        encoder_hidden_states = all_gather(encoder_hidden_states, dim=2, group=sp_group).contiguous()
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states
