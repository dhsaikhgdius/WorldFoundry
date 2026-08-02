# Copyright 2025 StepFun Inc. All Rights Reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
"""Module for base_models -> diffusion_model -> video -> step_video -> step_video_runtime -> stepvideo -> text_encoder -> flashattention.py functionality."""

import torch
from worldfoundry.core.attention import scaled_dot_product_attention

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True,
                    return_attn_probs=False, tp_group_rank=0, tp_group_size=1):
    """Flash attn func.

    Args:
        q: The q.
        k: The k.
        v: The v.
        dropout_p: The dropout p.
        softmax_scale: The softmax scale.
        causal: The causal.
        return_attn_probs: The return attn probs.
        tp_group_rank: The tp group rank.
        tp_group_size: The tp group size.
    """
    del return_attn_probs, tp_group_rank, tp_group_size
    q, k, v = (value.transpose(1, 2) for value in (q, k, v))
    output = scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    return output.transpose(1, 2)


class FlashSelfAttention(torch.nn.Module):
    """Flash self attention implementation."""
    def __init__(
        self,
        attention_dropout=0.0,
    ):
        """Init.

        Args:
            attention_dropout: The attention dropout.
        """
        super().__init__()
        self.dropout_p = attention_dropout


    def forward(self, q, k, v, cu_seqlens=None, max_seq_len=None):
        """Forward.

        Args:
            q: The q.
            k: The k.
            v: The v.
            cu_seqlens: The cu seqlens.
            max_seq_len: The max seq len.
        """
        if cu_seqlens is None:
            output = flash_attn_func(q, k, v, dropout_p=self.dropout_p)
        else:
            raise ValueError('cu_seqlens is not supported!')

        return output
    
