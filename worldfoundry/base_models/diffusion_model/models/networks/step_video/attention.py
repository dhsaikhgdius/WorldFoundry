"""Module for base_models -> diffusion_model -> video -> step_video -> step_video_runtime -> stepvideo -> modules -> attentions.py functionality."""

import torch
import torch.nn as nn
from einops import rearrange
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention

class Attention(nn.Module):
    """Attention implementation."""
    def __init__(self):
        """Init."""
        super().__init__()
    
    def attn_processor(self, attn_type):
        """Attn processor.

        Args:
            attn_type: The attn type.
        """
        if attn_type == 'torch':
            return self.torch_attn_func
        elif attn_type == 'parallel':
            raise ValueError("StepVideo parallel attention is configured by the shared runtime, not the model graph")
        else:
            raise Exception('Not supported attention type...')

    def torch_attn_func(
        self,
        q,
        k,
        v,
        attn_mask=None,
        causal=False,
        drop_rate=0.0,
        **kwargs
    ):
        """Torch attn func.

        Args:
            q: The q.
            k: The k.
            v: The v.
            attn_mask: The attn mask.
            causal: The causal.
            drop_rate: The drop rate.
        """

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
            
        if attn_mask is not None and attn_mask.ndim == 3:   ## no head
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        q, k, v = map(lambda x: rearrange(x, 'b s h d -> b h s d'), (q, k, v))
        x = _worldfoundry_scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
        x = rearrange(x, 'b h s d -> b s h d')
        return x        

    def parallel_attn_func(
        self,
        q,
        k,
        v,
        causal=False,
        **kwargs
    ):
        """Parallel attn func.

        Args:
            q: The q.
            k: The k.
            v: The v.
            causal: The causal.
        """
        raise RuntimeError("parallel attention must be installed through a WorldFoundry runtime extension")
