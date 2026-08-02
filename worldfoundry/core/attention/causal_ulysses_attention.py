from worldfoundry.core.attention.ulysses_attention import distributed_attention as _distributed_attention
from worldfoundry.core.attention.varlen import flash_attention


def distributed_attention(
    q,
    k,
    v,
    seq_lens,
    window_size=(-1, -1),
):
    """Run Ulysses attention with the causal LingBot attention kernel."""
    return _distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=window_size,
        attention_fn=flash_attention,
    )
