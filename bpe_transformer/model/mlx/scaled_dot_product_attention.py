"""Scaled dot-product attention using MLX."""

from math import sqrt
import mlx.core as mx

from bpe_transformer.model.mlx.softmax import softmax


def scaled_dot_product_attention(
    q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None = None
) -> mx.array:
    """Scaled dot-product attention mechanism.

    Computes: softmax(Q·K^T / sqrt(d_k)) · V

    Args:
        q: Query tensor of shape (..., seq_len_q, d_k).
        k: Key tensor of shape (..., seq_len_k, d_k).
        v: Value tensor of shape (..., seq_len_k, d_v).
        mask: Optional boolean mask of shape (..., seq_len_q, seq_len_k).
              True means attend, False means mask out.

    Returns:
        Attention output of shape (..., seq_len_q, d_v).
    """
    d_k = k.shape[-1]  # d_k = d_v
    score = mx.einsum("...nk, ...mk -> ...nm", q, k) / sqrt(d_k)

    # Apply '-inf' mask to pre-softmax values
    if mask is not None:
        score = mx.where(mask, score, float("-inf"))
    normalized_score = softmax(score, i=-1)
    attention = mx.einsum("...nm, ...mv -> ...nv", normalized_score, v)

    return attention
