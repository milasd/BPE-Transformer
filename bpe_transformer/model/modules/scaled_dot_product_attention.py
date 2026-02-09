from math import sqrt
import torch

from bpe_transformer.model.modules import softmax


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
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
    score = torch.einsum("...nk, ...mk -> ...nm", q, k) / sqrt(d_k)

    # Apply '-inf' mask to pre-softmax values
    if mask is not None:
        score.masked_fill_(~mask, value=float("-inf"))
        print(score)
    normalized_score = softmax(score, i=-1)
    attention = torch.einsum("...nm, ...mv -> ...nv", normalized_score, v)

    return attention
