from math import sqrt
import torch

from bpe_transformer.model.modules.softmax import softmax


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Return softmax(QtK/sqrt(d_k)) V # Shape [..., seq_len, d_v]
    """
    d_k = k.shape[-1]  # d_k = d_v
    score = torch.einsum("...nk, ...mk -> ...nm", q, k) / sqrt(d_k)

    # Apply '-inf' mask to pre-softmax values
    if mask is not None:
        score.masked_fill_(mask == False, value=float("-inf"))
        print(score)
    normalized_score = softmax(score, i=-1)
    attention = torch.einsum("...nm, ...mv -> ...nv", normalized_score, v)

    return attention
