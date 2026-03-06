"""Multi-head self-attention using MLX."""

import mlx.core as mx
import mlx.nn as nn
from einops import rearrange

from bpe_transformer.model.mlx.linear import Linear
from bpe_transformer.model.mlx.rope import RoPE
from bpe_transformer.model.mlx.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        rope: Optional RoPE module for positional encoding.

    Shape:
        - Input: (..., seq_len, d_model)
        - Output: (..., seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_v = d_model // num_heads

        # Store a single tensor for all k, q and v for efficiency
        self.w_qkv = Linear(in_features=d_model, out_features=3 * d_model)

        self.rope = rope

        # Output projection
        self.w_o = Linear(d_model, d_model)

    def __call__(self, x: mx.array, mask: mx.array | None = None, token_positions: mx.array | None = None) -> mx.array:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (..., seq_len, d_model).
            mask: Optional attention mask of shape (..., seq_len, seq_len).
                  True = attend, False = mask. Defaults to causal mask.
            token_positions: Position indices for RoPE of shape (..., seq_len).

        Returns:
            Output tensor of shape (..., seq_len, d_model).
        """
        if self.rope and token_positions is None:
            raise ValueError("Must pass token_positions if RoPE embeddings are to be applied.")

        qkv = self.w_qkv(x)
        qkv = rearrange(
            qkv,
            "... seq_len (n num_heads dim_head) -> n ... num_heads seq_len dim_head",
            n=3,
            num_heads=self.num_heads,
            dim_head=self.d_head,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Add RoPE Embedding to Q and K
        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Create causal mask if it's not provided
        if mask is None:
            seq_len = x.shape[-2]
            mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=0)

        attn = scaled_dot_product_attention(q=q, k=k, v=v, mask=mask)

        # concat
        attn_concat = rearrange(
            attn,
            "... num_heads seq_len dim_head -> ... seq_len (num_heads dim_head)",
            num_heads=self.num_heads,
            dim_head=self.d_head,
        )

        output = self.w_o(attn_concat)

        return output
