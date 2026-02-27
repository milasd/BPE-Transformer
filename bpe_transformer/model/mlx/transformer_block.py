"""Transformer block using MLX."""

import mlx.core as mx
import mlx.nn as nn

from bpe_transformer.model.mlx.multihead_self_attention import MultiHeadSelfAttention
from bpe_transformer.model.mlx.rms_norm import RMSNorm
from bpe_transformer.model.mlx.rope import RoPE
from bpe_transformer.model.mlx.swiglu import SwiGLU


class Transformer(nn.Module):
    """Transformer block with pre-normalization.

    Uses RMSNorm, multi-head self-attention, and SwiGLU feedforward.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension.
        rope: Optional RoPE module for positional encoding.

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE | None = None):
        super().__init__()
        self.rope = rope
        self.rms_norm1 = RMSNorm(d_model=d_model)
        self.mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope=rope)
        self.rms_norm2 = RMSNorm(d_model=d_model)
        self.ff = SwiGLU(d_model=d_model, d_ff=d_ff)

        self.num_heads = num_heads

    def __call__(self, x: mx.array, token_positions: mx.array | None = None) -> mx.array:
        """Apply transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            token_positions: Position indices for RoPE.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # 1. Norm
        pre_norm1 = self.rms_norm1(x)

        # 2. MHA (will use automatic causal masking)
        mha = self.mha(x=pre_norm1, token_positions=token_positions)

        # Add residual
        res1 = x + mha

        # 3. Norm
        pre_norm2 = self.rms_norm2(res1)

        # 4. Position-Wise FF
        ff_swiglu = self.ff(pre_norm2)

        # Add residual
        res2 = res1 + ff_swiglu

        return res2
