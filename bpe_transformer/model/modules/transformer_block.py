import torch
from torch import nn

from bpe_transformer.model.modules import MultiHeadSelfAttention, RMSNorm, RoPE, SwiGLU


class Transformer(nn.Module):
    """Transformer block with pre-normalization.

    Uses RMSNorm, multi-head self-attention, and SwiGLU feedforward.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension.
        device: Device for parameters. Defaults to None.
        dtype: Data type for parameters. Defaults to None.
        rope: Optional RoPE module for positional encoding.

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope: RoPE | None = None,
    ):
        super().__init__()
        self.rope = rope
        self.rms_norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, device=device, dtype=dtype, rope=rope)
        self.rms_norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
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
        res1 = torch.add(x, mha)

        # 3. Norm
        pre_norm2 = self.rms_norm2(res1)

        # 4. Position-Wise FF
        ff_swiglu = self.ff(pre_norm2)

        # Add residual
        res2 = torch.add(res1, ff_swiglu)

        return res2
