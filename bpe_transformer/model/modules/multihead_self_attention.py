import torch
from torch import nn
from einops import rearrange

from bpe_transformer.model.modules import RoPE, scaled_dot_product_attention, Linear


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        device: Device for parameters. Defaults to None.
        dtype: Data type for parameters. Defaults to None.
        rope: Optional RoPE module for positional encoding.

    Shape:
        - Input: (..., seq_len, d_model)
        - Output: (..., seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope: RoPE | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_v = d_model // num_heads

        # Store a single tensor for all k, q and v for efficiency
        self.w_qkv = Linear(in_features=d_model, out_features=3 * d_model)

        self.rope = rope

        # Output projection
        self.w_o = Linear(d_model, d_model, device, dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, token_positions: torch.Tensor | None = None):
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
        q, k, v = qkv.unbind(0)

        # Add RoPE Embedding to Q and K
        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Create causal mask if it's not provided
        if mask is None:
            seq_len = x.shape[-2]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=0)

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
