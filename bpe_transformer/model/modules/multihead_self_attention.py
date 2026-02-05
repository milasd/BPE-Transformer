import torch

from bpe_transformer.model.modules.scaled_dot_product_attention import scaled_dot_product_attention
from bpe_transformer.model.modules.linear import Linear
from einops import rearrange
from torch import nn


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device, dtype):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_v = d_model // num_heads

        self.w_q = Linear(d_model, d_model, device, dtype)
        self.w_k = Linear(d_model, d_model, device, dtype)
        self.w_v = Linear(d_model, d_model, device, dtype)

        # Output projection
        self.w_o = Linear(d_model, d_model, device, dtype)


    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x: Input tensor of shape [..., seq_len, d_model]
            mask: Optional attention mask. If provided, should be shape [..., seq_len, seq_len]
                  where True means "can attend" and False means "cannot attend"
                  If None, creates a causal mask to prevent attending to future tokens
        """
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Split into multiple heads
        q_h = rearrange(q, "... seq_len (num_heads dim_head) -> ... num_heads seq_len dim_head", num_heads=self.num_heads, dim_head=self.d_head)
        k_h = rearrange(k, "... seq_len (num_heads dim_head) -> ... num_heads seq_len dim_head", num_heads=self.num_heads, dim_head=self.d_head)
        v_h = rearrange(v, "... seq_len (num_heads dim_head) -> ... num_heads seq_len dim_head", num_heads=self.num_heads, dim_head=self.d_head)

        # Create causal mask if it's not provided
        if mask is None:
            seq_len = x.shape[-2]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=0)

        attn = scaled_dot_product_attention(q=q_h, k=k_h, v=v_h, mask=mask)

        # concat
        attn_concat = rearrange(attn, "... num_heads seq_len dim_head -> ... seq_len (num_heads dim_head)", num_heads=self.num_heads, dim_head=self.d_head)

        output = self.w_o(attn_concat)

        return output
