import torch

from bpe_transformer.model.modules.multihead_self_attention import MultiHeadSelfAttention
from bpe_transformer.model.modules.rms_norm import RMSNorm
from bpe_transformer.model.modules.rope import RoPE
from bpe_transformer.model.modules.swiglu import SwiGLU
from torch import nn


class Transformer(nn.Module):
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
        """
        x: input tensor (batch, seq_len, d_model)
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
