from torch import nn
import torch

from bpe_transformer.model.modules import Embedding, Linear, RMSNorm, RoPE, Transformer


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size=vocab_size, d_model=d_model, device=device, dtype=dtype)
        self.rope = rope
        self.transformer_blocks = nn.ModuleList(
            [
                Transformer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, device=device, dtype=dtype, rope=rope)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.linear = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        x contain text inputs
        """
        _, seq_len = x.shape
        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length {seq_len} exceeds context_length {self.context_length}")

        # 1. Embedding
        token_embeddings = self.token_embedding(token_ids=x)

        # 2. Transformer Blocks
        # Create token positions if rope embeddings will be added
        if self.rope is not None and token_positions is None:
            raise ValueError("Must pass token positions if rope embeddings will be used")

        x_transf = token_embeddings
        for transformer_block in self.transformer_blocks:
            x_transf = transformer_block(x=x_transf, token_positions=token_positions)

        # 3. Norm
        norm = self.norm(x=x_transf)

        # 4. Linear projection to vocab_size
        logits = self.linear(x=norm)

        return logits
