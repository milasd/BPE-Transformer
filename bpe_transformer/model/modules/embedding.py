import torch

from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()

        self.embeddings = nn.Parameter(
            data=nn.init.trunc_normal_(
                tensor=torch.empty(vocab_size, d_model, device=device, dtype=dtype), mean=0, std=1, a=-3, b=3
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch_size, sequence_length)

        Return:
           (batch_size, sequence_length, d_model)
        """
        return self.embeddings[token_ids]
