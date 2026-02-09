import torch

from torch import nn


class Embedding(nn.Module):
    """Token embedding layer for transformer models.

    Maps discrete token IDs to continuous vector representations. Uses truncated
    normal initialization for stable training.

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens).
        d_model: Dimensionality of the embedding vectors.
        device: Device to place the parameters on. Defaults to None (CPU).
        dtype: Data type of the parameters. Defaults to None (default dtype).

    Attributes:
        embeddings: Learnable embedding matrix of shape (vocab_size, d_model).

    Shape:
        - Input: (..., sequence_length) with token IDs in range [0, vocab_size).
        - Output: (..., sequence_length, d_model).

    Examples:
        >>> embedding = Embedding(vocab_size=50000, d_model=768)
        >>> token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> embedded = embedding(token_ids)
        >>> embedded.shape
        torch.Size([2, 3, 768])
    """

    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()

        self.embeddings = nn.Parameter(
            data=nn.init.trunc_normal_(
                tensor=torch.empty(vocab_size, d_model, device=device, dtype=dtype), mean=0, std=1, a=-3, b=3
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for the given token IDs.

        Args:
            token_ids: Integer tensor of shape (..., sequence_length) with token IDs.

        Returns:
            Embedding tensor of shape (..., sequence_length, d_model).
        """
        return self.embeddings[token_ids]
