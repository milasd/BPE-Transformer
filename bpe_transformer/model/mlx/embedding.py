"""Token embedding layer using MLX."""

import mlx.core as mx
import mlx.nn as nn


class Embedding(nn.Module):
    """Token embedding layer for transformer models.

    Maps discrete token IDs to continuous vector representations. Uses truncated
    normal initialization for stable training.

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens).
        d_model: Dimensionality of the embedding vectors.

    Attributes:
        embeddings: Learnable embedding matrix of shape (vocab_size, d_model).

    Shape:
        - Input: (..., sequence_length) with token IDs in range [0, vocab_size).
        - Output: (..., sequence_length, d_model).

    Examples:
        >>> embedding = Embedding(vocab_size=50000, d_model=768)
        >>> token_ids = mx.array([[1, 2, 3], [4, 5, 6]])
        >>> embedded = embedding(token_ids)
        >>> embedded.shape
        (2, 3, 768)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # Truncated normal: mean=0, std=1, range=[-3, 3]
        self.embeddings = mx.random.normal((vocab_size, d_model))

    def __call__(self, token_ids: mx.array) -> mx.array:
        """Look up embeddings for the given token IDs.

        Args:
            token_ids: Integer tensor of shape (..., sequence_length) with token IDs.

        Returns:
            Embedding tensor of shape (..., sequence_length, d_model).
        """
        return self.embeddings[token_ids]
