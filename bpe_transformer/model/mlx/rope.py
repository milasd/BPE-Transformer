"""Rotary Position Embedding (RoPE) using MLX."""

import mlx.core as mx
import mlx.nn as nn


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE).

    Args:
        theta: Base value for rotation angles. Typically 10000.
        d_k: Dimension of key/query vectors (must be even).
        max_seq_len: Maximum sequence length.

    Shape:
        - Input: (..., seq_len, d_k)
        - token_positions: (..., seq_len)
        - Output: (..., seq_len, d_k)
    """

    # Class constants for indexing
    COS_IDX = 0
    SIN_IDX = 1

    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # create rotation matrix buffer w/ cos and sin for every i and k
        self.cos_sin_cache = self._build_cache()

    def _build_cache(self) -> mx.array:
        """Build precomputed cos/sin cache for all positions and dimensions.

        Returns:
            Array of shape (max_seq_len, d_k//2, 2).
        """
        pos = mx.arange(self.max_seq_len)[:, None]  # (max_seq_len, 1)

        k = mx.arange(self.d_k // 2)[None, :]  # (1, d_k//2)

        angles = pos / (self.theta ** ((2 * k) / self.d_k))  # (max_seq_len, d_k//2)

        cos_sin_cache = mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)

        return cos_sin_cache

    def __call__(self, x: mx.array, token_positions: mx.array) -> mx.array:
        """Apply rotary position embeddings.

        Args:
            x: Input tensor of shape (..., seq_len, d_k).
            token_positions: Position indices of shape (..., seq_len).

        Returns:
            Rotated tensor of shape (..., seq_len, d_k).
        """
        # For each pair of elements (x1, x2) in each embedding,
        # 1. Rotated vector is: (x1, x2) * cos theta + (-x2, x1) * sin theta.
        cos = self.cos_sin_cache[token_positions, :, self.COS_IDX]  # [..., seq_len, d_k // 2]
        sin = self.cos_sin_cache[token_positions, :, self.SIN_IDX]  # [..., seq_len, d_k // 2]

        # If x has more dimensions than cos/sin (e.g., num_heads dimension),
        # unsqueeze cos/sin to match for broadcasting
        # x is (..., seq_len, d_k) or (..., num_heads, seq_len, d_k)
        while cos.ndim < x.ndim:
            cos = mx.expand_dims(cos, axis=-3)  # Add dimension before seq_len
            sin = mx.expand_dims(sin, axis=-3)

        x1 = x[..., 0::2]  # i = 0, 2, 4... [..., d_k // 2]
        x2 = x[..., 1::2]  # i = 1, 3, 5... [..., d_k // 2]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        rotated_x = mx.stack([rotated_x1, rotated_x2], axis=-1)  # [..., d_k // 2, 2]

        # Flatten back to [..., d_k]
        return rotated_x.reshape(x.shape)
