import torch

from torch import nn


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE).

    Args:
        theta: Base value for rotation angles. Typically 10000.
        d_k: Dimension of key/query vectors (must be even).
        max_seq_len: Maximum sequence length.
        device: Device for parameters. Defaults to None.
        dtype: Data type for parameters. Defaults to None.

    Shape:
        - Input: (..., seq_len, d_k)
        - token_positions: (..., seq_len)
        - Output: (..., seq_len, d_k)
    """

    # Class constants for indexing
    COS_IDX = 0
    SIN_IDX = 1

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.theta = theta

        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # create rotation matrix buffer w/ cos and sin for every i and k
        cos_sin_cache = self._build_cache()
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    def _build_cache(self) -> torch.Tensor:
        """Build precomputed cos/sin cache for all positions and dimensions.

        Returns:
            Tensor of shape (max_seq_len, d_k//2, 2).
        """
        pos = torch.arange(self.max_seq_len, device=self.device).unsqueeze(1)  # (max_seq_len, 1)

        k = torch.arange(self.d_k // 2, device=self.device).unsqueeze(0)  # (1, d_k//2)

        angles = pos / (self.theta ** ((2 * k) / self.d_k))  # (max_seq_len, d_k//2)

        cos_sin_cache = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        return cos_sin_cache

    @torch.no_grad()
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings.

        Args:
            x: Input tensor of shape (..., seq_len, d_k).
            token_positions: Position indices of shape (..., seq_len).

        Returns:
            Rotated tensor of shape (..., seq_len, d_k).
        """
        # For each pair of elements (x1, x2) in each embedding,
        # 1. Rotated vector is: (x1, x2) * cos theta + (-x2, x1) * sin theta.
        cos = self.cos_sin_cache[token_positions, :, self.COS_IDX]  # [..., d_k // 2]
        sin = self.cos_sin_cache[token_positions, :, self.SIN_IDX]  # [..., d_k // 2]

        x1 = x[..., 0::2]  # i = 0, 2, 4... [, d_k // 2]
        x2 = x[..., 1::2]  # i = 1, 3, 5... [, d_k // 2]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1)  # [..., d_k // 2, 2]

        # Flatten back to [..., d_k]
        return rotated_x.flatten(start_dim=-2)
