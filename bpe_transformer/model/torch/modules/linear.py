import torch
import torch.nn as nn


class Linear(nn.Module):
    """Bias-free linear transformation layer.

    Implements a linear transformation without bias, following modern LLM practices
    (e.g., LLaMA, GPT-NeoX). Uses truncated normal initialization with Xavier-style
    scaling for stable training.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        device: Device to place the parameters on. Defaults to None (CPU).
        dtype: Data type of the parameters. Defaults to None (default dtype).

    Attributes:
        weights: Learnable weight matrix of shape (out_features, in_features).

    Shape:
        - Input: (..., in_features) where * means any number of dimensions.
        - Output: (..., out_features) same shape as input except last dimension.

    Examples:
        >>> linear = Linear(128, 256)
        >>> x = torch.randn(32, 10, 128)
        >>> output = linear(x)
        >>> output.shape
        torch.Size([32, 10, 256])
    """

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        sigma = (2 / (in_features + out_features)) ** 0.5
        self.weights = nn.Parameter(
            data=nn.init.trunc_normal_(
                tensor=torch.empty(out_features, in_features, device=device, dtype=dtype),
                mean=0.0,
                std=sigma,
                a=(-3) * sigma,
                b=3 * sigma,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation: y = xW^T.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        return torch.einsum("o i, ... i -> ... o", self.weights, x)
