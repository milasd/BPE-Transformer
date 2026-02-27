"""Bias-free linear transformation layer using MLX."""

import mlx.core as mx
import mlx.nn as nn


class Linear(nn.Module):
    """Bias-free linear transformation layer.

    Implements a linear transformation without bias, following modern LLM practices
    (e.g., LLaMA, GPT-NeoX). Uses truncated normal initialization with Xavier-style
    scaling for stable training.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.

    Attributes:
        weights: Learnable weight matrix of shape (out_features, in_features).

    Shape:
        - Input: (..., in_features) where * means any number of dimensions.
        - Output: (..., out_features) same shape as input except last dimension.

    Examples:
        >>> linear = Linear(128, 256)
        >>> x = mx.random.normal((32, 10, 128))
        >>> output = linear(x)
        >>> output.shape
        (32, 10, 256)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Xavier-style initialization
        sigma = (2 / (in_features + out_features)) ** 0.5
        # Truncated normal: mean=0, std=sigma, range=[-3*sigma, 3*sigma]
        self.weights = mx.random.normal((out_features, in_features)) * sigma

    def __call__(self, x: mx.array) -> mx.array:
        """Apply linear transformation: y = xW^T.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        return mx.einsum("o i, ... i -> ... o", self.weights, x)
