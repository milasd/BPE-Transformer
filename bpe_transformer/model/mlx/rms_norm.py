"""Root Mean Square Layer Normalization using MLX."""

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes inputs using only the root mean square statistic, without
    centering (no mean subtraction). This is computationally cheaper than LayerNorm
    while maintaining similar performance. Used in modern LLMs like LLaMA.

    Formula: RMSNorm(x) = (x / RMS(x)) * γ
    where RMS(x) = sqrt(mean(x²) + eps)

    Args:
        d_model: Dimensionality of the input features (last dimension).
        eps: Small constant for numerical stability. Defaults to 1e-5.

    Attributes:
        weights: Learnable scale parameter γ of shape (d_model,).
        eps: Epsilon value for numerical stability.

    Shape:
        - Input: (..., d_model).
        - Output: (..., d_model) same shape as input.

    Examples:
        >>> norm = RMSNorm(d_model=768)
        >>> x = mx.random.normal((32, 128, 768))
        >>> output = norm(x)
        >>> output.shape
        (32, 128, 768)
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weights = mx.ones((d_model,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalized tensor of shape (..., d_model).
        """
        # Upcast to float32 for numerical stability
        in_dtype = x.dtype
        x = x.astype(mx.float32)

        # Compute RMS normalization
        norm = self._norm(x)
        rms_norm = (x / norm) * self.weights

        # Downcast to original dtype
        return rms_norm.astype(in_dtype)

    def _norm(self, x: mx.array) -> mx.array:
        """Compute root mean square along the last dimension.

        Args:
            x: Input tensor.

        Returns:
            RMS tensor with keepdim=True on last dimension.
        """
        return mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
