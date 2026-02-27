"""Softmax function using MLX."""

import mlx.core as mx


def softmax(x: mx.array, i: int = -1) -> mx.array:
    """Compute softmax along specified axis.

    Args:
        x: Input tensor.
        i: Axis along which to compute softmax. Defaults to -1.

    Returns:
        Softmax probabilities along the specified axis.
    """
    return mx.softmax(x, axis=i)
