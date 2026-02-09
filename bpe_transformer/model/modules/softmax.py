import torch


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    """Numerically stable softmax function.

    Computes softmax along the specified dimension with max subtraction for
    numerical stability, preventing overflow/underflow issues.

    Formula: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Args:
        x: Input tensor of any shape.
        i: Dimension along which to compute softmax.

    Returns:
        Tensor of same shape as input with softmax applied along dimension i.
        Output values are in range [0, 1] and sum to 1 along dimension i.

    Examples:
        >>> x = torch.randn(2, 3, 4)
        >>> output = softmax(x, i=-1)
        >>> output.shape
        torch.Size([2, 3, 4])
        >>> output[0, 0].sum()  # Should be close to 1.0
        tensor(1.0000)
    """
    x_max = x.max(dim=i, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=i, keepdim=True)
