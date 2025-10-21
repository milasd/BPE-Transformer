import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Implement a Linear block without bias, following modern LLM implementations.
    """

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        # Initialize weights
        self.weights = nn.Parameter(nn.init.trunc_normal_(torch.empty(out_features, in_features)), requires_grad=False)
        # TODO: use device and dtype
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation to a tensor x (eg.: y = xW^T) with einsum.
        """
        # matmul: x @ self.weight.t()
        return torch.einsum("o i, ... i -> ... o", self.weights, x)
