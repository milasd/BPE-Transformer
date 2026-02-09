from torch import nn
import torch

from bpe_transformer.model.modules import Linear


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU) feedforward network.

    SwiGLU is a variant of GLU that uses the Swish/SiLU activation function.
    It provides better performance than standard FFN layers in transformer models.
    Used in models like PaLM and LLaMA.

    Formula: SwiGLU(x) = W2(SiLU(W1·x) ⊙ W3·x)
    where SiLU(x) = x · σ(x) and ⊙ is element-wise multiplication.

    Args:
        d_model: Input and output dimensionality.
        d_ff: Inner/hidden dimensionality. Typically d_ff ≈ (8/3) * d_model
            to match parameter count of standard FFN with d_ff = 4 * d_model.
        device: Device to place the parameters on. Defaults to None (CPU).
        dtype: Data type of the parameters. Defaults to None (default dtype).

    Attributes:
        w1: First linear projection (d_model → d_ff).
        w2: Second linear projection (d_ff → d_model).
        w3: Third linear projection (d_model → d_ff).
        silu: SiLU activation function.

    Shape:
        - Input: (..., d_model).
        - Output: (..., d_model).

    Examples:
        >>> ffn = SwiGLU(d_model=768, d_ff=2048)
        >>> x = torch.randn(32, 128, 768)
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([32, 128, 768])
    """

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        # W1, W3: d_model -> d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        # W2: d_ff -> d_model
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model).
        """
        # Compute SiLU(W1·x) ⊙ W3·x, then project back
        silu_out = self.silu(self.w1(x))  # (..., d_model) -> (..., d_ff)
        return self.w2(silu_out * self.w3(x))  # (..., d_ff) -> (..., d_model)


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU) activation function.

    Also known as Swish activation. Computes x * sigmoid(x), providing smooth,
    non-monotonic activation that often outperforms ReLU in deep networks.

    Formula: SiLU(x) = x · σ(x) where σ is the sigmoid function.

    Shape:
        - Input: Any shape.
        - Output: Same shape as input.

    Examples:
        >>> silu = SiLU()
        >>> x = torch.tensor([-1.0, 0.0, 1.0])
        >>> silu(x)
        tensor([-0.2689,  0.0000,  0.7311])
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SiLU activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            Activated tensor of same shape as input.
        """
        return x * torch.sigmoid(x)
