from torch import nn
import torch

from bpe_transformer.model.modules.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        SiLU(x) = x · σ(x)
        FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
        Typically, d_ff = 8/3 * d_model
        """
        super().__init__()
        # W1, W3: d_model -> d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        # W2: d_ff -> d_model
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_model)

        Return:
            (..., d_model) Tensor
        """
        # SiLU(W1x)
        silu = self.silu(self.w1(x))  # (..., d_model) -> (..., d_ff)

        return self.w2(silu * self.w3(x))  # (..., d_ff) -> (..., d_model)


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
