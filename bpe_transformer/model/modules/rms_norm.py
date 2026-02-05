import torch

from torch import nn, sqrt


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weights = nn.Parameter(
            data=nn.init.ones_(tensor=torch.empty(d_model, device=device, dtype=dtype)), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., d_model]
        """
        # upcast to torch.float32
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # calculate rms first
        norm = self._norm(x)
        rms_norm = (x / norm) * self.weights

        # downcast to original dtype
        return rms_norm.to(in_dtype)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt((x.pow(2)).mean(dim=-1, keepdim=True) + self.eps)
