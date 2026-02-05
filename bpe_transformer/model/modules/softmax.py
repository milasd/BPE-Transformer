import torch


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    x_max = x.max(dim=i, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=i, keepdim=True)
