import torch


def gradient_clipping(params: torch.Tensor, max_l2_norm: float) -> torch.Tensor:
    l2_norm = torch.norm(params)
    
    if l2_norm <= max_l2_norm:
        return params
    
    return params / (l2_norm + 1e-6)
