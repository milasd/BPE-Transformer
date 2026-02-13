import torch
from collections.abc import Iterable


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip gradients to have L2 norm at most max_l2_norm.

    Args:
        params: Iterable of parameters with gradients
        max_l2_norm: Maximum L2 norm
    """
    # Compute total L2 norm across all parameter gradients
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in params if p.grad is not None))

    # Clip if necessary
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(clip_coef)
