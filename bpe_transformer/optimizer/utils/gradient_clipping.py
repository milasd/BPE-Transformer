from collections import defaultdict
import torch
from collections.abc import Iterable


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip gradients to have L2 norm at most max_l2_norm.

    Args:
        params: Iterable of parameters with gradients
        max_l2_norm: Maximum L2 norm
    """
    # Collect gradients by device
    grads_by_device = defaultdict(list)

    for p in params:
        if p.grad is not None:
            grads_by_device[p.grad.device].append(p.grad)

    # Compute total L2 norm across all parameter gradients, across all devices
    total_norm = torch.sqrt(sum(sum(g.norm() ** 2 for g in grads) for grads in grads_by_device.values()))

    # Clip if necessary
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)

        # Apply clipping using foreach for each device.
        for grads in grads_by_device.values():
            torch._foreach_mul_(grads, clip_coef)
