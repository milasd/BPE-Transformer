from math import cos
from torch import pi


def lr_cosine_schedule(t: int, t_warmup: int, t_cosine: int, lr_max: float, lr_min: float):
    # Warm-up
    if t < t_warmup:
        return lr_max * t / t_warmup
    # Post-annealing
    if t > t_cosine:
        return lr_min
    # Cosine annealing
    return lr_min + (0.5 * (1 + cos((t - t_warmup) * pi / (t_cosine - t_warmup))) * (lr_max - lr_min))
