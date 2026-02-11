from collections.abc import Callable
from torch import optim
import torch
import math


class AdamW(optim.Optimizer):
    def __init__(self, params, betas: tuple[float, float], weight_decay: float, lr: float = 1e-3, eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get state
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data

                if t == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]

                # increment timestep
                t += 1
                state["t"] = t

                # Update first and second moment estimates
                state["m"] = beta1 * m + (1 - beta1) * grad
                state["v"] = beta2 * v + (1 - beta2) * grad * grad

                lr_t = lr * math.sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t))

                # Update weights with Adam
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)

                # Weight decay
                p.data -= lr * weight_decay * p.data

        return loss
