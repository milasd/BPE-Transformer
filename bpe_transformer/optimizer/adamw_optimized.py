from collections.abc import Callable
from torch import optim
import torch
import math


class AdamWOptimized(optim.Optimizer):
    """Optimized AdamW with batched foreach operations for better performance."""

    def __init__(
        self,
        params,
        betas: tuple[float, float],
        weight_decay: float,
        lr: float = 1e-3,
        eps: float = 1e-8,
        foreach: bool = True,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
        self.foreach = foreach

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            if self.foreach:
                self._step_foreach(group)
            else:
                self._step_single(group)

        return loss

    def _step_single(self, group):
        """Single-parameter-at-a-time implementation."""
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            t = state.get("t", 0)
            grad = p.grad.data

            if t == 0:
                state["m"] = torch.zeros_like(p.data)
                state["v"] = torch.zeros_like(p.data)

            m = state["m"]
            v = state["v"]

            t += 1
            state["t"] = t

            state["m"] = beta1 * m + (1 - beta1) * grad
            state["v"] = beta2 * v + (1 - beta2) * grad * grad

            bias_correction1 = 1 - beta1**t
            bias_correction2 = 1 - beta2**t
            lr_t = lr * math.sqrt(bias_correction2) / bias_correction1

            p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)
            p.data -= lr * weight_decay * p.data

    def _step_foreach(self, group):
        """Batched implementation using torch._foreach for better performance."""
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []

        for p in group["params"]:
            if p.grad is None:
                continue

            params_with_grad.append(p)
            grads.append(p.grad.data)

            state = self.state[p]
            t = state.get("t", 0)

            if t == 0:
                state["m"] = torch.zeros_like(p.data)
                state["v"] = torch.zeros_like(p.data)

            state["t"] = t + 1
            exp_avgs.append(state["m"])
            exp_avg_sqs.append(state["v"])
            state_steps.append(state["t"])

        if len(params_with_grad) == 0:
            return

        # Batched momentum updates
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

        # Compute bias-corrected step sizes
        bias_correction1 = [1 - beta1**step for step in state_steps]
        bias_correction2 = [1 - beta2**step for step in state_steps]
        step_sizes = [lr * math.sqrt(bc2) / bc1 for bc1, bc2 in zip(bias_correction1, bias_correction2)]

        # Compute updates: step_size * m / (sqrt(v) + eps)
        denominators = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_add_(denominators, eps)

        updates = []
        for step_size, m, denom in zip(step_sizes, exp_avgs, denominators):
            updates.append(step_size * m / denom)

        # Update parameters with .data to avoid autograd tracking
        param_data = [p.data for p in params_with_grad]
        torch._foreach_sub_(param_data, updates)

        # Weight decay
        if weight_decay != 0:
            torch._foreach_mul_(param_data, 1 - lr * weight_decay)
