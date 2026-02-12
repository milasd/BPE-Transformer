from collections.abc import Callable
from torch import optim
import torch


class AdamW(optim.Optimizer):
    def __init__(self, params, betas: tuple[float, float], weight_decay: float, lr: float = 1e-3, eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        # Collect all tensors for a group
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            weight_decay_factor = 1 - lr * weight_decay

            params = []
            grads = []
            m_list = []
            v_list = []
            t_list = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Initialize state if t == 0.
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                t = state.get("t", 0)
                state["t"] = t + 1

                params.append(p)
                grads.append(p.grad)
                m_list.append(state["m"])
                v_list.append(state["v"])
                t_list.append(state["t"])

            # Update all tensors
            torch._foreach_mul_(m_list, beta1)  # m = m * beta1 for all
            torch._foreach_add_(m_list, grads, alpha=(1 - beta1))  # m += (1-beta1) * grad

            torch._foreach_mul_(v_list, beta2)  # v = v * beta2 for all
            torch._foreach_addcmul_(v_list, grads, grads, value=(1 - beta2))  # v += (1 - beta2) * grad * grad

            # Update weights: Weight decay
            if weight_decay > 0:
                torch._foreach_mul_(params, weight_decay_factor)

            # Bias correction
            bias_correction1 = [1 - beta1**t for t in t_list]
            bias_correction2 = [1 - beta2**t for t in t_list]

            m_hat_list = torch._foreach_div(m_list, bias_correction1)
            v_hat_list = torch._foreach_sqrt(
                torch._foreach_div(v_list, bias_correction2)
            )

            # denom: sqrt(v_hat) + eps
            torch._foreach_add_(v_hat_list, eps)

            # Update weights with Adam: p -= lr * m_hat / (sqrt(v_hat) + eps)
            torch._foreach_addcdiv_(params, m_hat_list, v_hat_list, value=-lr)

        return loss
