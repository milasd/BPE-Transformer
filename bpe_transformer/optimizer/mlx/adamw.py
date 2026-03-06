"""AdamW optimizer for MLX."""

import mlx.core as mx
import mlx.optimizers as optim


class AdamW(optim.Optimizer):
    """AdamW optimizer implementation for MLX.

    Implements Adam with decoupled weight decay regularization.

    Args:
        learning_rate: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if learning_rate < 0:
            raise ValueError(f"Invalid learning rate: {learning_rate}")

        super().__init__()
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state for a single parameter."""
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)
        state["t"] = 0

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Apply optimizer update to a single parameter.

        Args:
            gradient: Gradient for this parameter
            parameter: Parameter to update
            state: Optimizer state for this parameter

        Returns:
            Updated parameter
        """
        lr = self.learning_rate
        beta1, beta2 = self.betas
        eps = self.eps
        weight_decay = self.weight_decay

        # Increment timestep
        state["t"] += 1
        t = state["t"]

        # Update biased first moment estimate
        state["m"] = beta1 * state["m"] + (1 - beta1) * gradient

        # Update biased second raw moment estimate
        state["v"] = beta2 * state["v"] + (1 - beta2) * gradient * gradient

        # Compute bias-corrected estimates
        m_hat = state["m"] / (1 - beta1**t)
        v_hat = state["v"] / (1 - beta2**t)

        # Apply weight decay
        parameter_update = parameter * (1 - lr * weight_decay)

        # Apply Adam update
        parameter_update = parameter_update - lr * m_hat / (mx.sqrt(v_hat) + eps)

        return parameter_update
