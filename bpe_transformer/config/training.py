"""Training configuration for BPE Transformer."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Training configuration for BPE Transformer."""

    # Model architecture
    vocab_size: int
    context_length: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int

    # RoPE parameters
    theta: int

    # seed for reproducibility
    seed: int

    # Training hyperparameters
    batch_size: int
    num_iterations: int
    learning_rate_max: float
    learning_rate_min: float
    warmup_iterations: int
    grad_clip_norm: float
    use_mixed_precision: bool = False

    # Optimizer hyperparameters
    betas: tuple[float, float]
    weight_decay: float

    # Logging and checkpointing intervals
    checkpoint_interval: int
    val_interval: int
    log_interval: int

    # Optional: Experiment tracking w/ wandb
    wandb_project: str
    experiment_name: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert config to dict for saving."""
        return self.model_dump()
