"""Training script for BPE Transformer language model."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from bpe_transformer.model.modules import RoPE
from bpe_transformer.model import TransformerLM
from bpe_transformer.optimizer import AdamW, cross_entropy_loss, gradient_clipping, lr_cosine_schedule
from bpe_transformer.training import data_loader, load_checkpoint, save_checkpoint


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for training."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_model(config: dict[str, Any], device: str) -> nn.Module:
    """Initialize the transformer language model."""
    rope = RoPE(theta=config.get("theta", 10000))

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_heads=config["num_heads"],
        rope=rope,
        device=device,
    )

    return model


def init_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Initialize the optimizer."""
    learning_rate_max = config.get("learning_rate_max", 3e-4)
    weight_decay = config.get("weight_decay", 0.1)
    betas = tuple(config.get("betas", [0.9, 0.95]))

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate_max,
        betas=betas,
        weight_decay=weight_decay,
    )

    return optimizer


def setup_training(
    data_path: str | Path,
    device: str,
    resume_from: str | Path | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer, np.ndarray, dict[str, Any], int]:
    """Setup all components needed for training.

    Args:
        config_path: Path to YAML configuration file
        data_path: Path to training data (tokenized numpy array)
        device: Device to train on ('cuda', 'mps', or 'cpu')
        resume_from: Optional path to checkpoint to resume training from

    Returns:
        Tuple of (model, optimizer, training_data, config, start_iteration)
    """
    # Load training data
    logger.info(f"Loading training data from {data_path}")
    training_data = np.load(data_path)
    logger.info(f"Training data shape: {training_data.shape}")

    # Initialize model
    logger.info("Initializing model...")
    model = init_model(config, device)
    model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Initialize optimizer
    optimizer = init_optimizer(model, config)

    # Resume from checkpoint if specified
    start_iteration = 0
    if resume_from is not None:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        start_iteration = load_checkpoint(resume_from, model, optimizer)
        logger.info(f"Resumed from iteration {start_iteration}")

    return model, optimizer, training_data, start_iteration


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_data: np.ndarray,
    config: dict[str, Any],
    device: str,
    checkpoint_dir: str | Path = "checkpoints",
    start_iteration: int = 0,
) -> None:
    """Main training loop for the transformer language model.

    Args:
        model: Initialized transformer model
        optimizer: Initialized optimizer
        training_data: Tokenized training data as numpy array
        config: Configuration dictionary
        device: Device to train on ('cuda', 'mps', or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        start_iteration: Iteration to start/resume training from
    """
    # Training hyperparameters (with sensible defaults)
    batch_size = config.get("batch_size", 32)
    num_iterations = config.get("num_iterations", 10000)
    learning_rate_max = config.get("learning_rate_max", 3e-4)
    learning_rate_min = config.get("learning_rate_min", 3e-5)
    warmup_iterations = config.get("warmup_iterations", 1000)
    grad_clip_norm = config.get("grad_clip_norm", 1.0)

    # Checkpointing and logging
    checkpoint_interval = config.get("checkpoint_interval", 1000)
    log_interval = config.get("log_interval", 100)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set model to training mode
    model.train()

    # Training loop
    logger.info(f"Starting training for {num_iterations} iterations...")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Context length: {config['context_length']}")
    logger.info("-" * 60)

    for iteration in range(start_iteration, num_iterations):
        # Update learning rate
        lr = lr_cosine_schedule(
            t=iteration,
            t_warmup=warmup_iterations,
            t_cosine=num_iterations,
            lr_max=learning_rate_max,
            lr_min=learning_rate_min,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Load batch
        inputs, labels = data_loader(
            x=training_data,
            batch_size=batch_size,
            context_length=config["context_length"],
            device=device,
        )

        # Create position indices for RoPE
        token_positions = torch.arange(config["context_length"], device=device).unsqueeze(0).expand(batch_size, -1)

        # Forward pass
        logits = model(inputs, token_positions=token_positions)

        # Compute loss
        loss = cross_entropy_loss(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), max_l2_norm=grad_clip_norm)

        # Optimizer step
        optimizer.step()

        # Logging
        if (iteration + 1) % log_interval == 0:
            logger.info(f"Iteration {iteration + 1:>6}/{num_iterations} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

        # Save checkpoint
        if (iteration + 1) % checkpoint_interval == 0:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_iter_{iteration + 1}.pt"
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = Path(checkpoint_dir) / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, num_iterations, final_checkpoint_path)
    logger.info(f"Training complete! Final checkpoint saved: {final_checkpoint_path}")


if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="INFO")

    # Paths and device configuration
    config_path = "bpe_transformer/config/TinyStories_17M.yaml"
    data_path = "data/training_data.npy"  # Update with your actual data path
    device = "mps"  # Change to "cuda" if using NVIDIA GPU, or "cpu" for CPU
    checkpoint_dir = "checkpoints"
    resume_from = None  # Set to checkpoint path to resume training

    # Load configuration
    config = load_config(config_path)
    
    # Setup training components
    model, optimizer, training_data, start_iteration = setup_training(
        config_path=config_path,
        data_path=data_path,
        device=device,
        resume_from=resume_from,
    )

    # Start training
    train(
        model=model,
        optimizer=optimizer,
        training_data=training_data,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        start_iteration=start_iteration,
    )