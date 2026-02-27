import argparse
import logging
import math
import numpy as np
import random
import time
import torch
import yaml

from bpe_transformer.config import TrainingConfig
from bpe_transformer.model.torch.modules import RoPE
from bpe_transformer.model.torch import TransformerLM
from bpe_transformer.optimizer.torch import AdamW, cross_entropy_loss, gradient_clipping, lr_cosine_schedule
from bpe_transformer.training import data_loader, load_checkpoint, save_checkpoint
from datetime import datetime
from pathlib import Path
from torch import nn
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for training."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from loss.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity (exp(loss))
    """
    return math.exp(loss)


def create_run_directory(base_dir: str | Path, config: TrainingConfig) -> Path:
    """Create a unique directory for this training run.

    Args:
        base_dir: Base directory for all checkpoints
        config: Training configuration

    Returns:
        Path to the unique run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    lr = config.learning_rate_max
    bs = config.batch_size
    iters = config.num_iterations
    run_name = f"run_{timestamp}_lr{lr}_bs{bs}_iters{iters}"

    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_save_path = run_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    logger.info(f"Created run directory: {run_dir}")
    logger.info(f"Saved config to: {config_save_path}")

    return run_dir


def init_model(config: TrainingConfig, device: str) -> nn.Module:
    """Initialize the transformer language model."""
    rope = RoPE(
        theta=config.theta,
        d_k=config.d_model // config.num_heads,
        max_seq_len=config.context_length,
        device=device,
    )

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_heads=config.num_heads,
        rope=rope,
        device=device,
    )

    return model


def init_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Initialize the optimizer."""
    optimizer = AdamW(
        params=model.parameters(),
        lr=config.learning_rate_max,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    return optimizer


def load_data(data_path: str | Path, val_data_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load training and validation data using memory-mapped files.

    Uses memory-mapped mode to lazily load data on-demand, avoiding loading
    the entire dataset into RAM.

    Args:
        data_path: Path to training data (tokenized numpy array)
        val_data_path: Path to validation data (tokenized numpy array)
        vocab_size: Vocabulary size for validation

    Returns:
        Tuple of (training_data, val_data) as memory-mapped arrays
    """
    logger.info(f"Loading training data from {data_path} (memory-mapped)")
    training_data = np.load(data_path, mmap_mode="r")
    logger.info(f"Training data shape: {training_data.shape}")

    logger.info(f"Loading validation data from {val_data_path} ")
    val_data = np.load(val_data_path, mmap_mode="r")
    logger.info(f"Validation data shape: {val_data.shape}")

    return training_data, val_data


def setup_training(
    config: TrainingConfig,
    data_path: str | Path,
    val_data_path: str | Path,
    device: str,
    resume_from: str | Path | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer, np.ndarray, np.ndarray, int]:
    """Setup all components needed for training.

    Args:
        config: Configuration dictionary
        data_path: Path to training data (tokenized numpy array)
        val_data_path: Path to validation data (tokenized numpy array)
        device: Device to train on ('cuda', 'mps', or 'cpu')
        resume_from: Optional path to checkpoint to resume training from

    Returns:
        Tuple of (model, optimizer, training_data, val_data, start_iteration)
    """
    # Load data
    training_data, val_data = load_data(data_path, val_data_path)

    # init. model
    logger.info("Initializing model...")
    model = init_model(config, device)
    model = (
        torch.compile(model, backend="cudagraphs") if device == "cuda" else torch.compile(model, backend="aot_eager")
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    embedding_params = sum(p.numel() for p in model.token_embedding.parameters())
    non_embedding_params = num_params - embedding_params
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"  - Embedding parameters: {embedding_params:,}")
    logger.info(f"  - Non-embedding parameters: {non_embedding_params:,}")

    optimizer = init_optimizer(model, config)

    # Resume from checkpoint if specified
    start_iteration = 0
    if resume_from is not None:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        start_iteration = load_checkpoint(resume_from, model, optimizer)
        logger.info(f"Resumed from iteration {start_iteration}")

    return model, optimizer, training_data, val_data, start_iteration


def validate(
    model: nn.Module,
    val_data: np.ndarray,
    config: TrainingConfig,
    device: str,
    num_batches: int = 50,
) -> tuple[float, float]:
    """Run validation and return average loss and perplexity.

    Args:
        model: Model to validate
        val_data: Validation data
        config: Configuration dictionary
        device: Device to run validation on
        num_batches: Number of batches to validate on

    Returns:
        Tuple of (average validation loss, perplexity)
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            # Load batch
            inputs, labels = data_loader(
                x=val_data,
                batch_size=config.batch_size,
                context_length=config.context_length,
                device=device,
            )

            # Create position indices for RoPE
            token_positions = (
                torch.arange(config.context_length, device=device).unsqueeze(0).expand(config.batch_size, -1)
            )

            # Forward pass
            logits = model(inputs, token_positions=token_positions)
            loss = cross_entropy_loss(logits, labels)

            total_loss += loss.item()

    model.train()
    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)
    return avg_loss, perplexity


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_data: np.ndarray,
    val_data: np.ndarray,
    config: TrainingConfig,
    device: str,
    run_dir: str | Path,
    start_iteration: int = 0,
    use_wandb: bool = True,
) -> None:
    """Main training loop for the transformer language model.

    Args:
        model: Initialized transformer model
        optimizer: Initialized optimizer
        training_data: Tokenized training data as numpy array
        val_data: Tokenized validation data as numpy array
        config: Configuration dictionary
        device: Device to train on ('cuda', 'mps', or 'cpu')
        run_dir: Directory for this specific training run (checkpoints will be saved here)
        start_iteration: Iteration to start/resume training from
        use_wandb: Whether to use Weights & Biases for logging
    """
    # Initialize Weights & Biases if available and requested
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.experiment_name,
            config=config.to_dict(),
            resume="allow" if start_iteration > 0 else None,
        )
        # Log model architecture
        wandb.watch(model, log="all", log_freq=config.log_interval)
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("Weights & Biases not available. Install with: pip install wandb")

    model.train()

    best_val_loss = float("inf")

    logger.info(f"Starting training for {config.num_iterations} iterations...")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Context length: {config.context_length}")
    logger.info("-" * 60)

    # Track wallclock time for logging
    start_time = time.time()
    iteration_start_time = start_time

    pbar = tqdm(range(start_iteration, config.num_iterations), desc="Training", unit="iter")

    for iteration in pbar:
        # Update learning rate
        lr = lr_cosine_schedule(
            t=iteration,
            t_warmup=config.warmup_iterations,
            t_cosine=config.num_iterations,
            lr_max=config.learning_rate_max,
            lr_min=config.learning_rate_min,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # load batch
        inputs, labels = data_loader(
            x=training_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
        )

        token_positions = torch.arange(config.context_length, device=device).unsqueeze(0).expand(config.batch_size, -1)
        logits = model(inputs, token_positions=token_positions)
        loss = cross_entropy_loss(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=config.grad_clip_norm)
        optimizer.step()

        # calc. timing metrics for logging
        current_time = time.time()
        wallclock_time = current_time - start_time
        iter_time = current_time - iteration_start_time
        iteration_start_time = current_time

        # calculate perplexity
        train_ppl = calculate_perplexity(loss.item())

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "ppl": f"{train_ppl:.2f}",
                "lr": f"{lr:.6f}",
                "ms/iter": f"{iter_time * 1000:.1f}",
            }
        )

        if (iteration + 1) % config.log_interval == 0:
            logger.info(
                f"Iteration {iteration + 1:>6}/{config.num_iterations} | "
                f"Loss: {loss.item():.4f} | Perplexity: {train_ppl:.2f} | LR: {lr:.6f} | "
                f"Time: {wallclock_time:.2f}s | Iter: {iter_time * 1000:.1f}ms"
            )

            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/perplexity": train_ppl,
                        "train/learning_rate": lr,
                        "train/iteration": iteration + 1,
                        "train/wallclock_time": wallclock_time,
                        "train/iter_time_ms": iter_time * 1000,
                    },
                    step=iteration + 1,
                )

        # Validation across set intervals.
        if (iteration + 1) % config.val_interval == 0:
            val_loss, val_ppl = validate(model, val_data, config, device)
            logger.info(
                f"Iteration {iteration + 1:>6}/{config.num_iterations} | "
                f"Validation Loss: {val_loss:.4f} | Validation Perplexity: {val_ppl:.2f} | "
                f"Time: {wallclock_time:.2f}s"
            )

            # save newbest val. loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = Path(run_dir) / "checkpoint_best.pt"
                save_checkpoint(model, optimizer, iteration + 1, best_checkpoint_path)
                logger.info(f"New best validation loss: {val_loss:.4f} - Saved checkpoint: {best_checkpoint_path}")

            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                        "val/best_loss": best_val_loss,
                        "val/wallclock_time": wallclock_time,
                    },
                    step=iteration + 1,
                )

        # save checkpoint across intervals.
        if (iteration + 1) % config.checkpoint_interval == 0:
            checkpoint_path = Path(run_dir) / f"checkpoint_iter_{iteration + 1}.pt"
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    pbar.close()

    # Save final checkpoint
    final_checkpoint_path = Path(run_dir) / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, config.num_iterations, final_checkpoint_path)
    logger.info(f"Training complete! Final checkpoint saved: {final_checkpoint_path}")
    logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")

    # finish wandb run
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE Transformer language model")
    parser.add_argument(
        "--config",
        type=str,
        default="bpe_transformer/config/yaml/training/TinyStories_17M-2.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data", type=str, default="data/tokenizers/bpe_tinystories/train_tokens.npy", help="Path to training data"
    )
    parser.add_argument(
        "--val-data", type=str, default="data/tokenizers/bpe_tinystories/val_tokens.npy", help="Path to validation data"
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--no-wandb", action="store_false", dest="use_wandb", help="Disable Weights & Biases logging")
    parser.add_argument("--experiment-name", type=str, default=None, help="Name for the experiment (for W&B)")
    args = parser.parse_args()

    setup_logging(log_level="INFO")

    config = TrainingConfig.from_yaml(args.config)
    set_seed(config.seed)

    if args.experiment_name:
        config.experiment_name = args.experiment_name

    run_dir = create_run_directory(args.checkpoint_dir, config)

    # Init model, optimizers etc. for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model, optimizer, training_data, val_data, start_iteration = setup_training(
        config=config,
        data_path=args.data,
        val_data_path=args.val_data,
        device=device,
        resume_from=args.resume_from,
    )

    train(
        model=model,
        optimizer=optimizer,
        training_data=training_data,
        val_data=val_data,
        config=config,
        device=device,
        run_dir=run_dir,
        start_iteration=start_iteration,
        use_wandb=args.use_wandb,
    )
