import argparse
import logging
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import random
import time
import yaml

from bpe_transformer.config import TrainingConfig
from bpe_transformer.model.mlx import RoPE, TransformerLM
from bpe_transformer.optimizer.mlx_adamw import AdamW
from datetime import datetime
from einops import rearrange
from pathlib import Path
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
    mx.random.seed(seed)

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


def cross_entropy_loss(logits: mx.array, target: mx.array) -> mx.array:
    """Apply cross-entropy loss on logits without explicitly calling softmax function."""
    logits_flat = rearrange(logits, "... vocab_size -> (...) vocab_size")
    target_flat = rearrange(target, "... -> (...)")

    # 1. Subtract max value from logits for numerical stability
    max_val = mx.max(logits_flat, axis=-1, keepdims=True)
    stable_logits = logits_flat - max_val

    # 2. calculate max - log(sum(exp(stable_logits)))
    sum_exp = mx.sum(mx.exp(stable_logits), axis=-1, keepdims=True)
    log_sum_exp = mx.log(sum_exp)

    log_probs = stable_logits - log_sum_exp

    # 3. Compare to target: get probability
    batch_indices = mx.arange(logits_flat.shape[0])
    log_prob_targets = log_probs[batch_indices, target_flat]

    loss = -mx.mean(log_prob_targets)

    return loss


def lr_cosine_schedule(t: int, t_warmup: int, t_cosine: int, lr_max: float, lr_min: float):
    """Cosine learning rate schedule with warmup."""
    # Warm-up
    if t < t_warmup:
        return lr_max * t / t_warmup
    # Post-annealing
    if t > t_cosine:
        return lr_min
    # Cosine annealing
    return lr_min + (0.5 * (1 + math.cos((t - t_warmup) * math.pi / (t_cosine - t_warmup))) * (lr_max - lr_min))


def init_model(config: TrainingConfig) -> nn.Module:
    """Initialize the transformer language model."""
    rope = RoPE(
        theta=config.theta,
        d_k=config.d_model // config.num_heads,
        max_seq_len=config.context_length,
    )

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_heads=config.num_heads,
        rope=rope,
    )

    return model


def data_loader(
    x: np.ndarray, batch_size: int, context_length: int
) -> tuple[mx.array, mx.array]:
    """Load a random batch of data.

    Args:
        x: Input data (memory-mapped numpy array)
        batch_size: Number of sequences in batch
        context_length: Length of each sequence

    Returns:
        Tuple of (inputs, labels) as MLX arrays
    """
    # Sample random starting positions
    max_start = len(x) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    # Load sequences
    inputs = np.array([x[i : i + context_length] for i in starts])
    labels = np.array([x[i + 1 : i + context_length + 1] for i in starts])

    return mx.array(inputs), mx.array(labels)


def load_data(data_path: str | Path, val_data_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load training and validation data using memory-mapped files.

    Uses memory-mapped mode to lazily load data on-demand, avoiding loading
    the entire dataset into RAM.

    Args:
        data_path: Path to training data (tokenized numpy array)
        val_data_path: Path to validation data (tokenized numpy array)

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


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, iteration: int, path: str | Path) -> None:
    """Save model and optimizer state to checkpoint file."""
    checkpoint = {
        "model": model.parameters(),
        "optimizer": optimizer.state,
        "iteration": iteration,
    }
    mx.save_safetensors(str(path), checkpoint)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: optim.Optimizer) -> int:
    """Load model and optimizer state from checkpoint file.

    Returns:
        Iteration number from checkpoint
    """
    checkpoint = mx.load(str(path))
    model.update(checkpoint["model"])
    optimizer.state = checkpoint["optimizer"]
    return checkpoint["iteration"]


def loss_fn(model: nn.Module, inputs: mx.array, labels: mx.array, token_positions: mx.array) -> mx.array:
    """Forward pass and loss calculation."""
    logits = model(inputs, token_positions=token_positions)
    return cross_entropy_loss(logits, labels)


def setup_training(
    config: TrainingConfig,
    data_path: str | Path,
    val_data_path: str | Path,
    resume_from: str | Path | None = None,
) -> tuple[nn.Module, optim.Optimizer, np.ndarray, np.ndarray, int]:
    """Setup all components needed for training.

    Args:
        config: Configuration dictionary
        data_path: Path to training data (tokenized numpy array)
        val_data_path: Path to validation data (tokenized numpy array)
        resume_from: Optional path to checkpoint to resume training from

    Returns:
        Tuple of (model, optimizer, training_data, val_data, start_iteration)
    """
    # Load data
    training_data, val_data = load_data(data_path, val_data_path)

    # init. model
    logger.info("Initializing model...")
    model = init_model(config)

    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += count_params(v)
            elif isinstance(v, list):
                total += sum(count_params(item) if isinstance(item, dict) else item.size for item in v)
            else:
                total += v.size
        return total

    num_params = count_params(model.parameters())
    embedding_params = count_params(model.token_embedding.parameters())
    non_embedding_params = num_params - embedding_params
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"  - Embedding parameters: {embedding_params:,}")
    logger.info(f"  - Non-embedding parameters: {non_embedding_params:,}")

    # Initialize optimizer
    optimizer = AdamW(
        learning_rate=config.learning_rate_max,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

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
    num_batches: int = 50,
) -> tuple[float, float]:
    """Run validation and return average loss and perplexity.

    Args:
        model: Model to validate
        val_data: Validation data
        config: Configuration dictionary
        num_batches: Number of batches to validate on

    Returns:
        Tuple of (average validation loss, perplexity)
    """
    total_loss = 0.0

    for _ in range(num_batches):
        # Load batch
        inputs, labels = data_loader(
            x=val_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
        )

        # Create position indices for RoPE
        token_positions = mx.broadcast_to(
            mx.arange(config.context_length)[None, :], (config.batch_size, config.context_length)
        )

        # Forward pass
        logits = model(inputs, token_positions=token_positions)
        loss = cross_entropy_loss(logits, labels)

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)
    return avg_loss, perplexity


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    training_data: np.ndarray,
    val_data: np.ndarray,
    config: TrainingConfig,
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
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("Weights & Biases not available. Install with: pip install wandb")

    best_val_loss = float("inf")

    logger.info(f"Starting training for {config.num_iterations} iterations...")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Context length: {config.context_length}")
    logger.info("-" * 60)

    # Track wallclock time for logging
    start_time = time.time()
    iteration_start_time = start_time

    # Create loss and gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

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

        optimizer.learning_rate = lr

        # load batch
        inputs, labels = data_loader(
            x=training_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
        )

        token_positions = mx.broadcast_to(
            mx.arange(config.context_length)[None, :], (config.batch_size, config.context_length)
        )

        # Forward and backward pass
        loss, grads = loss_and_grad_fn(model, inputs, labels, token_positions)

        # Gradient clipping
        if config.grad_clip_norm > 0:
            grads, grad_norm = optim.clip_grad_norm(grads, max_norm=config.grad_clip_norm)

        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

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
            val_loss, val_ppl = validate(model, val_data, config)
            logger.info(
                f"Iteration {iteration + 1:>6}/{config.num_iterations} | "
                f"Validation Loss: {val_loss:.4f} | Validation Perplexity: {val_ppl:.2f} | "
                f"Time: {wallclock_time:.2f}s"
            )

            # save newbest val. loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = Path(run_dir) / "checkpoint_best.safetensors"
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
            checkpoint_path = Path(run_dir) / f"checkpoint_iter_{iteration + 1}.safetensors"
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    pbar.close()

    # Save final checkpoint
    final_checkpoint_path = Path(run_dir) / "checkpoint_final.safetensors"
    save_checkpoint(model, optimizer, config.num_iterations, final_checkpoint_path)
    logger.info(f"Training complete! Final checkpoint saved: {final_checkpoint_path}")
    logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")

    # finish wandb run
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE Transformer language model with MLX")
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

    model, optimizer, training_data, val_data, start_iteration = setup_training(
        config=config,
        data_path=args.data,
        val_data_path=args.val_data,
        resume_from=args.resume_from,
    )

    train(
        model=model,
        optimizer=optimizer,
        training_data=training_data,
        val_data=val_data,
        config=config,
        run_dir=run_dir,
        start_iteration=start_iteration,
        use_wandb=args.use_wandb,
    )
