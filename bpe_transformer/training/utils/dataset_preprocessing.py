import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bpe_transformer.tokenization.bpe_tokenizer import BPETokenizer

"""Preprocess TinyStories/OWT dataset by tokenizing and saving as numpy arrays."""

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for preprocessing."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load preprocessing configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def preprocess_dataset(
    tokenizer: BPETokenizer,
    input_file: str | Path,
    output_file: str | Path,
    n_workers: int | None = None,
    dtype: np.dtype = np.int32,
) -> None:
    """Tokenize a text file and save as numpy array using memory-efficient streaming.

    Args:
        tokenizer: BPE tokenizer instance
        input_file: Path to input text file
        output_file: Path to output .npy file
        n_workers: Number of parallel workers for encoding (None = auto-detect)
        dtype: NumPy dtype for the array (default: np.int32)
    """
    # Auto-detect workers if not specified
    if n_workers is None:
        n_workers = os.cpu_count() or 8

    logger.info(f"Processing {input_file}...")
    logger.info(f"Using {n_workers} workers")

    # Get file size for info
    file_size = os.path.getsize(input_file)
    logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB")

    # Estimate total tokens (rough estimate: ~4 characters per token for BPE)
    estimated_tokens = file_size // 4
    logger.info(f"Estimated tokens: ~{estimated_tokens / 1e6:.1f}M")

    # Encode and save using memory-efficient streaming
    with open(input_file, "r", encoding="utf-8") as f:
        tokenizer.encode_iterable_to_npy(
            iterable=f,
            output_path=Path(output_file),
            n_workers=n_workers,
            dtype=dtype,
        )

    logger.info("Encoding complete!")


def main(config_path: str | Path) -> None:
    """Main preprocessing function.

    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    logger.info(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # Convert dtype string to numpy dtype
    dtype = getattr(np, config.get("dtype", "int32"))

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=Path(config["vocab_filepath"]),
        merges_filepath=Path(config["merges_filepath"]),
        special_tokens=config.get("special_tokens"),
    )
    logger.info(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")

    # Process training data
    logger.info("\n" + "=" * 60)
    logger.info("Processing training data")
    logger.info("=" * 60)
    preprocess_dataset(
        tokenizer=tokenizer,
        input_file=Path(config["train_text_path"]),
        output_file=Path(config["train_output_path"]),
        n_workers=config.get("n_workers"),
        dtype=dtype,
    )

    # Process validation data
    logger.info("\n" + "=" * 60)
    logger.info("Processing validation data")
    logger.info("=" * 60)
    preprocess_dataset(
        tokenizer=tokenizer,
        input_file=Path(config["val_text_path"]),
        output_file=Path(config["val_output_path"]),
        n_workers=config.get("n_workers"),
        dtype=dtype,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging(log_level="INFO")

    config_path = Path("config/dataset/preprocessing_openwebtext.yaml")

    # Dataset preprocessing for OWT
    main(config_path=config_path)
