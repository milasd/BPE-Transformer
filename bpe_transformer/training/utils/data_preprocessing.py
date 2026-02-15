"""Preprocess TinyStories dataset by tokenizing and saving as numpy arrays."""

import logging
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from bpe_transformer.tokenization.bpe_tokenizer import BPETokenizer


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for preprocessing."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def preprocess_dataset(
    tokenizer: BPETokenizer,
    input_file: str | Path,
    output_file: str | Path,
    n_workers: int | None = None,
) -> None:
    """Tokenize a text file and save as numpy array.

    Args:
        tokenizer: BPE tokenizer instance
        input_file: Path to input text file
        output_file: Path to output .npy file
        n_workers: Number of parallel workers for encoding (None = auto-detect)
    """
    # Auto-detect workers if not specified
    if n_workers is None:
        n_workers = os.cpu_count() or 8

    logger.info(f"Processing {input_file}...")
    logger.info(f"Using {n_workers} workers")

    # Get file size for progress tracking
    file_size = os.path.getsize(input_file)
    logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB")

    # Estimate total tokens (rough estimate: ~4 characters per token for BPE)
    estimated_tokens = file_size // 4
    logger.info(f"Estimated tokens: ~{estimated_tokens / 1e6:.1f}M")

    # Read and encode the file with progress bar
    with open(input_file, "r", encoding="utf-8") as f:
        tokens = list(tqdm(
            tokenizer.encode_iterable(f, n_workers=n_workers),
            total=estimated_tokens,
            desc="Tokenizing",
            unit=" tokens",
            unit_scale=True,
        ))

    # Convert to numpy array
    tokens_array = np.array(tokens, dtype=np.int32)

    # Save to file
    np.save(output_file, tokens_array)

    logger.info(f"Saved {len(tokens_array):,} tokens to {output_file}")
    logger.info(f"Array shape: {tokens_array.shape}, dtype: {tokens_array.dtype}")


def main(
    vocab_path: str | Path,
    merges_path: str | Path,
    train_text_path: str | Path,
    val_text_path: str | Path,
    train_output_path: str | Path,
    val_output_path: str | Path,
    special_tokens: list[str] | None = None,
    n_workers: int | None = None,
) -> None:
    """Main preprocessing function.

    Args:
        vocab_path: Path to vocab.pkl file
        merges_path: Path to merges.pkl file
        train_text_path: Path to training text file
        val_text_path: Path to validation text file
        train_output_path: Path to save training tokens .npy
        val_output_path: Path to save validation tokens .npy
        special_tokens: Optional list of special tokens
        n_workers: Number of parallel workers for encoding (None = auto-detect)
    """
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=Path(vocab_path),
        merges_filepath=Path(merges_path),
        special_tokens=special_tokens,
    )
    logger.info(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")

    # Process validation data
    logger.info("\n" + "=" * 60)
    logger.info("Processing validation data")
    logger.info("=" * 60)
    preprocess_dataset(
        tokenizer=tokenizer,
        input_file=val_text_path,
        output_file=val_output_path,
        n_workers=n_workers,
    )

    # # Process training data
    # logger.info("\n" + "=" * 60)
    # logger.info("Processing training data")
    # logger.info("=" * 60)
    # preprocess_dataset(
    #     tokenizer=tokenizer,
    #     input_file=train_text_path,
    #     output_file=train_output_path,
    #     n_workers=n_workers,
    # )

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging(log_level="INFO")

    # Paths configuration
    tokenizer_dir = Path("data/tokenizers/bpe_tinystories")
    data_dir = Path("data")

    vocab_path = tokenizer_dir / "vocab.pkl"
    merges_path = tokenizer_dir / "merges.pkl"

    train_text_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    val_text_path = data_dir / "TinyStoriesV2-GPT4-valid.txt"

    train_output_path = data_dir / "train_tokens.npy"
    val_output_path = tokenizer_dir / "val_tokens.npy"

    special_tokens = ["<|endoftext|>"]
    n_workers = 8

    main(
        vocab_path=vocab_path,
        merges_path=merges_path,
        train_text_path=train_text_path,
        val_text_path=val_text_path,
        train_output_path=train_output_path,
        val_output_path=val_output_path,
        special_tokens=special_tokens,
        n_workers=n_workers,
    )
