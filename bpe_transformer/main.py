"""Utility functions for BPE Transformer."""

from pathlib import Path

from bpe_transformer.tokenization.bpe_trainer import BPETrainer

N_WORKERS = 8


def train_bpe(
    input_path: Path, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on a text corpus.

    Args:
        input_path: Path to input text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to add

    Returns:
        Tuple of (vocab dict, merges list)
    """
    # Check if vocab_size makes sense
    if vocab_size < 255 + len(special_tokens):
        raise ValueError("Input vocab_size is invalid: value too small.")

    bpe = BPETrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    bpe.train(input_path=input_path, n_workers=N_WORKERS)
    bpe.save_trainer(output_dir=Path("output/tokenizer"))
    return bpe.vocab, bpe.merges
