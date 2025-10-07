from bpe_transformer.tokenization.bpe_trainer import BPETrainer
from multiprocessing import cpu_count
from pathlib import Path

N_WORKERS = cpu_count()


def train_bpe(
    input_path: Path, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Check if vocab_size makes sense
    if vocab_size < 255 + len(special_tokens):
        raise ValueError("Input vocab_size is invalid: value too small.")

    bpe = BPETrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    bpe.train(input_path=input_path, num_processes=N_WORKERS)
    return bpe.vocab, bpe.merges