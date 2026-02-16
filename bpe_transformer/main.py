import torch
from bpe_transformer.model.transformer_lm import TransformerLM
from bpe_transformer.tokenization.bpe_tokenizer import BPETokenizer
from bpe_transformer.tokenization.bpe_trainer import BPETrainer
from multiprocessing import cpu_count
from pathlib import Path

N_WORKERS = cpu_count() - 4


def train_bpe(
    input_path: Path, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Check if vocab_size makes sense
    if vocab_size < 255 + len(special_tokens):
        raise ValueError("Input vocab_size is invalid: value too small.")

    bpe = BPETrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    bpe.train(input_path=input_path, n_workers=N_WORKERS)
    return bpe.vocab, bpe.merges


def _generate_text(model: TransformerLM, tokenizer: BPETokenizer, max_tokens: int, eos_token_id: int, p: float, prompt: str | None = None, top_k: int | None = None, temperature: float | None = None):
    # 1. If prompt is provided, tokenize it.
    input_tensor = torch.empty((0,), dtype=torch.long, device=model.device) if prompt is None else torch.tensor(tokenizer.encode(prompt), device=model.device, dtype=torch.long)
    print(input_tensor)
    
    # 2. generate next tokens with TransformerLM.
    generated_tokens = model.generate(input_tensor, eos_token_id=eos_token_id, max_tokens=max_tokens, p=p, top_k=top_k, temperature=temperature)
    
    # 3. Lazily decode the generated token IDs.
    if prompt:
        print(prompt)
    
    print(tokenizer.decode(generated_tokens[0].tolist()))
    
    
def generate_text(model_path: Path, merges_path: Path, vocab_path: Path, max_tokens: int, eos_token_id: int, p: float, prompt: str | None = None, top_k: int | None = None, temperature: float | None = None):
    # Load tokenizer
    bpe_tokenizer = BPETokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=["<|endoftext|>"])
    # Load model from file
    transformer_lm = torch.load(model_path)
    
    # Generate text
    _generate_text(model=transformer_lm, tokenizer=bpe_tokenizer, max_tokens=max_tokens, eos_token_id=eos_token_id, p=p, prompt=prompt, top_k=top_k, temperature=temperature)
    
    # Save somewhere later?
    
    