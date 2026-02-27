import logging
import torch
import yaml
from bpe_transformer.model.transformer_lm import TransformerLM
from bpe_transformer.tokenization.bpe_tokenizer import BPETokenizer
from bpe_transformer.tokenization.bpe_trainer import BPETrainer
from pathlib import Path

logger = logging.getLogger(__name__)

N_WORKERS = 8


def train_bpe(
    input_path: Path, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Check if vocab_size makes sense
    if vocab_size < 255 + len(special_tokens):
        raise ValueError("Input vocab_size is invalid: value too small.")

    bpe = BPETrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    bpe.train(input_path=input_path, n_workers=N_WORKERS)
    bpe.save_trainer(output_dir=Path("output/tokenizer"))
    return bpe.vocab, bpe.merges


def _generate_text(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    max_tokens: int,
    eos_token_id: int,
    p: float,
    prompt: str | None = None,
    top_k: int | None = None,
    temperature: float | None = None,
):
    # Log generation parameters
    logger.info(
        f"prompt='{prompt}' | max_tokens={max_tokens} | temperature={temperature} | p={p} | top_k={top_k} | eos_token_id={eos_token_id} | device={model.device}"
    )

    # 1. If prompt is provided, tokenize it; otherwise start with BOS token (using eos_token_id)
    if prompt is None:
        input_tensor = torch.tensor(
            [eos_token_id] if prompt is None else tokenizer.encode(prompt), dtype=torch.long, device=model.device
        )

    # 2. generate next tokens with TransformerLM.
    generated_tokens = model.generate(
        input_tensor, eos_token_id=eos_token_id, max_tokens=max_tokens, p=p, top_k=top_k, temperature=temperature
    )

    # 3. Lazily decode the generated token IDs.
    if prompt:
        print(prompt, end="")
    print(tokenizer.decode(generated_tokens[0].tolist()))


@torch.no_grad()
def generate_text(
    checkpoint_path: Path,
    tokenizer_config_path: Path,
    max_tokens: int,
    p: float,
    prompt: str | None = None,
    top_k: int | None = None,
    temperature: float | None = None,
    device: str | None = None,
):
    """Generate text from a trained model checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer_config_path: Path to tokenizer config YAML
        max_tokens: Maximum tokens to generate
        p: Nucleus sampling threshold
        prompt: Optional text prompt
        top_k: Optional top-k sampling parameter
        temperature: Optional temperature for sampling
        device: Device to use (cuda/mps/cpu), auto-detected if None
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load model config from checkpoint directory
    checkpoint_dir = checkpoint_path.parent
    model_config_path = checkpoint_dir / "config.yaml"
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Load tokenizer config
    with open(tokenizer_config_path) as f:
        tokenizer_config = yaml.safe_load(f)

    # Initialize model
    from bpe_transformer.model.modules import RoPE

    rope = RoPE(
        theta=model_config["theta"],
        d_k=model_config["d_model"] // model_config["num_heads"],
        max_seq_len=model_config["context_length"],
        device=device,
    )

    transformer_lm = TransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        num_heads=model_config["num_heads"],
        rope=rope,
        device=device,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Remove _orig_mod. prefix from compiled model
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    transformer_lm.load_state_dict(state_dict)
    transformer_lm.to(device)
    transformer_lm.eval()

    # Load tokenizer
    bpe_tokenizer = BPETokenizer.from_files(
        vocab_filepath=Path(tokenizer_config["vocab_filepath"]),
        merges_filepath=Path(tokenizer_config["merges_filepath"]),
        special_tokens=tokenizer_config["special_tokens"],
    )

    # Generate text
    _generate_text(
        model=transformer_lm,
        tokenizer=bpe_tokenizer,
        max_tokens=max_tokens,
        eos_token_id=tokenizer_config["eos_token_id"],
        p=p,
        prompt=prompt,
        top_k=top_k,
        temperature=temperature,
    )


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # import time
    # from datetime import datetime

    # start_time = time.time()
    # print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # vocab, merges = train_bpe(
    #     input_path=Path('data/owt_trainval_tokenizer.txt'),
    #     vocab_size=10000,
    #     special_tokens=['<|endoftext|>']
    # )

    # end_time = time.time()
    # duration = end_time - start_time
    # print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # print(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")

    # ckp_path = "checkpoints/run_20260215_064641_lr0.003_bs32_iters41500/checkpoint_final.pt"
    ckp_path = "checkpoints/checkpoint_final.pt"
    # Generate text
    generate_text(
        checkpoint_path=Path(ckp_path),
        tokenizer_config_path=Path("bpe_transformer/config/yaml/tokenizer/bpe_tinystories.yaml"),
        max_tokens=300,
        p=0.9,
        # prompt="Once upon a time",
        temperature=0.8,
    )
