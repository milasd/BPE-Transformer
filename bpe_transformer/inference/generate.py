"""Text generation script for BPE Transformer."""

import argparse
import logging
from pathlib import Path

import torch
import yaml

from bpe_transformer.model.torch.modules import RoPE
from bpe_transformer.model.torch.transformer_lm import TransformerLM
from bpe_transformer.tokenization.bpe_tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


@torch.no_grad()
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
    """Internal function to generate text."""
    # Log generation parameters
    logger.info(
        f"prompt='{prompt}' | max_tokens={max_tokens} | temperature={temperature} | p={p} | top_k={top_k} | eos_token_id={eos_token_id} | device={model.device}"
    )

    # 1. If prompt is provided, tokenize it; otherwise start with BOS token (using eos_token_id)
    input_tensor = torch.tensor(
        [eos_token_id] if prompt is None else tokenizer.encode(prompt), dtype=torch.long, device=model.device
    )

    # 2. Generate next tokens with TransformerLM
    generated_tokens = model.generate(
        input_tensor, eos_token_id=eos_token_id, max_tokens=max_tokens, p=p, top_k=top_k, temperature=temperature
    )

    # 3. Decode the generated token IDs
    generated_text = tokenizer.decode(generated_tokens[0].tolist())

    if prompt:
        logger.info(f"Generated text:\n{prompt}{generated_text}")
    else:
        logger.info(f"Generated text:\n{generated_text}")


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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    checkpoint_dir = checkpoint_path.parent
    model_config_path = checkpoint_dir / "config.yaml"
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    with open(tokenizer_config_path) as f:
        tokenizer_config = yaml.safe_load(f)

    # Init model.
    # TODO: parse model_config etc Config class instead of accessing keys
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
    parser = argparse.ArgumentParser(description="Generate text from a trained BPE Transformer model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=str,
        default="bpe_transformer/config/yaml/tokenizer/bpe_tinystories.yaml",
        help="Path to tokenizer config YAML",
    )
    parser.add_argument("--max-tokens", type=int, default=300, help="Maximum tokens to generate")
    parser.add_argument("--p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt (optional)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling parameter (optional)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu, auto-detected if None)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    generate_text(
        checkpoint_path=Path(args.checkpoint),
        tokenizer_config_path=Path(args.tokenizer_config),
        max_tokens=args.max_tokens,
        p=args.p,
        prompt=args.prompt,
        top_k=args.top_k,
        temperature=args.temperature,
        device=args.device,
    )
