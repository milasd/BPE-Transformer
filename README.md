# BPE Transformer: Byte-Pair Encoding, Transfomer LLM training & inference

Implementation of the Byte-Pair Encoding tokenizer (training, encoding and decoding), a Transformer-based LLM architecture w/ RoPE Embeddings, SwiGLU and AdamW optimizer (Muon: in progress) from scratch with PyTorch. Train the LM & generate text with a trained model and tokenizer from the experiment. 

*Recently added: Ported PyTorch model/optimizer/trainer to* **MLX** *for personal experiments.*


## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
  - [Byte-Pair Encoding Tokenizer](#byte-pair-encoding-tokenizer)
  - [Transformer Model](#transformer-model)
- [Dataset Preprocessing](#dataset-preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Demo Notebooks](#demo-notebooks)
- [Testing](#testing)


---

```
bpe_transformer/
    ├── config/             # Default configuration files
    │   ├── training.py     # TrainingConfig class
    │   └── yaml/           # YAML files
    │       ├── training/   # Training hyperparameters
    │       ├── dataset/    # Dataset preprocessing
    │       └── tokenizer/  # Tokenization settings
    ├── tokenization/       # BPE tokenizer implementation
    ├── model/              # TransformerLM
    ├── optimizer/          # AdamW optimizer and training utilities
    ├── training/           # Training scripts and utilities
    └── inference/          # Text generation and inference

notebooks/             # Jupyter notebooks for demonstrations
tests/                 # Test suite
data/                  # Dataset directory
checkpoints/           # Model checkpoints from training
```
---

## Setup

### Environment
Install `uv` [here](https://github.com/astral-sh/uv) for package management.

You can try installing the environment:
```sh
uv sync
```

To run python scripts, use the command below:
```sh
uv run <python_file_path>
```

### Download datasets
Download the TinyStories data:

``` sh
mkdir -p data
cd data

curl -L -o TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -o TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

## Usage

### Byte-Pair Encoding Tokenizer

The implementation for the BPE Tokenizer can be found in `bpe_transformer/tokenization`.

```
bpe_transformer/
  └── tokenization/
      ├── preprocessing/                # Data pre-processing scripts
      │   ├── __init__.py
      │   └── pretokenization.py        # Pre-tokenization scripts (following GPT-2)
      ├── __init__.py
      ├── bpe_trainer.py                # Trainer for BPE
      ├── bpe_tokenizer.py              # BPE tokenizer class with encode/decode methods etc.
      └── tokenizer.py                  # Base tokenizer interface
```

#### **Training**

Examples: 

```python
from bpe_transformer.tokenization import train_bpe

# Train tokenizer on your corpus
vocab, merges = train_bpe(
                            corpus_path="data/TinyStoriesV2-GPT4-valid.txt",
                            vocab_size=10000,
                            special_tokens=["<|endoftext|>"]
                         )
```
or

```python
from bpe_transformer.tokenization import BPETrainer

input_path = Path("../data/TinyStoriesV2-GPT4-train.txt")
special_tokens = ["<|endoftext|>"]
output_dir = "."

if __name__ == "__main__":
    # Create BPE Trainer
    bpe = BPETrainer(vocab_size=10000, special_tokens=special_tokens)
    # Train BPE
    bpe.train(input_path=input_path, n_workers=4)
    # Serialize the resulting vocab and merge
    bpe.save_trainer(output_dir=output_dir)

    print(f"Saved vocab and merges to {output_dir}.")
    print("Vocab:")
    print(bpe.vocab)
    print("Merges:")
    print(bpe.merges)
```

---
#### **Encoding and Decoding**

Examples:

```python
from bpe_transformer.tokenization.bpe_tokenizer import BPETokenizer
from tests.common import FIXTURES_PATH

input_dir = Path("notebooks/sample_data/bpe_tokenizer")

# Load Tokenizer from vocab and merge files (obtained post-training on TinyStories)
bpe = BPETokenizer.from_files(
    vocab_filepath=Path(input_dir / "vocab.pkl"), merges_filepath=Path(input_dir / "merges.pkl")
)

# Encode text to token IDs
text = "Once upon a time, there was a little fairy."
token_ids = tokenizer.encode(text)

# Decode token IDs back to text
decoded_text = tokenizer.decode(token_ids)

# Lazy encode for larger text file streams
corpus_path = FIXTURES_PATH / "tinystories_sample.txt"

ids = []
with open(corpus_path) as f:
    for _ids in bpe.encode_iterable(f):
        ids.append(_ids)

print(ids)
print(len(ids))
```

---

### Transformer Model

The transformer implementation can be found in `bpe_transformer/model`.

```
bpe_transformer/
  └── model/
      ├── modules/                      # Core building blocks
      │   ├── __init__.py
      │   ├── embedding.py              # Token embeddings
      │   ├── linear.py                 # Bias-free linear layer
      │   ├── rms_norm.py               # RMSNorm layer
      │   ├── rope.py                   # Rotary Position Embeddings
      │   ├── scaled_dot_product_attention.py
      │   ├── multihead_self_attention.py
      │   ├── swiglu.py                 # SwiGLU feedforward
      │   └── transformer_block.py      # Transformer block
      ├── __init__.py
      └── transformer_lm.py             # TransfomerLM Large Language Model
```

#### **Usage**

```python
from bpe_transformer.model import TransformerLM
from bpe_transformer.model.modules import RoPE

# Create RoPE embeddings
rope = RoPE(theta=10000.0, d_k=64, max_seq_len=2048)

# Initialize model
model = TransformerLM(
    vocab_size=10000,
    context_length=2048,
    num_layers=12,
    d_model=768,
    d_ff=2048,
    num_heads=12,
    rope=rope
)

# Forward pass
import torch
token_ids = torch.randint(0, 10000, (2, 128))  # (batch, seq_len)
token_positions = torch.arange(128)
logits = model(token_ids, token_positions)  # (batch, seq_len, vocab_size)
```

## Dataset Preprocessing

Before training, you need to tokenize your dataset. The training script expects tokenized data as `.npy` files containing token IDs.

Preprocess your dataset using the provided script:

```sh
uv run python bpe_transformer/training/utils/dataset_preprocessing.py --config path/to/config.yaml
```

The script defaults `--config` to the TinyStories dataset configuration at `bpe_transformer/config/yaml/dataset/preprocessing_tinystories.yaml`. You can modify this config file according to your data paths, or create new configuration files following the same pattern.

## Training

Train a LLM on the tokenized dataset:

```sh
uv run bpe_transformer/training/train.py \
  --config path/to/config.yaml \
  --data path/to/train_tokens.npy \
  --val-data path/to/val_tokens.npy
```

#### Args

- `--config`: Path to model dims and training hyperparameter configuration YAML file
- `--data`: Path to tokenized training data (`.npy` file)
- `--val-data`: Path to tokenized validation data  (`.npy` file)
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints`)
- `--resume-from`: (Optional) Path to checkpoint to resume training from 
- `--no-wandb`: Disable Weights & Biases logging
- `--experiment-name`: Name for the experiment (for W&B tracking)

#### Examples


Resume from checkpoint:
```sh
uv run bpe_transformer/training/train.py --resume-from checkpoints/checkpoint_iter_5000.pt
```

## Inference

Generate text from a trained model checkpoint:

```sh
uv run bpe_transformer/inference/generate.py \
  --checkpoint path/to/checkpoint.pt \
  --tokenizer-config path/to/tokenizer_config.yaml \
  --prompt "Once upon a time"
```

#### Args

- `--checkpoint`: Path to model checkpoint (default: `checkpoints/checkpoint_final.pt`)
- `--tokenizer-config`: Path to tokenizer config YAML (default: `bpe_transformer/config/yaml/tokenizer/bpe_tinystories.yaml`)
- `--max-tokens`: Maximum tokens to generate (default: `300`)
- `--p`: Nucleus sampling threshold (default: `0.9`)
- `--prompt`: Text prompt (optional, generates from scratch if not provided)
- `--top-k`: Top-k sampling parameter (optional)
- `--temperature`: Sampling temperature (default: `0.8`)
- `--device`: Device to use - cuda/mps/cpu (auto-detected if not specified)

#### Examples

Generate with custom prompt:
```sh
uv run bpe_transformer/inference/generate.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --prompt "Once upon a time" \
  --temperature 0.7 \
  --max-tokens 200
```

Generate from scratch (no prompt):
```sh
uv run bpe_transformer/inference/generate.py \
  --checkpoint checkpoints/checkpoint_final.pt \
  --temperature 1.0 \
  --max-tokens 500
```

## Demo Notebooks

If you don't wish to run the Python scripts, but would like to see demonstrations on some of the implemented concepts, there are some iPython notebooks inside `notebooks`, which you can also try to run in your own personal computer.

```
notebooks/
├── 1_pretokenization.ipynb                    # (GPT-2) Regex-based pre-tokenization step
├── 2_bpe_tokenization_training.ipynb          # Train BPE tokenizer from scratch
└── 3_bpe_tokenization_encode_decode.ipynb     # Encoding and decoding with BPE
```

## Testing

Run the test suite with:
```sh
uv run pytest
```

To run a specific test file only:

```sh
uv run pytest test/test_tokenizer.py
```

To run a specific function from a specific file:

```sh
uv run pytest tests/test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens
```

Add `-v` or `--v` in the command line to debug errors.
