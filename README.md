# BPE Transformer: Byte-Pair Encoding for Transformers

Implementation of the Byte-Pair Encoding tokenizer (training, encoding and decoding), a Transformer architecture w/ RoPE Embeddings, SwiGLU and AdamW optimizer from scratch.


```
bpe_transformer/
    ├── tokenization/       # BPE tokenizer implementation
    ├── transformer/        # Transformer model implementation
    └── settings.py         # Configuration settings

notebooks/             # Jupyter notebooks for demonstrations
tests/                 # Test suite
data/                  # Dataset directory (suggestion)
```


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

## Demo Notebooks

If you don't wish to run the Python scripts, but would like to see demonstrations on some of the implemented concepts, there are some iPython notebooks inside `notebooks`, which you can also try to run in your own personal computer.

```
notebooks/
├── 1_pretokenization.ipynb                    # (GPT-2) Regex-based pre-tokenization step
├── 2_bpe_tokenization_training.ipynb          # Train BPE tokenizer from scratch
└── 3_bpe_tokenization_encode_decode.ipynb     # Encoding and decoding with BPE
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