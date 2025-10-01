# BPE Tokenization: Byte-Pair Encoding for Transformers

Implementation of the Byte-Pair Encoding tokenizer from scratch using some PyTorch basic functions. 

When the BPE implementation is finished, I will implement from scratch a simple Transformer (+ RoPE, AdamW optimizer etc).


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

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

### Demos

If you don't wish to run the Python scripts, but would like to see demonstrations on some of the implemented concepts, there are some iPython notebooks inside `notebooks`, which you can also try to run in your own personal computer.