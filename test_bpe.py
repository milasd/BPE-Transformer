from pathlib import Path
from multiprocessing import cpu_count
from bpe_transformer.tokenization.bpe_tokenizer import BPETokenizer

if __name__ == '__main__':
    input_path = Path("data/TinyStoriesv2-val-test.txt")
    n_cpus = cpu_count()
    bpe = BPETokenizer(1000, ["<|endoftext|>"])
    bpe.train(input_path, n_cpus)
    print(bpe.vocab)
    print(bpe.merges)