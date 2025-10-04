from pathlib import Path
from multiprocessing import cpu_count
from bpe_transformer.tokenization.bpe_tokenizer import BPETokenizer

if __name__ == "__main__":
    input_path = Path("/Users/miladeoliveira/projects/BPE-transformer/tests/fixtures/corpus.en")
    n_cpus = cpu_count()
    bpe = BPETokenizer(500, ["<|endoftext|>"])
    bpe.train(input_path, n_cpus)
    print(bpe.vocab)
    print(bpe.merges)
