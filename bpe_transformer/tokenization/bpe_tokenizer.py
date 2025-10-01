from collections import Counter
from multiprocessing import cpu_count
from pathlib import Path
from .tokenizer import Tokenizer
from .preprocessing import parallel_pretokenization


"""
PS
the current pre-tokenization will only remove one special token. todo later--add support to multiple special words
"""

N_WORKERS = cpu_count()

def train_bpe(
    input_path: Path, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Check if vocab_size makes sense
    if vocab_size < 255 + len(special_tokens):
        raise ValueError("Input vocab_size is invalid: value too small.")

    bpe = BPETokenizer(vocab_size=vocab_size, special_tokens=special_tokens)
    bpe.train(input_path=input_path, num_processes=N_WORKERS)
    return bpe.vocab, bpe.merges


class BPETokenizer(Tokenizer):
    """
    Implementation of a greedy Byte-Pair Encoding Tokenizer.
    """

    def __init__(self, vocab_size: int, special_tokens: list[str]):
        # We can keep track of the most frequent pairs with a heap.
        # self._vocab_cache: dict[bytes, int] = {bytes[i]: i for i in range(256)}
        self._merges: list[tuple[bytes, bytes]] = []
        self.special_tokens: set = set(special_tokens)
        self.encoding = "utf-8"

        # initialize vocab: 256 bytes from 0 to 255, plus end of token if there's one.
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # If there are any special tokens, add to vocab. Check if i might be adding it twice...
        if self.special_tokens:
            for t in self.special_tokens:
                self.add_new_vocab(t.encode(encoding=self.encoding))

    @property
    def vocab(self) -> dict[int, bytes]:
        return self._vocab

    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        return self._merges

    def add_new_vocab(self, new_value: bytes) -> None:
        """
        Add new token to the BPE vocab dict.

        Args:
            new_value: The new token to be registered, as bytes.
        """
        self._vocab[len(self._vocab)] = new_value
        # self._vocab_cache[new_value] = len(self._vocab)

    def train(self, input_path: Path, num_processes: int | None) -> None:
        """
        Train the BPE Tokenizer on an input file.
        """
        # Invoke pre-tokenization of input file
        pretoken_counter = self._get_pretokenization(input_path, num_processes)

        # Merge pairs of bytes

    def _merge_tokens(self, pretoken_counter: Counter) -> None:
        """
        Given a dict of pretokens, merge
        """

    def _get_pretokenization(self, input_path: Path, num_processes: int | None) -> Counter:
        """
        Call parallel pretokenization for a given input file.

        Args:
            input_path: Path to the data file to pretokenize.

        Return:
            Counter object with pre-token ocurrences in an input file.
        """
        pretoken_counter = parallel_pretokenization(file_path=input_path, num_processes=num_processes)
        return pretoken_counter
