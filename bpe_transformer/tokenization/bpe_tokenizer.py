import heapq

from collections import Counter
from pathlib import Path
from .tokenizer import Tokenizer
from .preprocessing import parallel_pretokenization


class BPETokenizer(Tokenizer):
    """
    Implementation of a greedy Byte-Pair Encoding Tokenizer.
    """

    def __init__(self, vocab_size: int, special_tokens: list[str]):
        if vocab_size < 256:
            raise ValueError("Invalid Vocab size: must be at least 256")
        
        self.encoding = "utf-8"
        self._vocab_size = vocab_size

        # We can keep track of the most frequent pairs with a heap.
        self._vocab_cache: list[int, tuple[bytes]]= []
        self._merges: list[tuple[bytes, bytes]] = []
        self._special_tokens: set = set(special_tokens)
        self._vocab = self._build_initial_vocab()

    @property
    def vocab(self) -> dict[int, bytes]:
        """Return the vocabulary."""
        return self._vocab

    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        return self._merges

    @property
    def special_tokens(self) -> set[str]:
        return self._special_tokens
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _build_initial_vocab(self) -> dict[int, bytes]:
        """
        Build initial vocabulary with 256 base bytes and special tokens.

        Returns:
            Dictionary mapping token IDs to byte values.
        """
        # Base vocabulary: 256 bytes (0-255)
        vocab = {i: bytes([i]) for i in range(256)}
        
        # Add special tokens
        for idx, token in enumerate(self.special_tokens, start=256):
            vocab[idx] = token.encode(self.encoding)
            
        return vocab

    def add_new_vocab_to_dict(self, vocab_dict: dict[int, bytes], new_value: bytes) -> None:
        """
        Add new token to a vocab dictionary.

        Args:
            vocab_dict: The vocabulary dictionary to update.
            new_value: The new token to be registered, as bytes.
        """
        vocab_dict[len(vocab_dict)] = new_value

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
        self._merge_tokens(pretoken_counter)

    def _merge_tokens(self, pretoken_counter: Counter[tuple[bytes], int]) -> None:
        """
        Given a dict of pretokens, merge pairs
        """
        # Pre-token dict is in form {list[bytes]: count}.
        # pre-token: {'low': 3, 'high': 2}
        # ex.: {(l, o, w): 3, (h, i, g, h): 2}
        # What I want to do is create a cache with counter of pairs.
        # i have to iterate until there are no potential pairs anymore

        #1. Create initial cache of pairs:
        # {(l, o): 3, (o, w): 3, (h, i): 2, (i, g): 2, (g, h): 2}
        #2. merge bytes for vocab: 
        for pretoken, count in pretoken_counter.items():
            # multiplies count to -1 to have a max heap.
            vocab_counter = (-1) * count

            # Edge case: only a single byte/pair in pretoken -- no merge.
            if len(pretoken) == 1 or len(pretoken) == 2: 
                # heapq.heappush(
                #     self._vocab_cache,
                #     (vocab_counter, pretoken)
                # )
                continue

            #two or more letters -- add pairs to heap
            for i in range(1, len(pretoken)):
                pair = (pretoken[i-1], pretoken[i])
                heapq.heappush(self._vocab_cache, 
                               (vocab_counter, pair)
                               )
                
            # register merge
                self._merges.append(pair)
        

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

