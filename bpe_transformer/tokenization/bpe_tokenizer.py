import heapq

from collections import Counter
from pathlib import Path
from collections.abc import Iterable
from bpe_transformer.tokenization.tokenizer import Tokenizer
from bpe_transformer.tokenization.preprocessing import parallel_pretokenization


class BPETokenizer(Tokenizer):
    """
    Implementation of a greedy Byte-Pair Encoding (BPE).

    BPE is a data compression algorithm adapted for tokenization that iteratively merges
    the most frequent pair of adjacent tokens. Starting with individual bytes (0-255),
    it builds a vocabulary by repeatedly finding and merging the most common byte pairs
    until reaching the desired vocabulary size.

    Attributes:
        _vocab (dict[int, bytes]): Maps token IDs to their byte representations
        _merges (list[tuple[bytes, bytes]]): Ordered list of merged bytes operations performed
        _vocab_size (int): Target vocabulary size
        _special_tokens (set[str]): Special tokens to preserve during tokenization
        _vocab_pairs_heap (list[MaxHeapItem]): Max-heap of (count, pair) for frequent pairs
        _vocab_pairs_counter (Counter): True counts of all byte pairs

    Usage Example:
        >>> tokenizer = BPETokenizer(vocab_size=500, special_tokens=["<|endoftext|>"])
        >>> tokenizer.train(input_path=Path("corpus.txt"), num_processes=4)
        >>> # Tokenizer is now trained with 500 tokens (256 bytes + 244 merges)
    """

    class MaxHeapItem:
        """
        Wrapper class for heap items used in BPE training to track pair frequencies.

        Python's heapq module implements a min-heap, but we need a max-heap to always
        get the most frequent pair. This class reverses the comparison logic to achieve
        max-heap behavior.

        When counts are equal, vocab pairs are compared lexicographically by their byte
        representation (larger ones first) for deterministic tie-breaking.

        Args:
            count: Frequency of this pair in the corpus
            pair: Tuple of vocab IDs (int, int) representing the byte pair
            vocab: Reference to the vocabulary to get byte representations for comparison
        """

        def __init__(self, count, pair, vocab):
            self.count = count
            self.pair = pair
            # Store the byte representation for comparison
            self.pair_bytes = (vocab[pair[0]], vocab[pair[1]])

        def __lt__(self, other):
            if self.count != other.count:
                return self.count > other.count

            # Lexicographically greater bytes comes first (max heap behavior)
            return self.pair_bytes > other.pair_bytes

        def __repr__(self):
            return f"MaxHeapItem({self.count}, {self.pair})"

    def __init__(self, vocab_size: int, special_tokens: list[str]):
        if vocab_size < 256:
            raise ValueError("Invalid Vocab size: must be at least 256")

        self._encoding = "utf-8"
        self._vocab_size = vocab_size

        self._merges: list[tuple[bytes, bytes]] = list()
        self._special_tokens: set = set(special_tokens)
        self._vocab = self._build_initial_vocab()
        self._initial_vocab_size: int = len(self._vocab)  # 256 + len(special_tokens)

        # Internal variables
        # We can keep track of the most frequent pairs with a max heap.
        self._vocab_pairs_heap: list[BPETokenizer.MaxHeapItem] = []

    @property
    def vocab(self) -> dict[int, bytes]:
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

    @property
    def encoding(self) -> str:
        return self.encoding

    def _build_initial_vocab(self) -> dict[int, bytes]:
        """
        Build initial vocabulary with 256 base bytes and special tokens.

        Returns:
            Dictionary mapping token IDs to byte values.
        """
        # Base vocabulary: 256 values for bytes (0-255)
        vocab = {i: bytes([i]) for i in range(256)}

        # Add special tokens
        for idx, token in enumerate(self.special_tokens, start=256):
            vocab[idx] = token.encode(self.encoding)

        return vocab

    def add_new_vocab(self, id: int, new_value: bytes) -> None:
        """
        Add new token to the BPE vocab dict.

        Args:
            new_value: The new token to be registered, as bytes.
        """
        self._vocab[id] = new_value

    def train(self, input_path: Path, num_processes: int | None) -> None:
        """
        Train the BPE Tokenizer on an input file.
        Given the input file, we start from our base vocab (256 + special tokens),
        and keep "merging" the most frequent pairs of vocab tokens
        until we either reach our vocab size.

        Args:
            input_path: Path to the input file to be used for training.

        The training process can be summarised as:
            1. Initialize vocabulary with 256 base bytes (0-255) and special tokens.
            2. Count all adjacent byte pairs in the training corpus
            3. Merge the most frequent pair into a new token;
               If there are "ties" in frequency, the lexicographical shall be chosen
            4. Update pair frequencies and repeat
               until either the vocab_size is reached
               or the number of adjacent pairs to merge is empty.

        Implementation:
            - Uses a max-heap with lazy deletion to easily pop the most frequent
            - Pair counts are maintained in self._vocab_pairs_counter for O(1) lookups
            - Outdated heap entries (with outdated counts) are skipped during processing
            - Lexicographic tie-breaking
        """
        # Invoke pre-tokenization of input file
        pretoken_counter = self._get_pretokenization(input_path=input_path, num_processes=num_processes)

        # Initialize self._vocab_pairs_heap and self._vocab_pairs_counter:
        # Counts frequency of adjacency pairs in pretoken_counter
        self._initialize_pairs_cache(pretoken_counter)

        # Merge pairs of bytes
        self._merge_tokens(pretoken_counter)

        # Clean up training artifacts
        self._clear_training_cache()

    def _initialize_pairs_cache(self, pretoken_counter: Counter) -> None:
        """
        From the result of the pre-tokenization process,
        counts the frequency of all adjacent pairs of bytes in all pretokens,
        finally storing them as a max heap which orders the most frequent pairs.
        """
        # First, aggregate all pair counts
        pair_counter = Counter()
        for pretoken, count in pretoken_counter.items():
            if len(pretoken) == 1:
                continue

            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i + 1])
                pair_counter[pair] += count

        # set self._vocab_pairs_counter
        self._vocab_pairs_counter = pair_counter

        # Then build the heap from aggregated counts
        for pair, total_count in pair_counter.items():
            heapq.heappush(self._vocab_pairs_heap, self.MaxHeapItem(total_count, pair, self._vocab))

    def _merge_tokens(self, pretoken_counter: Counter[bytes, int]) -> None:
        """
        Given a dict of pretokens, merge pairs
        """
        # Pre-token dict is in form {list[bytes]: count}.
        # pre-token: {'low': 3, 'high': 2}
        # ex.: {(l, o, w): 3, (h, i, g, h): 2}
        # What I want to do is create a cache with counter of pairs.
        # i have to iterate until there are no potential pairs anymore

        new_id = len(self.vocab)
        # We will keep merging our vocab pairs
        # either until we reach the limit of the desired vocab size
        # or until there are no pairs left to merge.
        # print(f"N. of Pretokens to merge: {len(pretoken_counter)}")

        if not self._vocab_pairs_heap:
            raise AttributeError(
                "Warning: self._vocab_pairs_heap was not initialized. No pairs to start merging process."
            )

        while len(self.vocab) < self._vocab_size and len(self._vocab_pairs_heap) > 0:
            item = heapq.heappop(self._vocab_pairs_heap)
            pair = item.pair

            # Since we're not directly updating old pair counter in heap,
            # just adding it back, then we need to verify if it's a not-updated entry.
            if self._vocab_pairs_counter[pair] != item.count:
                continue

            # tracks the new pairs that were created and old pairs that decreased after the merge.
            updated_pairs = set()

            # Search for the pair in ALL pre-tokens from our training data.
            for s in list(pretoken_counter.keys()):
                if len(s) < 2:
                    continue
                merge_result = self._merge_pair(pretoken=s, pair=pair, vocab_id=new_id)

                if merge_result:
                    merged_token, merge_pos, pre_merge_pairs = merge_result

                    # Get the new adjacency pairs of merged token and frequency
                    new_pairs = self._get_adjacency_pairs(token=merged_token, positions=merge_pos)

                    # Update current_merge_pairs
                    for p in new_pairs:
                        self._vocab_pairs_counter[p[0]] += p[1] * pretoken_counter[s]
                        updated_pairs.add(p[0])

                    # Now, we need to take care of the count of the adjacency pairs
                    # of the pre-merged tokens.   Example: Merge (h,e) → [259, l, l, o]
                    # we have to decrease the occurence of (e, l), because it's not in the token anymore.
                    # The merge consumes TWO positions, so we need to account for neighbors of both
                    for p in pre_merge_pairs:
                        # We have to decrease the count of the old pairs that were removed
                        self._vocab_pairs_counter[p[0]] -= p[1] * pretoken_counter[s]
                        updated_pairs.add(p[0])

                    # Add the new merged token.
                    pretoken_counter[merged_token] += pretoken_counter[s]
                    # Finally, delete the pre-merge token.
                    del pretoken_counter[s]

            # Only add to vocab (once) if we found at least one merge
            if updated_pairs:
                self._merges.append((self._vocab[pair[0]], self._vocab[pair[1]]))

                # Register new vocab.
                merged_bytes = self._vocab[pair[0]] + self._vocab[pair[1]]
                self.add_new_vocab(id=new_id, new_value=merged_bytes)
                new_id += 1

                # Push to heap: all the new adjacency pairs
                # and the updated count for adjacency pairs removed after merge.
                [
                    heapq.heappush(
                        self._vocab_pairs_heap, self.MaxHeapItem(self._vocab_pairs_counter[p], p, self._vocab)
                    )
                    for p in updated_pairs
                    if self._vocab_pairs_counter[p] > 0
                ]

    def _merge_pair(self, pretoken: list[int], pair: tuple[int], vocab_id: int) -> tuple[tuple[int], tuple[int]] | None:
        """
        Given a list of vocab indexes of a word
        (which contains 0-255 only if no merges happened yet),
        replace a desired pair with the new index of the vocab.

        Eg.:
        pretoken = [2, 16, 45, 33, 1, 16, 45], pair = (16, 45), vocab_id = 334
        -> return [2, 334, 33, 1, 334], [(2, 334), (1, 334)], 2

        Args:
            pretoken: A pre-token consisting of a list of ids in the vocab
            pair: Tuple of ints of the pair we want to replace with a single vocab_id
            vocab_id: the vocab_id to replace the pair.
        Return:
            If no merges were made, return None.
            Else, return:   updated pretoken after substitutions,
                            an iterable containing the position(indexes) of the new merged vocab,
                            an iterable containing the pairs that will cease to "exist" after the merge.
        """
        if len(pretoken) < 2:
            raise ValueError("Merge pair call invalid: pre-token len. < 2")

        merged_token = []
        # position(s) of the new vocab id in the merged string
        new_vocab_pos = []
        # registers the pairs that will disappear after merging
        pre_merge_pos = Counter()
        # find pair position in pretoken
        found_pair = False
        i = 1
        while i < len(pretoken):
            if (pretoken[i - 1], pretoken[i]) != pair:
                # we don't add pretoken[i]
                # because it might be the start of a matching pair later.
                merged_token.append(pretoken[i - 1])
                i += 1
                continue

            # Register new merged token position
            found_pair = True
            merged_token.append(vocab_id)
            new_vocab_pos.append(len(merged_token) - 1)

            # Registers pairs that will cease to exist
            if (i - 2) >= 0:
                pre_merge_pos[(pretoken[i - 2], pretoken[i - 1])] += 1
            if (i + 1) < len(pretoken):
                pre_merge_pos[(pretoken[i], pretoken[i + 1])] += 1

            # skip to next available unmerged element
            i += 2

        # Add the last element if we didn't end with a merge
        if i == len(pretoken):
            merged_token.append(pretoken[-1])

        return (tuple(merged_token), iter(new_vocab_pos), iter(pre_merge_pos.items())) if found_pair else None

    def _get_adjacency_pairs(self, token: list[int], positions: list[int]) -> Iterable:
        """
        Given a pretoken and a list of positions of a merged new element,
        will return the adjacency pairs of integers and each frequency.

        (1, 2, 60, 5, 60, 3, 5, 60) -> {(2, 60): 1, (60, 5): 1, (5, 60): 2, (60, 3): 1}

        Args:
            token: post-merge token
            positions: list of indexes of the merged new token
        Return:
            A counter of ocurrences of adjacency pairs of the merged token.
        """
        adjacent_pairs = Counter()

        for pos in positions:
            # left neighbor
            if (pos - 1) >= 0:
                pair = (token[pos - 1], token[pos])
                adjacent_pairs[pair] += 1
            # right neighbor
            if (pos + 1) < len(token):
                pair = (token[pos], token[pos + 1])
                adjacent_pairs[pair] += 1

        return iter(adjacent_pairs.items())

    def _clear_training_cache(self) -> None:
        """
        Clear internal data structures used during training.

        After training completes, the heap and pair counter are no longer needed
        and can be cleared to free memory. Only the vocabulary and merge rules
        are retained for tokenization.
        """
        self._vocab_pairs_counter = Counter()
        self._vocab_pairs_heap = []

    def _get_pretokenization(self, input_path: Path, num_processes: int | None) -> Counter:
        """
        Call parallel pretokenization for a given input file.

        Args:
            input_path: Path to the data file to pretokenize.

        Return:
            Counter object with pre-token ocurrences in an input file.
        """
        pretoken_counter = parallel_pretokenization(
            file_path=input_path, num_processes=num_processes, special_tokens=list(self._special_tokens)
        )
        return pretoken_counter
