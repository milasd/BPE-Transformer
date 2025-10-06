import heapq

from collections import Counter
from pathlib import Path
from collections.abc import Iterable
from bpe_transformer.settings import DEFAULT_OUTPUT_DIR
from bpe_transformer.tokenization.preprocessing import parallel_pretokenization


class BPETrainer:
    """
    Implementation of a greedy Byte-Pair Encoding Tokenizer trainer.
    """

    class MaxHeapItem:
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

    def __init__(self, vocab_size: int, special_tokens: list[str] | None = None):
        if vocab_size < 256:
            raise ValueError("Invalid Vocab size: must be at least 256")

        self.encoding = "utf-8"
        self._vocab_size = vocab_size

        # We can keep track of the most frequent pairs with a max heap.
        self._vocab_pairs_heap: list[BPETrainer.MaxHeapItem] = []
        self._merges: list[tuple[bytes, bytes]] = list()
        self._special_tokens: set = set(special_tokens)
        self._vocab = self._build_initial_vocab()
        self._initial_vocab_size: int = len(self._vocab)  # 256 + len(special_tokens)

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
        """
        # Invoke pre-tokenization of input file
        pretoken_counter = self._get_pretokenization(input_path=input_path, num_processes=num_processes)

        # Initialize self._vocab_pairs_heap:
        # Counts frequency of adjacency pairs in pretoken_counter
        self._initialize_pairs_cache(pretoken_counter)

        # Merge pairs of bytes
        self._merge_tokens(pretoken_counter)

    def _initialize_pairs_cache(self, pretoken_counter: Counter) -> None:
        """
        From the result of the pre-tokenization process,
        counts the frequency of all adjacent pairs of bytes in all pretokens,
        finally storing them as a max heap which orders the most frequent pairs.
        """
        # First, aggregate all pair counts
        pair_counter = Counter()
        # Track which pretokens contain which pairs for faster merging
        self._pair_to_pretokens = {}

        for pretoken, count in pretoken_counter.items():
            if len(pretoken) == 1:
                continue

            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i + 1])
                pair_counter[pair] += count

                # Track pretoken membership
                if pair not in self._pair_to_pretokens:
                    self._pair_to_pretokens[pair] = set()
                self._pair_to_pretokens[pair].add(pretoken)

        # set self._vocab_pairs_counter
        self._vocab_pairs_counter = pair_counter

        # Build heap to easily pop the max counter
        self._vocab_pairs_heap = [
            self.MaxHeapItem(total_count, pair, self._vocab) for pair, total_count in pair_counter.items()
        ]
        heapq.heapify(self._vocab_pairs_heap)

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

            # tracks the new pairs that were created and old pairs that decreased after the merge.
            updated_pairs = set()

            # Since we're not directly updating old pair counter in heap,
            # just adding it back, then we need to verify if it's a not-updated entry.
            if self._vocab_pairs_counter[pair] != item.count:
                continue

            merge_found = False

            # Only process pretokens that contain this pair (optimization)
            pretokens_with_pair = self._pair_to_pretokens.get(pair, set()).copy()

            for s in pretokens_with_pair:
                if s not in pretoken_counter:
                    # Pretoken was already merged in a previous iteration
                    continue

                # print(f"Current string: {s}")
                merge_result = self._merge_pair(pretoken=s, pair=pair, vocab_id=new_id)

                if merge_result:
                    merged_token, merge_pos, pre_merge_pairs = merge_result
                    merge_found = True
                    # print(f"Replaced {pair} in : {s} -> {merged_token}")

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

                    # Remove old pretoken from pair-to-pretokens mapping
                    for i in range(len(s) - 1):
                        old_pair = (s[i], s[i + 1])
                        if old_pair in self._pair_to_pretokens:
                            self._pair_to_pretokens[old_pair].discard(s)

                    # Add new pretoken to pair-to-pretokens mapping
                    for i in range(len(merged_token) - 1):
                        new_pair = (merged_token[i], merged_token[i + 1])
                        if new_pair not in self._pair_to_pretokens:
                            self._pair_to_pretokens[new_pair] = set()
                        self._pair_to_pretokens[new_pair].add(merged_token)

                    # Add the new merged token.
                    pretoken_counter[merged_token] += pretoken_counter[s]
                    # Finally, delete the pre-merge token.
                    del pretoken_counter[s]

            # Only add to vocab (once) if we found at least one merge
            if merge_found:
                self._merges.append((self._vocab[pair[0]], self._vocab[pair[1]]))
                # print(f"Added {pair} into new vocab token {new_id}")

                # Register new vocab.
                merged_bytes = self._vocab[pair[0]] + self._vocab[pair[1]]
                self.add_new_vocab(id=new_id, new_value=merged_bytes)
                new_id += 1

                # Push all the new adjacency pairs and decreased adjacency pairs for pre-merge
                [
                    heapq.heappush(
                        self._vocab_pairs_heap, self.MaxHeapItem(self._vocab_pairs_counter[p], p, self._vocab)
                    )
                    for p in updated_pairs
                    if self._vocab_pairs_counter[p] > 0
                ]

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

    # todo: rewrite __lr__ for max heap and remove * (-1) for counter.
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
                            and an iterable containing the position(indexes) of the new merged int.
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

        return (tuple(merged_token), tuple(new_vocab_pos), iter(pre_merge_pos.items())) if found_pair else None

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

    def save_trainer(self, output_dir: Path = Path(DEFAULT_OUTPUT_DIR / "tokenizer" / "bpe_trainer")) -> None:
        """
        Serializes the vocab and merges in individual files
        and export to output directory.

        Default is in project base directory/output/tokenizer/bpe_trainer.
        """
        import pickle

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save vocab
        with open(output_dir / "vocab.pkl", "wb") as f:
            pickle.dump(self._vocab, f)

        # Save merges
        with open(output_dir / "merges.pkl", "wb") as f:
            pickle.dump(self._merges, f)
