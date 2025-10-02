import heapq

from collections import Counter
from pathlib import Path
from typing import Iterable
from bpe_transformer.tokenization.tokenizer import Tokenizer
from bpe_transformer.tokenization.preprocessing import parallel_pretokenization


class MaxHeapItem:
    def __init__(self, count, pair):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        # Reverse the count comparison for max heap.
        if self.count != other.count:
            return self.count > other.count
        
        # If counts are the same, larger lexicographical pair comes first.
        return self.pair > other.pair
    
    def __repr__(self):
        return f"MaxHeapItem({self.count}, {self.pair})"


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
        self._vocab_cache: list[int, bytes]= []
        self._merges: set[tuple[bytes, bytes]] = set()
        self._special_tokens: set = set(special_tokens)
        self._vocab = self._build_initial_vocab()
        self._initial_vocab_size: int = len(self._vocab) # 256 + len(special_tokens)


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
        pretoken_counter = self._get_pretokenization(input_path, num_processes)

        # Merge pairs of bytes
        self._merge_tokens(pretoken_counter)


    def _merge_tokens(self, pretoken_counter: Counter[bytes, int]) -> None:
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

            # Edge case: only a single byte/pair in pretoken -- no merge.
            if len(pretoken) == 1 or len(pretoken) == 2: 
                continue

            #two or more letters -- add pairs to heap
            for i in range(1, len(pretoken)):
                pair = (pretoken[i-1], pretoken[i])
                heapq.heappush(self._vocab_cache, 
                               MaxHeapItem(count, pair)
                               )
                
        new_id = len(self.vocab)
        # We will keep merging our vocab
        # either until we reach the limit of the desired vocab size
        # or until there are no pairs left to merge.
        my_vocab_cache = self._vocab_cache
        print("Test strings:")
        print(pretoken_counter[:10])
        print(f"n. of test strings: {len(pretoken_counter)}")

        while len(self.vocab) < self._vocab_size and len(my_vocab_cache) > 0:
            # For each pair in our vocab pair counter,
            item = heapq.heappop(my_vocab_cache)
            pair = item.pair
            print(f"Current pair to look after: {pair} (count: {item.count})")
            # pair = heapq.heappop(self._vocab_cache)
            merge_found = False

            # counts the frequency of new merged pairs for this merge iteration.
            current_merge_pairs = Counter()
            
            # Replace the pair in ALL strings in our sample
            for s in list(pretoken_counter.keys()):
                if len(s) < 2: continue
                # print(f"Current string: {s}")
                merge_result = self._merge_pair(pretoken=s, pair=pair, vocab_id=new_id)

                if merge_result:
                    merged_token, merge_pos = merge_result
                    merge_found = True
                    print(f"Replaced {pair} in : {s} -> {merged_token}")

                    # Get the new adjacency pairs of merged token and frequency
                    new_pairs = self._get_adjacency_pairs(token=merged_token, positions=merge_pos)

                    # Update current_merge_pairs
                    for p in new_pairs:
                        current_merge_pairs[p[0]] += p[1]

                    # Push back the merged token and delete the pre-merge token.
                    pretoken_counter[merged_token] += pretoken_counter[s]
                    del pretoken_counter[s]


            ## TODO::: look how to change the counter to a heap queue to pop, or an iterable if above doesnt work.
            
            # Only add to vocab (once) if we found at least one merge
            if merge_found:
                self._merges.add((pair, new_id))
                print(f"Added {pair} into new vocab token {new_id}")

                # Register new vocab.
                self.add_new_vocab(new_id, new_value=pair)
                new_id += 1

                # And the all new adjacency pairs observed in all strings after the merge to vocab_cache.
                [heapq.heappush(my_vocab_cache, 
                                MaxHeapItem(counter, new_pair)) for new_pair, counter in current_merge_pairs.items()]


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
                pair = (token[pos-1], token[pos])
                adjacent_pairs[pair] += 1
            # right neighbor
            if (pos + 1) < len(token):
                pair = (token[pos], token[pos+1])
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
        if len(pretoken) < 2: raise ValueError("Merge pair call invalid: pre-token len. < 2")
        
        merged_token = []
        # position(s) of the new vocab id in the merged string
        new_vocab_pos = []
        # find pair position in pretoken
        found_pair = False
        i = 1
        while i < len(pretoken):
            if (pretoken[i-1], pretoken[i]) != pair:
                # we don't add pretoken[i] 
                # because it might be the start of a matching pair later.
                merged_token.append(pretoken[i-1])
                i += 1
                continue
            
            found_pair = True
            merged_token.append(vocab_id)
            new_vocab_pos.append(len(merged_token) - 1)
            #skip to next available unmerged element
            i += 2

        # Add the last element if we didn't end with a merge
        if i == len(pretoken):
            merged_token.append(pretoken[-1])

        return (tuple(merged_token), tuple(new_vocab_pos)) if found_pair else None


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

