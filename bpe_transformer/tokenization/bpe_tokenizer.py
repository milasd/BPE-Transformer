from bpe_transformer.tokenization.preprocessing.pretokenization import pretokenize_text, split_on_special_tokens
from bpe_transformer.settings import ENCODING_STD
from bpe_transformer.tokenization.tokenizer import Tokenizer
from collections.abc import Iterable
from pathlib import Path


class BPETokenizer(Tokenizer):
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = set(special_tokens) if special_tokens else {}
        self._bytes_to_id_cache = None

    @property
    def vocab(self) -> dict[int, bytes]:
        return self._vocab

    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        return self._merges

    @property
    def special_tokens(self) -> list[str] | None:
        return list(self._special_tokens)

    @property
    def _bytes_to_id(self) -> dict[bytes, int]:
        """Cached reverse vocab mapping (bytes to int) for faster lookups."""
        if self._bytes_to_id_cache is None:
            self._bytes_to_id_cache = {v: k for k, v in self._vocab.items()}
        return self._bytes_to_id_cache

    @classmethod
    def from_files(
        cls, vocab_filepath: Path, merges_filepath: Path, special_tokens: list[str] | None = None
    ) -> "BPETokenizer":
        """
        Creates an instance of BPETokenizer based on files for vocab and merges,
        and the list of special tokens to consider.
        Will add special tokens to vocab if they're not in the provided vocab file.

        Args:
            Path to file containing vocab
            Path to file containing merges
            List of special tokens str
        Returns:
            BPETokenizer with loaded vocab, merges and special tokens
        """
        return cls(
            vocab=cls.load_vocab(file_path=vocab_filepath, special_tokens=special_tokens),
            merges=cls.load_merges(merges_filepath),
            special_tokens=special_tokens,
        )

    def decode(self, ids: list[int]) -> str:
        """
        Given a list of ints, which are the ids inside the vocab,
        return a decoded string. Invalid ids will be replaced with the default Replacement error.

        Args:
            ids: list containing ids from vocab
        Return:
            decoded string
        """
        decoded_text = b""
        replacement_bytes = b"\xef\xbf\xbd"
        for id in ids:
            decoded_text += self._vocab.get(id, replacement_bytes)

        return decoded_text.decode(ENCODING_STD, errors="replace")

    def encode(self, text: str) -> list[int]:
        """
        Encodes a text string based on the class vocab and merges list.

        First, the text is pretokenized, considering any special characters.
        The pretokens are mapped to the corresponding vocab ids.

        Then, the function will start to try merging the pretoken ids,
        associating the merged bytes to its vocab id.
        The merging process is greedy, that is,
        the encoding will find the first merge inside the merges list
        that is applicable to the pretoken.
        This is repeated to every pretoken, except special tokens,
        which are directly mapped to the vocab id.

        The final token (after every possible merge) ids will be appended
        to the encoded text, which is a list of every token id.

        Args:
            text string
        Returns:
            Encoded test, which is a list of int ids from the vocab.
        """
        encoded_text: list[int] = []

        # 1. pretokenize text.
        # take care of special tokens if there are any.
        if self.special_tokens:
            text_parts = split_on_special_tokens(text=text, training=False, special_tokens=self.special_tokens)
        else:
            text_parts = [text]

        # pretokenize non-special tokens only.
        for t in text_parts:
            if t in self.special_tokens:
                # no pre-tokenization; just get id from vocab.
                encoded_text.append(self._bytes_to_id[t.encode(ENCODING_STD)])
                continue

            # get pretokens for text part.
            pretokens = pretokenize_text(t)
            pretokens_vocab = [self._initialize_pretoken_vocab(pretoken) for pretoken in pretokens]

            # encode all pretokens
            encoded_text_part = self._encode_pretokens(pretokens_vocab=pretokens_vocab)

            # Add encoded text part to encoded text.
            encoded_text.extend(encoded_text_part)

        return encoded_text

    def encode_parallel(self, text: str, chunk_size: int = 10000, n_workers: int = 4) -> list[int]:
        pass

    def _initialize_pretoken_vocab(self, pretoken: bytes) -> list[int]:
        """
        Given a pretoken in bytes, we'll construct their initial encoding
        with our vocab.
        This is necessary as the custom vocab order might differ from the automatic
        encoding/decoding order from utf-8.

        Args:
            pretoken: the pretoken in bytes
        Return:
            List of vocab idx of the pretoken (Array of ints)
        """
        pretoken_vocab = []
        for b in pretoken:
            value = bytes([b])
            pretoken_vocab.append(self._bytes_to_id[value])
        return pretoken_vocab

    def _encode_pretokens(self, pretokens_vocab: list[list[int]]) -> list[int]:
        """
        Given a list of pretokens, already mapped to their vocab ids,
        will try to apply greedy merging to each pretoken.

        The merge to be applied shall always be the first one existing
        inside self.merges list that is also in the pretoken.

        The function will try to merge the pretoken as much as possible.
        If no merges are found/can be done anymore,
        will skip to next pretoken.

        Args:
            pretokens_vocab: list of pretokens list with initial mapping to vocab

        Return:
            Encoded text, a list of ints containing final encoding of all tokens.
        """
        encoded_text = []
        # Try to apply the first merge from self.merges available to pretoken
        for i in range(len(pretokens_vocab)):
            # pretokenize every pretoken until it's not possible anymore.
            while len(pretokens_vocab[i]) >= 2:
                # search for each pair of pretoken bytes inside merges.
                merged_token = self._find_pair_in_merges(pretokens_vocab[i])

                # didn't find any possible merge; go to next pretoken.
                if not merged_token:
                    break

                # successfully found merge pair; see if we can merge new token again.
                pretokens_vocab[i] = merged_token

            encoded_text.extend(pretokens_vocab[i])

        return encoded_text

    def _find_pair_in_merges(self, pretoken: list[int]) -> list[int] | None:
        """
        Search for the first applicable merge inside merges list,
        apply it to pretoken if any are found and return the post-merge token.

        Args:
            pretoken: list of ints (vocab ids)
        Return:
            If no possible merges are found, will return None;
            If a merge is possible, will return the first post-merge token.
        """
        # search for the first possible merge inside merges
        for merge in self.merges:
            for i in range(1, len(pretoken)):
                if self._vocab[pretoken[i - 1]] == merge[0] and merge[1] == self._vocab[pretoken[i]]:
                    # found match. Return merged bytes string.
                    merged_bytes = self._vocab[pretoken[i - 1]] + self._vocab[pretoken[i]]
                    # [a, b, c, d, e] -> [a, b, cd, e]
                    return pretoken[: i - 1] + [self._bytes_to_id[merged_bytes]] + pretoken[i + 1 :]

        return None

    @staticmethod
    def load_vocab(file_path: Path, special_tokens: list[str] | None) -> dict[int, bytes]:
        """
        Load vocab from a file path, adding special tokens to the vocab
        if they're not in the loaded vocab file.

        Args:
            file_path: Path to file containing vocab only
            special_tokens: list of special tokens to consider
        Return:
            a dict containing the vocab, matching id to bytes
        """
        import pickle

        # Load vocab
        with open(file_path, "rb") as f:
            vocab: dict[int, bytes] = pickle.load(f)

        if not special_tokens:
            return vocab

        # Check if special tokens are already inside vocab.
        vocab_tokens = set(vocab.values())
        for t in special_tokens:
            if (t_bytes := t.encode(ENCODING_STD)) not in vocab_tokens:
                vocab[len(vocab)] = t_bytes

        return vocab

    @staticmethod
    def load_merges(file_path: Path) -> list[tuple[bytes, bytes]]:
        """
        Load merges list from a file path.

        Args:
            file_path: Path to file containing vocab only
        Return:
            the merge list containing tuples of merged bytes
        """
        import pickle

        with open(file_path, "rb") as f:
            merges: list[tuple[bytes, bytes]] = pickle.load(f)
        return merges

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        pass
