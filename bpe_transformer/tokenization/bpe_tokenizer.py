import re

from pathlib import Path
from bpe_transformer.tokenization.preprocessing.pretokenization import pretokenize_text
from bpe_transformer.tokenization.settings import ENCODING_STD
from bpe_transformer.tokenization.tokenizer import Tokenizer


class BPETokenizer(Tokenizer):
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = special_tokens

    @property
    def vocab(self) -> dict[int, bytes]:
        return self._vocab

    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        return self._merges

    @property
    def special_tokens(self) -> list[str] | None:
        return self._special_tokens

    def encode_special_token(self, text: str) -> list[int]:
        if not self.special_tokens:
            raise EnvironmentError("Cannot encode text with special tokens without defined special tokens")

        # 1. pretokenize text.
        #   take care of special tokens if there are any.
        if self.special_tokens:
            # separate text chunks by special tokens (sort by length descending to match longest first)
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_tokens]
            split_pattern = f"({'|'.join(escaped_tokens)})"
            text_parts = re.split(split_pattern, text)

        encoded_text: list[int] = []
        # pretokenize non-special tokens only.
        for t in text_parts:
            if t in self.special_tokens:
                # no pre-tokenization; just get id from vocab.
                encoded_text.append(self._bytes_to_id[t.encode(ENCODING_STD)])
                continue

            # get pretokens for text part.
            pretokens = pretokenize_text(t)
            pretokens_vocab = [self._initialize_pretoken_vocab(pretoken) for pretoken in pretokens]

            # encode every pretoken
            for i in range(len(pretokens)):
                while len(pretokens_vocab[i]) >= 2:
                    merged_token = self._find_pair_in_merges(pretokens_vocab[i])

                    # no merges available; skip to next pretoken
                    if not merged_token:
                        break

                    # merge found; try merging more
                    pretokens_vocab[i] = merged_token

                # Add final encoded token to encoded text.
                encoded_text.extend(pretokens_vocab[i])

        return encoded_text

    def _find_pair_in_merges(self, pretoken: list[int]) -> list[int] | None:
        # search for the first possible merge inside merges
        for merge in self.merges:
            for i in range(1, len(pretoken)):
                if self.vocab[pretoken[i - 1]] == merge[0] and merge[1] == self.vocab[pretoken[i]]:
                    # found match. Return merged bytes string.
                    merged_bytes = self.vocab[pretoken[i - 1]] + self.vocab[pretoken[i]]
                    # [a, b, c, d, e] -> [a, b, cd, e]
                    return pretoken[: i - 1] + [self._bytes_to_id[merged_bytes]] + pretoken[i + 1 :]

        return None

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

    def encode(self, text: str) -> list[int]:
        # Reverse vocab key-values for faster vocab id fetching.
        self._bytes_to_id = {v: k for k, v in self.vocab.items()}

        if self.special_tokens:
            raise ValueError("Special tokens not supported yet")

        pretokens = pretokenize_text(text)
        encoded_text = []

        # convert pretokens to list of ints for vocab merging.
        pretokens_vocab = [self._initialize_pretoken_vocab(pretoken) for pretoken in pretokens]

        # Try to apply the first merge from self.merges found
        # in order, try to apply merges
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
            decoded_text += self.vocab.get(id, replacement_bytes)

        return decoded_text.decode(ENCODING_STD, errors="replace")

    def save_tokenizer(self, output_dir: Path = Path("output/tokenizer/bpe")) -> None:
        import pickle

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save vocab
        with open(output_dir / "vocab.pkl", "wb") as f:
            pickle.dump(self._vocab, f)

        # Save merges
        with open(output_dir / "merges.pkl", "wb") as f:
            pickle.dump(self._merges, f)

    @staticmethod
    def load_vocab(file_path: Path, special_tokens: list[str] | None) -> dict[int, bytes]:
        import pickle

        # Load vocab
        with open(file_path, "rb") as f:
            vocab = pickle.load(f)

        if not special_tokens:
            return vocab

        # Search if special tokens are already inside vocab.
        vocab_tokens = set(vocab.values())
        for t in special_tokens:
            if (t_bytes := t.encode(ENCODING_STD)) not in vocab_tokens:
                vocab[len(vocab)] = t_bytes

        return vocab

    @staticmethod
    def load_merges(file_path: Path) -> list[tuple[bytes, bytes]]:
        import pickle

        with open(file_path, "rb") as f:
            merges = pickle.load(f)
        return merges

    @classmethod
    def from_files(
        cls, vocab_filepath: Path, merges_filepath: Path, special_tokens: list[str] | None = None
    ) -> "BPETokenizer":
        return cls(
            vocab=cls.load_vocab(file_path=vocab_filepath, special_tokens=special_tokens),
            merges=cls.load_merges(merges_filepath),
            special_tokens=special_tokens,
        )
