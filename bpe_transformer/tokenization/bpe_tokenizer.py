from pathlib import Path
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
