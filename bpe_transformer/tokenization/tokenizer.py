from abc import ABC, abstractmethod
from pathlib import Path


class Tokenizer(ABC):
    @property
    @abstractmethod
    def vocab(self) -> dict[int, bytes]:
        pass

    @property
    @abstractmethod
    def merges(self) -> list[tuple[bytes, bytes]]:
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        pass

    @classmethod
    def from_files(cls, vocab_filepath: Path, merges_filepath: Path, special_tokens=None) -> None:
        pass

    @abstractmethod
    def save_tokenizer(self, output_dir: Path) -> None:
        pass
