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
    def train(self, input_path: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        pass

    # @abstractmethod
    # def encode(self):
    #     pass

    # @abstractmethod
    # def decode(self):
    #     pass

    # @abstractmethod
    # def load_tokenizer(self):
    #     pass

    # @abstractmethod
    # def save_tokenizer(self):
    #     pass
