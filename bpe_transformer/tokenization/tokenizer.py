from abc import ABC, abstractmethod
from collections.abc import Iterable
from collections.abc import Iterator


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
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        pass

    @classmethod
    def from_files(cls) -> None:
        pass
