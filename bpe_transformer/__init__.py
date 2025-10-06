import importlib.metadata
from .main import train_bpe

__version__ = importlib.metadata.version("bpe_transformer")

__all__ = ["train_bpe"]
