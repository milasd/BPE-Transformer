"""MLX implementations of model components for Apple Silicon."""

from bpe_transformer.model.mlx.embedding import Embedding
from bpe_transformer.model.mlx.linear import Linear
from bpe_transformer.model.mlx.multihead_self_attention import MultiHeadSelfAttention
from bpe_transformer.model.mlx.rms_norm import RMSNorm
from bpe_transformer.model.mlx.rope import RoPE
from bpe_transformer.model.mlx.scaled_dot_product_attention import scaled_dot_product_attention
from bpe_transformer.model.mlx.softmax import softmax
from bpe_transformer.model.mlx.swiglu import SiLU, SwiGLU
from bpe_transformer.model.mlx.transformer_block import Transformer
from bpe_transformer.model.mlx.transformer_lm import TransformerLM

__all__ = [
    "Embedding",
    "Linear",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RoPE",
    "scaled_dot_product_attention",
    "softmax",
    "SiLU",
    "SwiGLU",
    "Transformer",
    "TransformerLM",
]
