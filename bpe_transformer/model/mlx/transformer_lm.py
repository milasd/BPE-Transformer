"""Transformer language model using MLX."""

from einops import rearrange, repeat
import mlx.core as mx
import mlx.nn as nn

from bpe_transformer.model.mlx.embedding import Embedding
from bpe_transformer.model.mlx.linear import Linear
from bpe_transformer.model.mlx.rms_norm import RMSNorm
from bpe_transformer.model.mlx.rope import RoPE
from bpe_transformer.model.mlx.softmax import softmax
from bpe_transformer.model.mlx.transformer_block import Transformer


class TransformerLM(nn.Module):
    """Transformer language model.

    Decoder-only transformer for autoregressive language modeling.

    Args:
        vocab_size: Size of the vocabulary.
        context_length: Maximum sequence length.
        num_layers: Number of transformer blocks.
        d_model: Model dimensionality.
        d_ff: Feedforward hidden dimension.
        num_heads: Number of attention heads.
        rope: Optional RoPE module for positional encoding.

    Shape:
        - Input: (batch, seq_len) with token IDs
        - Output: (batch, seq_len, vocab_size) with logits
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        rope: RoPE | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.rope = rope
        self.transformer_blocks = [
            Transformer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=rope) for _ in range(num_layers)
        ]
        self.norm = RMSNorm(d_model=d_model)
        self.linear = Linear(in_features=d_model, out_features=vocab_size)

    def __call__(self, x: mx.array, token_positions: mx.array | None = None) -> mx.array:
        """Forward pass through the language model.

        Args:
            x: Input token IDs of shape (batch, seq_len).
            token_positions: Position indices for RoPE.

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        _, seq_len = x.shape
        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length {seq_len} exceeds context_length {self.context_length}")

        # 1. Embedding
        token_embeddings = self.token_embedding(x)

        # 2. Transformer Blocks
        # Create token positions if rope embeddings will be added
        if self.rope is not None and token_positions is None:
            raise ValueError("Must pass token positions if rope embeddings will be used")

        x_transf = token_embeddings
        for transformer_block in self.transformer_blocks:
            x_transf = transformer_block(x=x_transf, token_positions=token_positions)

        # 3. Norm
        norm = self.norm(x=x_transf)

        # 4. Linear projection to vocab_size
        logits = self.linear(x=norm)

        return logits

    def generate(
        self,
        x: mx.array,
        eos_token_id: int,
        max_tokens: int,
        p: float,
        top_k: int | None = None,
        temperature: int | None = None,
    ) -> mx.array:
        """Predict next token given a sequence of tokens t until we reach end of token.
        input: sequence of tokens x0,...,xt
        predict xt+1
        add xt+1 to input
        loop: input sequence of tokens x0, ..., xt+1. predict xt+2. seq_len+1,
        end loop when xt+n == end of sequence token.
        """
        # unsqueeze if tensor is 1 dim only (seq_len,) -> (1, seq_len)
        if x.ndim == 1:
            x = rearrange(x, "n_tokens -> 1 n_tokens")

        # Keep original seq. len to retrieve generated tokens later
        original_seq_len = x.shape[-1]

        n_tokens: int = 0
        predicted_token = -1
        while n_tokens <= max_tokens:
            # If seq_len > context_length, get only past n=context_length tokens.
            if x.shape[1] > self.context_length:
                x = x[:, -self.context_length :]

            # 1. predict next token logits
            batch_size, seq_len = x.shape
            token_positions = repeat(mx.arange(seq_len), "seq_len -> batch seq_len", batch=batch_size)

            logits = self(x, token_positions=token_positions)[:, -1, :]  # get last logit only for prediction

            # if temperature is provided, scale
            if temperature is not None and temperature > 0:
                logits = logits / temperature

            # if top_k is required, apply
            if top_k:
                # get top k values
                top_k_values = mx.topk(logits, k=min(top_k, logits.shape[-1]))
                # get smallest value from top k
                min_k = top_k_values[:, -1:, None]
                mask = logits >= min_k
                # change non-top k values to -inf for softmax.
                logits = mx.where(mask, logits, float("-inf"))

            # apply softmax
            logits_prob = softmax(x=logits, i=-1)

            # Nucleus/Top-p logits.
            sorted_probs = mx.sort(logits_prob, axis=-1)[:, ::-1]  # descending
            sorted_indices = mx.argsort(logits_prob, axis=-1)[:, ::-1]
            # Sum of all probabilities
            cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
            # Truncate all elements which make sum > p
            mask = cumsum_probs <= p
            # set 1st as True for safety: if prob > p, mask would be all False.
            mask[:, 0] = True
            sorted_probs = mx.where(mask, sorted_probs, 0)
            filtered_probs = mx.zeros_like(logits_prob)
            # scatter-like operation using batch indices
            batch_indices = mx.arange(filtered_probs.shape[0])[:, None]
            filtered_probs[batch_indices, sorted_indices] = sorted_probs
            # now, normalize filtered probs
            filtered_probs = filtered_probs / mx.sum(filtered_probs, axis=-1, keepdims=True)

            # Sample from filtered distribution
            prediction = mx.random.categorical(mx.log(filtered_probs + 1e-10))  # (batch_size,)
            predicted_token = int(prediction[0])

            # Check if prediction must end
            if predicted_token == eos_token_id:
                break

            # Append next_token to x after sampling
            x = mx.concatenate([x, prediction[:, None]], axis=-1)

            n_tokens += 1

        # Get the ids of the generated tokens only
        new_token_ids = x[:, original_seq_len:]
        return new_token_ids
