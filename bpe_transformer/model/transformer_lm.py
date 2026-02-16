from einops import rearrange, repeat
from torch import nn
import torch

from bpe_transformer.model.modules import Embedding, Linear, RMSNorm, RoPE, Transformer, softmax


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
        device: Device for parameters. Defaults to None.
        dtype: Data type for parameters. Defaults to None.

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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size=vocab_size, d_model=d_model, device=device, dtype=dtype)
        self.rope = rope
        self.transformer_blocks = nn.ModuleList(
            [
                Transformer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, device=device, dtype=dtype, rope=rope)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.linear = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
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
        token_embeddings = self.token_embedding(token_ids=x)

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


    @torch.no_grad()
    def generate(self, x: torch.Tensor, eos_token_id: int, max_tokens: int, p: float,  top_k: int | None = None, temperature: int | None = None) -> torch.Tensor:
        """Predict next token given a sequence of tokens t until we reach end of token.
        input: sequence of tokens x0,...,xt
        predict xt+1
        add xt+1 to input
        loop: input sequence of tokens x0, ..., xt+1. predict xt+2. seq_len+1, 
        end loop when xt+n == end of sequence token.
        """
        # unsqueeze if tensor is 1 dim only (seq_len,) -> (1, seq_len)
        if x.dim() == 1:
            x = rearrange(x, 'n_tokens -> 1 n_tokens')
        
        # Keep original seq. len to retrieve generated tokens later
        original_seq_len = x.shape[-1]
        
        n_tokens: int = 0
        predicted_token = -1
        while n_tokens <= max_tokens:
            # If seq_len > context_length, get only past n=context_length tokens.
            if x.size(1) > self.context_length:
                x = x[:, -self.context_length :]
            
            # 1. predict next token logits
            batch_size, seq_len = x.shape
            token_positions = repeat(torch.arange(seq_len, device=x.device), 'seq_len -> batch seq_len', batch=batch_size)
            
            logits = self.forward(x, token_positions=token_positions)[:, -1, :] # get last logit only for prediciton 
            
            # if temperature is provided, scale 
            if temperature is not None and temperature > 0: 
                logits = logits / temperature
                
            # if top_k is required, apply
            if top_k:
                # get top k values
                top_k_values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                # get smallest value from top k
                min_k = top_k_values[:, -1].unsqueeze(-1)
                mask = logits >= min_k
                # change non-top k values to -inf for softmax.
                logits[~mask] = float('-inf')
                
            # apply softmax
            logits_prob = softmax(x=logits, i=-1)
            
            # Nucleus/Top-p logits.
            sorted_probs, sorted_indices = torch.sort(logits_prob, dim=-1, descending=True)
            # Sum of all probabilities
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1, dtype=sorted_probs.dtype)
            # Truncate all elements which make sum > p
            mask = (cumsum_probs <= p)
            # set 1st as True for safety: if prob > p, mask would be all False. 
            mask[:, 0] = True  
            filtered_probs = torch.zeros_like(logits_prob)
            filtered_probs.scatter_(dim=-1, index=sorted_indices[mask], src=sorted_probs[mask])
            # now, normalize filtered probs
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            
            # Sample from filtered distribution
            prediction = torch.multinomial(filtered_probs, num_samples=1) # (batch_size, 1)
            predicted_token = prediction.item()
            
            # Check if prediction must end
            if predicted_token == eos_token_id:
                break
            
            # Append next_token to x after sampling
            x = torch.cat([x, prediction], dim=-1)
            
            n_tokens += 1
            
        # Get the ids of the generated tokens only
        new_token_ids = x[:, original_seq_len:]
        return new_token_ids
        