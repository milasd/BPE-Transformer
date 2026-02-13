import torch

import numpy as np


def data_loader(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    input: nparray w/ tokens of all files.
    
    output: tuple of tensors, both shape [batch, seq_len]
    """
    # Need context_length + 1 tokens (input + target)
    max_start = len(x) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)

    inputs = torch.stack([
        torch.from_numpy(x[start : start + context_length].astype(np.int64))
        for start in starts
    ]).to(device)

    labels = torch.stack([
        torch.from_numpy(x[start + 1 : start + context_length + 1].astype(np.int64))
        for start in starts
    ]).to(device)

    return inputs, labels
