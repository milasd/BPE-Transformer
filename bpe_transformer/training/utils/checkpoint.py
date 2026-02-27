import os
import torch

from typing import IO, BinaryIO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    obj = {
        "t": iteration,
        "optimizer_state_dict": optimizer.state_dict(),
        "model_state_dict": model.state_dict(),
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src)

    iteration: int = checkpoint["t"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])

    return iteration
