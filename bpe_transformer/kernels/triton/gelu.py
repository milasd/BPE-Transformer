"""
Triton kernel containing an implementation of GeLU (Gaussian Error Linear Units).

GeLU(x) = x * Φ(x),
where Φ(x) is the cumulative distribution function for Gaussian distribution.

We use the tanh approximation:
GeLU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
"""

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024


def gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)  # Allocating the output tensor

    # Grid
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    gelu_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return y


@triton.jit
def gelu_kernel(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Offsets for thread block operations
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Handle boundaries
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)

    # GeLU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    sqrt_2_over_pi = 0.79788456  # sqrt(2/pi)
    c = 0.044715

    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + c * x_cubed)
    exp = tl.exp(2 * inner)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1.0 + tanh)

    # Store output
    tl.store(y_ptr + offsets, y, mask=mask)
