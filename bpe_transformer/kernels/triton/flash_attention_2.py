"""
Triton Implementation of Flash Attention 2 (in progress).
----
Currently contains a complete forward pass. Backward pass is in progress.

Attention:  O = softmax(Q @ K^T / sqrt(d)) @ V

Flash Attention performance improvements:
- Breaks Q and K/V into small blocks that fit in fast SRAM
- Uses "online softmax" to accumulate results chunk by chunk
- Avoids writing the NxN attention matrix to HBM memory

Flash Attention 2 vs v1:
  - Switched loop order (outer=Q, inner=K/V) -> less memory traffic
  - Keeps running stats (max, sum) in registers instead of slow memory
  - Better parallelism across GPU cores? (<- look into this in more details!)

Expected shapes: [batch, n_heads, seq_len, head_dim]

** The currently uv formatting standards are not looking good here... Look into a better one later.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_v2_fwd_kernel(
    # Tensor pointers (where stuff lives in GPU memory)
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,  # log-sum-exp, needed for backward pass
    # Strides tell us how to navigate through memory for each tensor dimension
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,  # Q strides
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,  # K strides
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,  # V strides
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,  # O strides
    stride_lse_b,
    stride_lse_h,
    stride_lse_m,  # LSE strides
    # Problem size
    seq_len,
    head_dim: tl.constexpr,  # constexpr = compile-time constant, lets Triton optimize
    scale,  # 1/sqrt(d)
    IS_CAUSAL: tl.constexpr,  # causal masking flag
    # Block sizes - how we tile the work
    BLOCK_M: tl.constexpr,  # Q rows per block
    BLOCK_N: tl.constexpr,  # K/V rows per iteration
    BLOCK_D: tl.constexpr,  # head_dim rounded to power of 2
):
    """
    Each kernel instance processes BLOCK_M rows of Q for one (batch, head) pair.

    Grid: [num_q_blocks, batch*heads]
    """

    # Figure out which chunk of work this thread block is responsible for
    q_block_idx = tl.program_id(0)  # which Q block (e.g., rows 256-383 if BLOCK_M=128)
    bh_idx = tl.program_id(1)  # which (batch, head) pair

    # Split the flattened batch*head index back into separate indices
    n_heads = stride_qb // stride_qh
    batch_idx = bh_idx // n_heads
    head_idx = bh_idx % n_heads

    # Set up base pointers for this specific (batch, head)
    # These point to [batch_idx, head_idx, 0, 0] in each tensor
    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    o_base = O_ptr + batch_idx * stride_ob + head_idx * stride_oh
    lse_base = LSE_ptr + batch_idx * stride_lse_b + head_idx * stride_lse_h

    # Create index arrays for the rows and columns we'll access
    q_row_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)  # which Q rows
    d_offsets = tl.arange(0, BLOCK_D)  # dimension indices (0 to head_dim-1)

    # Load our chunk of Q from slow memory into fast registers
    q_ptrs = q_base + q_row_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_mask = (q_row_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim)
    Q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Scale Q now instead of scaling every attention score later
    Q_block = (Q_block * scale).to(tl.float16)

    # Initialize running statistics for online softmax
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # running sum of exp
    O_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # output accumulator

    # Figure out how many K/V blocks we need to loop over
    if IS_CAUSAL:
        # For causal, we only attend to positions <= current position
        # So we don't need to process K/V blocks beyond our Q rows
        kv_block_end = tl.cdiv(q_block_idx * BLOCK_M + BLOCK_M, BLOCK_N)
        kv_block_end = tl.minimum(kv_block_end, tl.cdiv(seq_len, BLOCK_N))
    else:
        # Non-causal: attend to everything
        kv_block_end = tl.cdiv(seq_len, BLOCK_N)

    # Main loop: process K/V in chunks
    for kv_block_idx in range(0, kv_block_end):
        # Load this chunk of K
        kv_row_offsets = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + kv_row_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd
        k_mask = (kv_row_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim)
        K_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute attention scores for this Q/K pair
        # Remember Q was already scaled by 1/sqrt(d)
        S = tl.dot(Q_block, tl.trans(K_block))  # [BLOCK_M, BLOCK_N]

        # Apply causal mask if needed (can't attend to future positions)
        if IS_CAUSAL:
            causal_mask = q_row_offsets[:, None] >= kv_row_offsets[None, :]
            S = tl.where(causal_mask, S, float("-inf"))

        # Mask out padding tokens at the end of the sequence
        kv_valid_mask = kv_row_offsets[None, :] < seq_len
        S = tl.where(kv_valid_mask, S, float("-inf"))

        # Online softmax update
        # Find max in this chunk
        m_ij = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, m_ij)  # update running max

        # Compute attention weights (unnormalized)
        P = tl.exp(S - m_new[:, None])

        # Correction factor: when max changes, we need to rescale old values
        # If new max is bigger, old exp values were too large
        alpha = tl.exp(m_i - m_new)

        # Update running sum of exp (the denominator)
        l_i = l_i * alpha + tl.sum(P, axis=1)

        # Rescale previous output accumulator with correction factor
        O_acc = O_acc * alpha[:, None]

        # Load V and add weighted contribution to output
        v_ptrs = v_base + kv_row_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd
        v_mask = (kv_row_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim)
        V_block = tl.load(v_ptrs, mask=v_mask, other=0.0)

        O_acc += tl.dot(P.to(V_block.dtype), V_block)

        # Update running max for next iteration
        m_i = m_new

    # Normalize output by dividing by the sum
    O_acc = O_acc / l_i[:, None]

    # Compute log-sum-exp for backward pass
    # LSE = log(sum(exp(S))) = m + log(sum(exp(S - m)))
    # Backward pass needs this to recompute softmax without storing full attention matrix
    lse = m_i + tl.log(l_i)

    # Write output back to global memory
    o_ptrs = o_base + q_row_offsets[:, None] * stride_om + d_offsets[None, :] * stride_od
    o_mask = (q_row_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim)
    tl.store(o_ptrs, O_acc.to(tl.float16), mask=o_mask)

    lse_ptrs = lse_base + q_row_offsets * stride_lse_m
    lse_mask = q_row_offsets < seq_len
    tl.store(lse_ptrs, lse, mask=lse_mask)


def flash_attention_v2_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flash Attention 2 forward pass.

    Args:
        Q, K, V: [batch, n_heads, seq_len, head_dim], float16, contiguous
        causal: whether to mask future positions

    Returns:
        O: attention output [B, H, N, d]
        LSE: log-sum-exp [B, H, N] (needed for backward)
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "need GPU tensors"
    assert Q.dtype == torch.float16, "only float16 supported"

    batch, n_heads, seq_len, head_dim = Q.shape
    scale = head_dim**-0.5

    O = torch.empty_like(Q)
    LSE = torch.empty(batch, n_heads, seq_len, device=Q.device, dtype=torch.float32)

    # Tile sizes - these work well for most cases
    # Can tune for specific hardware/problem sizes
    BLOCK_M = 128  # Q rows per block
    BLOCK_N = 64  # K/V rows per iteration
    BLOCK_D = triton.next_power_of_2(head_dim)  # must be power of 2 for matmul

    # Launch one thread block per (Q_block, batch*head) pair
    num_q_blocks = triton.cdiv(seq_len, BLOCK_M)
    grid = (num_q_blocks, batch * n_heads)

    flash_attention_v2_fwd_kernel[grid](
        Q,
        K,
        V,
        O,
        LSE,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        O.stride(3),
        LSE.stride(0),
        LSE.stride(1),
        LSE.stride(2),
        seq_len=seq_len,
        head_dim=head_dim,
        scale=scale,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return O, LSE


def torch_attention(Q, K, V, causal=True):
    """Test simple implementation of attention w/ PyTorch."""
    scale = Q.shape[-1] ** -0.5
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if causal:
        seq_len = Q.shape[2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        S.masked_fill_(mask, float("-inf"))

    A = torch.softmax(S, dim=-1)
    return torch.matmul(A, V)


def test_flash_attention():
    """Simple test to check if this script matches reference implementation."""
    torch.manual_seed(42)

    B, H, N, d = 2, 4, 512, 64
    Q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)

    # Test causal
    O_flash, _ = flash_attention_v2_forward(Q, K, V, causal=True)
    O_ref = torch_attention(Q, K, V, causal=True)

    max_diff = (O_flash - O_ref).abs().max().item()
    mean_diff = (O_flash - O_ref).abs().mean().item()

    print(f"Causal attention:")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Status: {'PASS' if max_diff < 0.01 else 'FAIL'}")

    # Test non-causal
    O_flash_nc, _ = flash_attention_v2_forward(Q, K, V, causal=False)
    O_ref_nc = torch_attention(Q, K, V, causal=False)
    max_diff_nc = (O_flash_nc - O_ref_nc).abs().max().item()

    print(f"\nNon-causal attention:")
    print(f"  Max diff: {max_diff_nc:.6f}")
    print(f"  Status: {'PASS' if max_diff_nc < 0.01 else 'FAIL'}")


if __name__ == "__main__":
    test_flash_attention()
