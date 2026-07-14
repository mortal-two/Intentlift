"""Source-side Triton kernels for the temporary SCRC feasibility probe.

This module is the source-visible side of the experiment.  The replay model is
never given this file; it only receives ``scrc_probe_inputs.json``.
"""

import triton
import triton.language as tl


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a + b
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    tl.store(out_ptr + row, total)


@triton.jit
def softmax_kernel(x_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=-float("inf"))
    row_max = tl.max(x, axis=0)
    shifted = x - row_max
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator
    tl.store(out_ptr + row * N + cols, out, mask=mask)


@triton.jit(do_not_specialize=["eps"])
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, M, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    squared = x * x
    square_sum = tl.sum(squared, axis=0)
    mean_square = square_sum / N
    shifted = mean_square + eps
    inv_rms = tl.rsqrt(shifted)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normalized = x * inv_rms
    out = normalized * w
    tl.store(out_ptr + row * N + cols, out, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_lanes = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_tile * BLOCK_K + k_lanes
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)
