"""Blind replay generated from ``scrc_probe_inputs.json`` only.

Requested/returned model: gpt-5.5
Response id: resp_09484adc73d88790016a55f4e4fd188195868c5a7ef804a822
Generation attempt: 1 (no compiler repair yet)
"""

import triton
import triton.language as tl


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    offset = pid * BLOCK + lane
    mask = offset < n_elements

    a = tl.load(a_ptr + offset, mask=mask, other=0.0)
    b = tl.load(b_ptr + offset, mask=mask, other=0.0)
    out = a + b

    tl.store(out_ptr + offset, out, mask=mask)


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_N)
    mask = col < N

    x = tl.load(x_ptr + row * N + col, mask=mask, other=0.0)
    out = tl.sum(x, axis=0)

    tl.store(out_ptr + row, out)


@triton.jit
def softmax_kernel(x_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_N)
    mask = col < N

    x = tl.load(x_ptr + row * N + col, mask=mask, other=-float("inf"))
    row_max = tl.max(x, axis=0)
    shifted = x - row_max
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(out_ptr + row * N + col, y, mask=mask)


@triton.jit(do_not_specialize=["eps"])
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, M, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_N)
    mask = col < N

    x = tl.load(x_ptr + row * N + col, mask=mask, other=0.0)
    squared = x * x
    mean_square = tl.sum(squared, axis=0) / N
    shifted = mean_square + eps
    inv_rms = tl.rsqrt(shifted)
    w = tl.load(w_ptr + col, mask=mask, other=0.0)
    normalized = x * inv_rms
    y = normalized * w

    tl.store(out_ptr + row * N + col, y, mask=mask)


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

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_lanes = tl.arange(0, BLOCK_K)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_tile * BLOCK_K + k_lanes

        a = tl.load(
            a_ptr + m[:, None] * K + k[None, :],
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k[:, None] * N + n[None, :],
            mask=(k[:, None] < K) & (n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)

    tl.store(
        out_ptr + m[:, None] * N + n[None, :],
        acc,
        mask=(m[:, None] < M) & (n[None, :] < N),
    )
