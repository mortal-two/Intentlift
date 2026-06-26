"""A self-contained Triton FlashAttention-forward golden (the vs-Triton baseline).

Non-causal, fp32, single-head [Sq, D] x [Sk, D] x [Sk, D] -> [Sq, D]. This is the
fused online-softmax algorithm (running max m_i, running sum l_i, accumulator
acc, rescaled per K-tile) — the same recurrence the intent `loop` form expresses.
We use it as the external reference for both correctness (vs our intent kernel,
which computes the identical attention output) and performance (M11 vs-Triton).

Adapted from the Triton fused-attention tutorial, reduced to the bring-up shape.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover - triton optional
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _flash_fwd(
        Q, K, V, scale, Out,
        sq, sk, d,
        BLOCK_N: tl.constexpr, D: tl.constexpr,
    ):
        # One program per query row (BLOCK_M = 1) — the bring-up granularity that
        # mirrors our "one thread per query" fused intent lowering. Correct and
        # fused; not the warp-tiled tutorial perf point (that is what retune is for).
        qm = tl.program_id(0)
        q = tl.load(Q + qm * d + tl.arange(0, D))  # [D]

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([D], dtype=tl.float32)

        for start in range(0, sk, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < sk
            k = tl.load(K + offs_n[:, None] * d + tl.arange(0, D)[None, :],
                        mask=mask_n[:, None], other=0.0)        # [BLOCK_N, D]
            v = tl.load(V + offs_n[:, None] * d + tl.arange(0, D)[None, :],
                        mask=mask_n[:, None], other=0.0)        # [BLOCK_N, D]
            s = tl.sum(q[None, :] * k, axis=1) * scale          # [BLOCK_N]
            s = tl.where(mask_n, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)                               # [BLOCK_N]
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)  # [D]
            m_i = m_new

        acc = acc / l_i
        tl.store(Out + qm * d + tl.arange(0, D), acc)


def triton_flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: float = 1.0, block_n: int = 16,
) -> torch.Tensor:
    """Single-head flash forward: q[Sq,D], k/v[Sk,D] -> [Sq,D]."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    sq, d = q.shape
    sk = k.shape[0]
    out = torch.empty_like(q)
    _flash_fwd[(sq,)](
        q, k, v, scale, out, sq, sk, d,
        BLOCK_N=block_n, D=d,
    )
    return out
