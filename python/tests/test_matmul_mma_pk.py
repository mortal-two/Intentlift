"""PK the tiled-K tensor-core matmul lowering vs torch.matmul (f16 -> f32).

The M7a deliverable: a single ``intent.contraction{"mk,nk->mn"}`` (A[M,K] f16,
W[N,K] f16 -> Y[M,N] f32) lowered by ``-convert-intent-contraction-to-gpu`` to
ONE tiled-K tensor-core ``gpu.launch`` (BM=BN=64, BK=16, a 4x4 grid of
m16n16k16 WMMA tiles over 4 warps; per-warp fp32 accumulator fragments carried
across the K loop as scf.for iter_args; A/W staged through dynamic shared). The
contraction is realized by NVIDIA WMMA (gpu.subgroup_mma_* -> wmma.mma.sync on
sm_120a), the first CUDA-core-free path — NOT a linalg.generic scalar MAC.

M7a bar = CORRECTNESS + RUNS on sm_120a (rel under the f16 matmul tolerance vs
torch.matmul(A, W.t())). Perf is reported as an INFORMATIONAL datapoint only
(perf-tuning + bias/activation epilogue is M7b): a plain scf.for WMMA with no
cp.async multi-stage pipeline / no L2 swizzle is expected to trail a tuned
tl.dot / cuBLAS; that gap is the named M7b follow-up, not an M7a gate.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_matmul_mma_pk.py -q -s
"""

from __future__ import annotations

import pathlib

import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                       reason="intent-opt not built"),
    pytest.mark.skipif(not pathlib.Path(MLIR_OPT).exists(),
                       reason="mlir-opt not present"),
    pytest.mark.skipif(not pathlib.Path(CUDA_BIN).exists(),
                       reason="CUDA toolkit not present"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]


def _mlir(M, N, K):
    A, W, Y = f"tensor<{M}x{K}xf16>", f"tensor<{N}x{K}xf16>", f"tensor<{M}x{N}xf32>"
    return f"""
func.func @matmul(%A: {A}, %W: {W}) -> {Y} {{
  %Y = intent.contraction %A, %W {{subscripts = "mk,nk->mn"}} : ({A}, {W}) -> {Y}
  return %Y : {Y}
}}
"""


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _bench(fn):
    import triton
    return triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median")


@pytest.mark.parametrize("M,N,K", [
    (64, 64, 32),       # one block, one K-trip (the smoke case; BK=32)
    (128, 128, 64),     # multi-block, multi-trip
    (512, 512, 512),
    (1024, 1024, 1024),
])
def test_matmul_mma_pk(M, N, K):
    from intentir.benchmarks.row_reduction import MatmulKernel

    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    W = torch.randn(N, K, device="cuda", dtype=torch.float16).contiguous()
    Y = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    k = MatmulKernel(_mlir(M, N, K))
    go = k.thunk([A, W, Y])
    go()
    torch.cuda.synchronize()

    # f16 inputs, f32 accumulate == torch.matmul on the upcast operands.
    ref = torch.matmul(A.float(), W.float().t())
    rel = _rel(Y, ref)
    # f16 matmul tolerance: rel grows ~sqrt(K) with the accumulation length but
    # stays well under 2e-2 (observed ~1e-7..1e-6 vs the fp32-accumulated ref).
    assert rel < 2e-2, f"({M},{N},{K}) rel={rel:.3e}"

    # Informational perf datapoint (NOT a gate). Compare kernel-only median vs
    # torch.matmul(f16) (cuBLAS). The plain scf.for WMMA has no cp.async pipeline
    # / L2 swizzle, so trailing cuBLAS is expected and is the named M7b work.
    try:
        Yt = torch.empty(M, N, device="cuda", dtype=torch.float16)
        Wt = W.t().contiguous()

        def torch_mm():
            torch.matmul(A, Wt, out=Yt)

        tm, tt = _bench(go), _bench(torch_mm)
        flop = 2 * M * N * K
        print(f"\nmatmul ({M},{N},{K}) grid={k.grid} block={k.block} "
              f"smem={k.shared}B: rel={rel:.2e}  mine {tm*1e3:.1f}us "
              f"({flop/(tm/1e3)/1e9:.0f} GFLOP/s)  torch.matmul {tt*1e3:.1f}us "
              f"({flop/(tt/1e3)/1e9:.0f} GFLOP/s)  mine/torch {tm/tt:.2f}  "
              f"[informational; perf + epilogue is M7b]")
    except Exception as e:  # perf datapoint is best-effort, never a gate
        print(f"\nmatmul ({M},{N},{K}): rel={rel:.2e} RUNS; perf skipped ({e})")
