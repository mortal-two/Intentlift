"""PK the nvgpu.ldmatrix + nvgpu.mma.sync matmul path vs torch.matmul + the WMMA
path (gpu.subgroup_mma_*).

The M7f deliverable: the same intent.contraction{"mk,nk->mn"} (A[M,K] f16,
W[N,K] f16 -> Y[M,N] f32) lowered by ``-convert-intent-contraction-to-gpu-mma``
to ONE tiled-K gpu.launch whose contraction is realized at the REGISTER level --
nvgpu.ldmatrix (shared->register warp-cooperative fragment load) + nvgpu.mma.sync
(m16n8k16, KNOWN lane layout -> vector<2x2xf32> accumulator carried across the K
loop as scf.for iter_args). This is the parallel path explored for full
compute-bound parity; the WMMA gpu.mma_matrix pass is kept 100% intact and
remains the default (MatmulKernel(mma=False)).

Bar = CORRECTNESS + RUNS on sm_120a (rel under the f16 tolerance vs
torch.matmul(A, W.t())). Perf is an INFORMATIONAL datapoint: empirically the
mma.sync path reaches parity with the WMMA path at 512^3 and modestly improves
the large-shape ratios (4096^3: 1.25 -> ~1.19 vs tl.dot) but does NOT reach
parity (<=1.05); the compute-bound gap to tl.dot persists.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_matmul_mmasync_pk.py -q -s
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
    (64, 64, 32),       # one block, one K-trip
    (128, 128, 64),     # multi-block, multi-trip
    (512, 512, 512),
    (1024, 1024, 1024),
])
def test_matmul_mmasync_pk(M, N, K):
    from intentir.benchmarks.row_reduction import MatmulKernel

    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    W = torch.randn(N, K, device="cuda", dtype=torch.float16).contiguous()
    Y = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    k = MatmulKernel(_mlir(M, N, K), mma=True)
    go = k.thunk([A, W, Y])
    go()
    torch.cuda.synchronize()

    ref = torch.matmul(A.float(), W.float().t())
    rel = _rel(Y, ref)
    assert rel < 2e-2, f"({M},{N},{K}) rel={rel:.3e}"

    # Informational perf datapoint (NOT a gate): mma.sync vs the WMMA path.
    try:
        Yw = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        kw = MatmulKernel(_mlir(M, N, K), mma=False)
        gw = kw.thunk([A, W, Yw])
        gw()
        torch.cuda.synchronize()
        relw = _rel(Yw, ref)
        tm, tw = _bench(go), _bench(gw)
        flop = 2 * M * N * K
        print(f"\nmma.sync ({M},{N},{K}) grid={k.grid} block={k.block} "
              f"smem={k.shared}B: rel={rel:.2e}  mma.sync {tm*1e3:.1f}us "
              f"({flop/(tm/1e3)/1e9:.0f} GFLOP/s)  wmma {tw*1e3:.1f}us "
              f"(rel={relw:.1e})  mma/wmma {tm/tw:.2f}  [informational]")
    except Exception as e:
        print(f"\nmma.sync ({M},{N},{K}): rel={rel:.2e} RUNS; perf skipped ({e})")
