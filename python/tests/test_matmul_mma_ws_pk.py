"""PK the M7g WARP-SPECIALIZED (producer/consumer) nvgpu.mma.sync matmul path.

The M7g experiment: split the block's warps into PRODUCER warps (issue cp.async
loads into an N-stage shared buffer, signal a per-stage `full` mbarrier, wait the
matching `empty` mbarrier) and CONSUMER warps (wait `full`, run ldmatrix +
mma.sync into the register accumulator, signal `empty`), decoupling load-issue
from compute the way tl.dot/cuBLAS hide load latency. Selected by
``-convert-intent-contraction-to-gpu-mma-ws`` (MatmulKernel(warp_spec=True)); the
M7f mma.sync path and the WMMA path are kept 100% intact.

FINDINGS (the honest M7g verdict, recorded here so the result is durable):

* GATE G5 PASSED: the nvgpu.mbarrier primitives (init/arrive/try_wait.parity)
  AND nvvm.barrier (named/partial bar.sync id,count) both lower to PTX and
  EXECUTE correctly on sm_120a -- a producer warp's shared writes are observed by
  a decoupled consumer warp. Warp-spec is expressible at this IR layer.

* The warp-specialized matmul is CORRECT (rel < 2e-2 vs torch.matmul) -- but only
  at OCCUPANCY 1, i.e. grid <= the SM count (170 on the 5090D). The shapes tested
  here are all single-wave (grid <= 64) for exactly this reason.

* FATAL LIMITATION (the ceiling): at occupancy >= 2 (grid > 170, which is every
  large target shape: 1024^2x8192 grid=256, 4096^3 grid=4096) the spin-wait
  producer/consumer DEADLOCKS -- co-resident blocks' spinning consumer warps
  starve the producer warp of issue slots (no hardware forward-progress
  guarantee on sm_120a; consumer Blackwell lacks Hopper's warpgroup/TMA). The
  failure threshold is exactly the SM count. The M7f path (gpu.barrier, no spin)
  runs fine at all grids.

* Even where it runs (512^3, grid 64), warp-spec is SLOWER, not faster: ~16.4us
  (mine/tl.dot ~1.49) vs the M7f mma.sync baseline 10.2us (mine/tl.dot ~0.93) --
  a single producer warp bottlenecks the loads and the extra warp costs
  occupancy. Warp-spec did NOT close the residual; the M7f mma.sync result
  (best-tuned bm64bn64ns4: 0.93 / 1.30 / 1.19 vs tl.dot) remains the established
  compute ceiling on sm_120a.

This test therefore only asserts CORRECTNESS on single-wave shapes (running the
deadlock-prone large shapes here would hang CI). Run:
  export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_matmul_mma_ws_pk.py -q -s
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


# Single-wave shapes only (grid = (M/64)*(N/64) <= 170): the warp-spec mainloop
# is correct at occupancy 1; it deadlocks at occupancy >= 2 (see module docstring).
@pytest.mark.parametrize("M,N,K,ns,pw", [
    (64, 64, 64, 2, 1),        # one block, shallow pipeline
    (64, 64, 2048, 3, 1),      # one block, deep K (numK=64)
    (128, 128, 128, 3, 1),     # 4 blocks
    (512, 512, 512, 3, 1),     # 64 blocks (still single-wave on 170 SMs)
])
def test_matmul_mma_ws_correct(M, N, K, ns, pw):
    from intentir.benchmarks.row_reduction import MatmulKernel

    grid_blocks = (M // 64) * (N // 64)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    if grid_blocks > sm:
        pytest.skip(f"grid {grid_blocks} > {sm} SMs: warp-spec deadlocks at "
                    f"occupancy>=2 (the M7g ceiling)")

    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    W = torch.randn(N, K, device="cuda", dtype=torch.float16).contiguous()
    Y = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    k = MatmulKernel(_mlir(M, N, K), num_stages=ns, block_m=64, block_n=64,
                     warp_spec=True, num_producer_warps=pw)
    go = k.thunk([A, W, Y])
    go()
    torch.cuda.synchronize()

    ref = torch.matmul(A.float(), W.float().t())
    rel = _rel(Y, ref)
    assert rel < 2e-2, f"({M},{N},{K}) ns{ns} pw{pw} rel={rel:.3e}"
    print(f"\nWS ({M},{N},{K}) ns{ns} pw{pw} grid={k.grid} block={k.block} "
          f"smem={k.shared}B: rel={rel:.2e}  CORRECT (occupancy-1)")
