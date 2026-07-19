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
import re

import pytest

from intentir.backend.compile import (CUDA_BIN, INTENT_OPT, MLIR_OPT,
                                      _unescape_mlir_bytes)

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


def _mlir(M, N, K, storage_type="f16", explicit_widen=False):
    A = f"tensor<{M}x{K}x{storage_type}>"
    W = f"tensor<{N}x{K}x{storage_type}>"
    Y = f"tensor<{M}x{N}xf32>"
    if explicit_widen:
        Af, Wf = f"tensor<{M}x{K}xf32>", f"tensor<{N}x{K}xf32>"
        body = f"""
  %Af = intent.cast %A {{to_dtype = "f32"}} : ({A}) -> {Af}
  %Wf = intent.cast %W {{to_dtype = "f32"}} : ({W}) -> {Wf}
  %Y = intent.contraction %Af, %Wf {{subscripts = "mk,nk->mn"}} : ({Af}, {Wf}) -> {Y}
"""
    else:
        body = f"""
  %Y = intent.contraction %A, %W {{subscripts = "mk,nk->mn"}} : ({A}, {W}) -> {Y}
"""
    return f"""
func.func @matmul(%A: {A}, %W: {W}) -> {Y} {{
{body}
  return %Y : {Y}
}}
"""


def _saved_bf16_mlir(M, N, K):
    A, W = f"tensor<{M}x{K}xbf16>", f"tensor<{N}x{K}xbf16>"
    Af, Wf = f"tensor<{M}x{K}xf32>", f"tensor<{N}x{K}xf32>"
    Bias, Biasf = f"tensor<{N}xbf16>", f"tensor<{N}xf32>"
    Yf, Y = f"tensor<{M}x{N}xf32>", f"tensor<{M}x{N}xbf16>"
    return f"""
func.func @bf16_linear_saved(%A: {A}, %W: {W}, %bias: {Bias}) -> ({Y}, {Y}) {{
  %Af = intent.cast %A {{to_dtype = "f32"}} : ({A}) -> {Af}
  %Wf = intent.cast %W {{to_dtype = "f32"}} : ({W}) -> {Wf}
  %acc = intent.contraction %Af, %Wf {{subscripts = "mk,nk->mn"}} : ({Af}, {Wf}) -> {Yf}
  %biasf = intent.cast %bias {{to_dtype = "f32"}} : ({Bias}) -> {Biasf}
  %pre = intent.map_binary %acc, %biasf {{fn = "add"}} : ({Yf}, {Biasf}) -> {Yf}
  %out = intent.cast %pre {{to_dtype = "bf16"}} : ({Yf}) -> {Y}
  %saved = intent.cast %pre {{to_dtype = "bf16"}} : ({Yf}) -> {Y}
  return %out, %saved : {Y}, {Y}
}}
"""


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _bench(fn):
    import triton
    return triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median")


def _ptx(kernel):
    assembly = re.search(
        r'assembly = "((?:[^"\\]|\\.)*)"', kernel.binary_mlir
    )
    assert assembly is not None
    return _unescape_mlir_bytes(assembly.group(1)).decode()


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


@pytest.mark.parametrize(
    "M,N,K,explicit_widen",
    [
        (64, 64, 32, False),
        (128, 256, 64, True),
        (1024, 1024, 1024, False),
    ],
)
def test_bf16_matmul_mmasync_pk(M, N, K, explicit_widen):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device has no BF16 tensor-core support")

    from intentir.benchmarks.row_reduction import MatmulKernel

    torch.manual_seed(1)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).contiguous()
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).contiguous()
    Y = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    kernel = MatmulKernel(
        _mlir(M, N, K, "bf16", explicit_widen=explicit_widen), mma=True
    )
    go = kernel.thunk([A, W, Y])
    go()
    torch.cuda.synchronize()

    reference = torch.mm(A, W.t(), out_dtype=torch.float32)
    rel = _rel(Y, reference)
    assert rel < 1.0e-3, f"BF16 ({M},{N},{K}) rel={rel:.3e}"

    ptx = _ptx(kernel)
    bf16_opcode = (
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
    )
    assert bf16_opcode in ptx
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" not in ptx
    assert "ldmatrix.sync.aligned.m8n8.x4.shared.b16" in ptx
    assert "ldmatrix.sync.aligned.m8n8.x2.shared.b16" in ptx

    if (M, N, K) == (1024, 1024, 1024):
        A16 = A.to(torch.float16)
        W16 = W.to(torch.float16)
        Y16 = torch.empty_like(Y)
        f16_kernel = MatmulKernel(_mlir(M, N, K, "f16"), mma=True)
        f16_go = f16_kernel.thunk([A16, W16, Y16])
        f16_go()
        torch.cuda.synchronize()

        torch_output = torch.empty_like(Y)
        Wt = W.t()

        def torch_bf16_f32():
            torch.mm(A, Wt, out=torch_output, out_dtype=torch.float32)

        ours_ms = _bench(go)
        f16_ms = _bench(f16_go)
        torch_ms = _bench(torch_bf16_f32)
        flop = 2 * M * N * K
        print(
            f"\nbf16 mma.sync ({M},{N},{K}): rel={rel:.2e} "
            f"ours {ours_ms*1e3:.1f}us "
            f"({flop/(ours_ms/1e3)/1e9:.0f} GFLOP/s)  "
            f"f16-same-backend {f16_ms*1e3:.1f}us "
            f"(bf16/f16 {ours_ms/f16_ms:.2f})  "
            f"torch-bf16-f32 {torch_ms*1e3:.1f}us "
            f"(ours/torch {ours_ms/torch_ms:.2f})"
        )
        assert ours_ms <= f16_ms * 1.25


def test_bf16_saved_bias_epilogue_stays_in_one_mma_kernel():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device has no BF16 tensor-core support")

    from intentir.benchmarks.row_reduction import MatmulKernel

    M, N, K = 64, 64, 32
    torch.manual_seed(2)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).contiguous()
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).contiguous()
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16).contiguous()
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    saved = torch.empty_like(out)

    kernel = MatmulKernel(_saved_bf16_mlir(M, N, K), mma=True)
    go = kernel.thunk([A, W, bias, out, saved])
    go()
    torch.cuda.synchronize()

    reference = (torch.mm(A, W.t(), out_dtype=torch.float32) + bias.float()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(out, reference, rtol=0.0, atol=0.0)
    torch.testing.assert_close(saved, reference, rtol=0.0, atol=0.0)
