"""PK the two-output fused_add_rms_norm lowering vs the REAL liger kernel.

fused_add_rms_norm (liger, casting_mode none) computes, from one read-once tile:
    S = X + R           (R = residual, a Row[M,N] input; S is the new residual)
    Y = S * rsqrt(mean(S*S) + eps) * (offset + W)
It is lowered from atomic IntentIR through the C++
``-convert-intent-row-reduction-to-gpu`` pass (ZERO new mechanism — one
last-axis mean reduce + a two-RowOut epilogue, the exact same path as rms_norm)
to ONE read-once block-cooperative sm_120a cubin with TWO destination-passing
[M,N] outputs (S and Y), both written from the register tile. S = X + R is an
[M,N] intermediate that is ALSO the reduce input (emitAt recomputes it from the
X and R register tiles — read-once of both). A separate S output buffer is used
(no in-place aliasing in M5a). Compared head-to-head with the real liger Triton
kernel for correctness (rel < 1e-3 on S and Y, vs fp32 ref AND vs liger) and
kernel-only median latency.

fused_add_rms_norm is a SINGLE-pass read-once single-block-per-row memory-bound
kernel (liger launches one program per row; reads X+R once, writes S+Y once) —
so the honest bar is PARITY (~1500 GB/s peak HBM, no HBM traffic left to
remove). A beat would require a named mechanism (none here: both read-once), so
we state parity and assert mine <= liger * 1.5 within run-to-run clock noise.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_fused_add_rms_norm_pk.py -q -s
"""

from __future__ import annotations

import os
import pathlib

import pytest

from intentir.backend.cuda.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

os.environ["PATH"] = CUDA_BIN + os.pathsep + os.environ.get("PATH", "")

pytestmark = [
    pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                       reason="intent-opt not built"),
    pytest.mark.skipif(not pathlib.Path(MLIR_OPT).exists(),
                       reason="mlir-opt not present"),
    pytest.mark.skipif(not pathlib.Path(CUDA_BIN).exists(),
                       reason="CUDA toolkit not present"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]

_PASS = "convert-intent-row-reduction-to-gpu"


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median")


def _fused_mlir(M, N):
    """Atomic fused_add_rms_norm intent DAG (parameterized M,N): S = X + R from
    the read-once X,R register tiles; one mean(S*S) reduce; two [M,N] RowOut
    outputs S (new residual) and Y = S*rstd*(offset+W)."""
    R = f"tensor<{M}x{N}xf32>"
    C = f"tensor<{N}xf32>"
    S1 = f"tensor<{M}x1xf32>"
    return f"""
func.func @fused_add_rms_norm(%X: {R}, %Resid: {R}, %W: {C}, %eps: tensor<f32>, %offset: tensor<f32>)
    -> ({R}, {R}) {{
  %S = intent.map_binary %X, %Resid {{fn = "add"}} : ({R}, {R}) -> {R}
  %sq = intent.map_binary %S, %S {{fn = "mul"}} : ({R}, {R}) -> {R}
  %ms = intent.reduce %sq {{fn = "mean", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {S1}
  %mse = intent.map_binary %ms, %eps {{fn = "add"}} : ({S1}, tensor<f32>) -> {S1}
  %rstd = intent.map_unary %mse {{fn = "rsqrt"}} : ({S1}) -> {S1}
  %Sn = intent.map_binary %S, %rstd {{fn = "mul"}} : ({R}, {S1}) -> {R}
  %Woff = intent.map_binary %W, %offset {{fn = "add"}} : ({C}, tensor<f32>) -> {C}
  %Y = intent.map_binary %Sn, %Woff {{fn = "mul"}} : ({R}, {C}) -> {R}
  return %S, %Y : {R}, {R}
}}
"""


@pytest.mark.parametrize("M,N", [(8192, 1024), (4096, 4096)])
def test_fused_add_rms_norm_pk_vs_liger(M, N):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    from liger_kernel.ops.fused_add_rms_norm import _fused_add_rms_norm_forward_kernel
    from liger_kernel.ops.utils import calculate_settings

    BLOCK_SIZE, num_warps = calculate_settings(N)
    T = num_warps * 32  # match liger's per-thread column count K = N/T
    assert N % T == 0, f"N={N} not divisible by T={T}"
    eps, offset = 1e-6, 0.0
    casting_mode = -1  # _CASTING_MODE_NONE: Y = S*rstd*(offset+W), all fp32

    torch.manual_seed(0)
    X = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    Resid = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    W = torch.randn(N, device="cuda", dtype=torch.float32).contiguous()

    # --- mine: compile + launch the read-once two-output kernel ----------- #
    k = RowReductionKernel("fused_add_rms_norm", {"M": M, "N": N}, T,
                           pass_name=_PASS, mlir_text=_fused_mlir(M, N))
    S = torch.empty(M, N, device="cuda", dtype=torch.float32)
    Y = torch.empty(M, N, device="cuda", dtype=torch.float32)
    epsd = torch.tensor(eps, device="cuda", dtype=torch.float32)
    offd = torch.tensor(offset, device="cuda", dtype=torch.float32)
    # sig order after the pass: X, R, W, eps, offset, then S, Y
    go = k.thunk([X, Resid, W, epsd, offd, S, Y])
    go()
    torch.cuda.synchronize()

    # --- fp32 torch reference --------------------------------------------- #
    S_ref = X + Resid
    rstd_ref = torch.rsqrt((S_ref * S_ref).mean(-1, keepdim=True) + eps)
    Y_ref = S_ref * rstd_ref * (offset + W)
    assert _rel(S, S_ref) < 1e-3
    assert _rel(Y, Y_ref) < 1e-3

    # --- liger: call the RAW forward kernel directly (skip autograd) ------- #
    Yl = torch.empty(M, N, device="cuda", dtype=torch.float32)
    Sl = torch.empty(M, N, device="cuda", dtype=torch.float32)
    RSTDl = torch.empty(M, device="cuda", dtype=torch.float32)

    def liger():
        _fused_add_rms_norm_forward_kernel[(M,)](
            Yl, Yl.stride(0), Sl, Sl.stride(0), X, X.stride(0),
            Resid, Resid.stride(0), W, W.stride(0), RSTDl, RSTDl.stride(0),
            N, eps, offset, casting_mode,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

    liger()
    torch.cuda.synchronize()
    # head-to-head vs the real liger kernel
    assert _rel(S, Sl) < 1e-3
    assert _rel(Y, Yl) < 1e-3

    # --- kernel-only median latency, no host sync in the timed region ----- #
    tm, tl = _bench(go), _bench(liger)
    gb = 4 * M * N * 4 / 1e9  # read X + R, write S + Y
    print(f"\nfused_add_rms_norm ({M},{N}) T={T} num_warps={num_warps}: "
          f"mine {tm*1e3:.1f}us ({gb/(tm/1e3):.0f} GB/s) "
          f"liger {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} GB/s)  mine/liger {tm/tl:.2f} "
          f"(single-pass read-once single-block-per-row -> honest PARITY floor)")
    assert tm <= tl * 1.5
