"""PK the fused-epilogue tensor-core linear kernel vs the REAL Triton baseline.

The M7b deliverable: ``intent.contraction{"mk,nk->mn"}`` (A[M,K] f16, W[N,K] f16
-> Y[M,N] f32) FOLLOWED BY a pointwise epilogue (bias-add against a Col[N] arg +
an activation -- squared_relu / gelu_approx, expressed in the scalar vocab) is
lowered by ``-convert-intent-contraction-to-gpu`` to ONE tiled-K tensor-core
``gpu.launch``. The 4 per-warp COp accumulator fragments are spilled to a
dynamic-shared [BM,BN] staging tile, the block reloads each element and
evaluates the bias-add + activation DAG (reusing MapEmit.h scalar emitters),
and writes Y.

Two correctness oracles per (activation, bias):
  * an fp32 reference  activation(A.float() @ W.float().T + bias.float())  with
    NO mid-pipeline cast (Triton casts only on store), and
  * the REAL flash_attn ``linear.kernel_fwd`` (tl.dot), driven as a bare
    ``@triton.jit`` with a fixed config (Triton 3.6 removed
    ``triton.ops.matmul_perf_model`` and rejects the library's non-constexpr
    activation globals, so the autotune wrapper cannot be used as-is -- the bare
    jit is the honest head-to-head, NOT cuBLAS).

The kernel writes f32; Triton writes f16 (cast-on-store). They are compared
upcast-to-fp32 with the f16 tolerance rel < 2e-2 (the spec's bar) -- this is a
tolerance match, NOT a bit-exact claim.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_linear_pk.py -q -s
"""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
import sys
import types

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

_SQRT2PI = math.sqrt(2.0 / math.pi)


# --- load the REAL flash_attn linear.kernel_fwd standalone -------------------
# flash_attn/__init__.py imports a CUDA extension we don't have, and Triton 3.6
# removed triton.ops.matmul_perf_model, so we stub the package chain + the
# perf-model dep and exec the two relevant source files directly.
def _load_triton_kernel():
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
    base = "/home/kingdom/phdworks/Dataset/flash-attention/flash_attn"
    if not pathlib.Path(base).exists():
        return None
    triton = pytest.importorskip("triton")
    ops = types.ModuleType("triton.ops")
    sys.modules.setdefault("triton.ops", ops)
    mm = types.ModuleType("triton.ops.matmul_perf_model")
    mm.early_config_prune = lambda *a, **k: (a[0] if a else None)
    mm.estimate_matmul_time = lambda *a, **k: 1.0
    sys.modules.setdefault("triton.ops.matmul_perf_model", mm)
    for name, path in [("flash_attn", base), ("flash_attn.ops", base + "/ops"),
                       ("flash_attn.ops.triton", base + "/ops/triton")]:
        if name not in sys.modules:
            mo = types.ModuleType(name)
            mo.__path__ = [path]
            sys.modules[name] = mo

    def _exec(n, f):
        spec = importlib.util.spec_from_file_location(n, f)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[n] = mod
        spec.loader.exec_module(mod)
        return mod

    _exec("flash_attn.ops.triton.k_activations",
          base + "/ops/triton/k_activations.py")
    lin = _exec("flash_attn.ops.triton.linear", base + "/ops/triton/linear.py")
    return lin.kernel_fwd.fn.fn  # the bare JITFunction under heuristics+autotune


_JIT = _load_triton_kernel()

# A small config menu; per shape we pick the fastest that compiles (autotune-ish
# over the strong tl.dot configs). All are SPLIT_K=1, fp32 accumulate.
_CFGS = [
    (128, 256, 32, 8, 3), (256, 128, 32, 8, 3), (128, 128, 32, 4, 3),
    (128, 64, 32, 4, 4), (64, 128, 32, 4, 4), (128, 128, 64, 4, 4),
]


def _triton_run(jit, x, w, bias, act, cfg):
    import triton
    M, K = x.shape
    N = w.shape[0]
    BM, BN, BK, nw, ns = cfg
    out = torch.empty((M, N), device="cuda", dtype=x.dtype)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    jit[grid](out, x, x, w, bias if bias is not None else x,
              M, N, K, M // 32, N // 32, K // 32,
              out.stride(0), x.stride(0), x.stride(1), w.stride(0), w.stride(1),
              BLOCK_M=BM, GROUP_M=8, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=1,
              EVEN_K=(K % BK == 0), A_ROWMAJOR=(x.stride(1) == 1),
              B_COLMAJOR=(w.stride(1) == 1), BIAS=bias is not None,
              SAVE_ACT_INPUT=False, ACTIVATION=act, num_warps=nw, num_stages=ns)
    return out


def _bench(fn, warmup=50, rep=100):
    import triton
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep,
                                   return_mode="median")


def _best_triton(jit, x, w, bias, act):
    """Pick the fastest compiling config -> (closure, time_s, config)."""
    best = None
    for cfg in _CFGS:
        try:
            _triton_run(jit, x, w, bias, act, cfg)  # compile/smoke
            t = _bench(lambda: _triton_run(jit, x, w, bias, act, cfg),
                       warmup=20, rep=40)
        except Exception:
            continue
        if best is None or t < best[1]:
            best = (lambda c=cfg: _triton_run(jit, x, w, bias, act, c), t, cfg)
    if best is None:
        return None
    # re-bench the winner properly
    tgo, _, cfg = best
    return (tgo, _bench(tgo), cfg)


_FIXED = (128, 128, 32, 4, 3)  # a config that compiles for every spec shape


# --- intent MLIR builders ----------------------------------------------------
def _act_chain(x, act, MN):
    """Return (epilogue body text, final-ssa-name) for `act` applied to `x`."""
    t = f"tensor<{MN}xf32>"
    if act == "id":
        return "", x
    if act == "squared_relu":
        body = f"""  %z = arith.constant dense<0.0> : {t}
  %c = intent.map_binary {x}, %z {{fn = "ge"}} : ({t}, {t}) -> tensor<{MN}xi1>
  %r = intent.select %c, {x}, %z : (tensor<{MN}xi1>, {t}, {t}) -> {t}
  %sq = intent.map_binary %r, %r {{fn = "mul"}} : ({t}, {t}) -> {t}
"""
        return body, "%sq"
    if act == "gelu_approx":
        body = f"""  %c1 = arith.constant dense<{_SQRT2PI!r}> : {t}
  %c2 = arith.constant dense<0.044715> : {t}
  %half = arith.constant dense<0.5> : {t}
  %one = arith.constant dense<1.0> : {t}
  %xx = intent.map_binary {x}, {x} {{fn = "mul"}} : ({t}, {t}) -> {t}
  %g1 = intent.map_binary %xx, %c2 {{fn = "mul"}} : ({t}, {t}) -> {t}
  %g2 = intent.map_binary %g1, %one {{fn = "add"}} : ({t}, {t}) -> {t}
  %g3 = intent.map_binary {x}, %c1 {{fn = "mul"}} : ({t}, {t}) -> {t}
  %g4 = intent.map_binary %g3, %g2 {{fn = "mul"}} : ({t}, {t}) -> {t}
  %gt = intent.map_unary %g4 {{fn = "tanh"}} : ({t}) -> {t}
  %g5 = intent.map_binary %gt, %one {{fn = "add"}} : ({t}, {t}) -> {t}
  %g6 = intent.map_binary {x}, %half {{fn = "mul"}} : ({t}, {t}) -> {t}
  %g  = intent.map_binary %g6, %g5 {{fn = "mul"}} : ({t}, {t}) -> {t}
"""
        return body, "%g"
    raise ValueError(act)


def _mlir(M, N, K, act, bias):
    A, W, Y = f"tensor<{M}x{K}xf16>", f"tensor<{N}x{K}xf16>", f"tensor<{M}x{N}xf32>"
    MN = f"{M}x{N}"
    args = f"%A: {A}, %W: {W}"
    if bias:
        args += f", %bias: tensor<{N}xf32>"
    head = f"""func.func @lin({args}) -> {Y} {{
  %Y = intent.contraction %A, %W {{subscripts = "mk,nk->mn"}} : ({A}, {W}) -> {Y}
"""
    base = "%Y"
    if bias:
        head += (f'  %b = intent.map_binary %Y, %bias {{fn = "add"}} : '
                 f'({Y}, tensor<{N}xf32>) -> {Y}\n')
        base = "%b"
    body, last = _act_chain(base, act, MN)
    return head + body + f"  return {last} : {Y}\n}}\n"


def _ref(x, w, bias, act):
    rl = x.float() @ w.float().t()
    if bias is not None:
        rl = rl + bias.float()
    if act == "id":
        return rl
    if act == "squared_relu":
        return torch.relu(rl) ** 2
    return 0.5 * rl * (1.0 + torch.tanh(_SQRT2PI * (rl + 0.044715 * rl ** 3)))


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _build_and_run(M, N, K, act, bias_on):
    """Compile the fused-epilogue kernel, run it once, return (Y, A, W, bias)."""
    from intentir.benchmarks.row_reduction import MatmulKernel

    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    W = torch.randn(N, K, device="cuda", dtype=torch.float16).contiguous()
    bias = (torch.randn(N, device="cuda", dtype=torch.float16).contiguous()
            if bias_on else None)
    Y = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    # My epilogue is f32 (the contraction accumulates f32, bias-add in f32), so
    # my kernel takes an f32 bias; Triton takes the f16 bias (same values, the
    # f16->f32 widening is exact). The fp32 ref uses bias.float() == bias_f32.
    bias_f32 = bias.float().contiguous() if bias_on else None
    k = MatmulKernel(_mlir(M, N, K, act, bias_on))
    tensors = [A, W, bias_f32, Y] if bias_on else [A, W, Y]
    go = k.thunk(tensors)
    go()
    torch.cuda.synchronize()
    return go, Y, A, W, bias


@pytest.mark.parametrize("act", ["id", "squared_relu", "gelu_approx"])
@pytest.mark.parametrize("bias_on", [False, True])
def test_linear_epilogue_correctness(act, bias_on):
    """Correctness of the fused epilogue vs an fp32 ref AND the real Triton
    kernel_fwd (single fixed tl.dot config), id/squared_relu/gelu_approx x
    bias{on,off}. Tolerance match (f16) rel < 2e-2; observed ~1e-6 vs fp32."""
    M = N = K = 512
    go, Y, A, W, bias = _build_and_run(M, N, K, act, bias_on)

    # Oracle 1: fp32 reference (NO mid-pipeline cast; Triton casts only on store).
    ref = _ref(A, W, bias, act)
    rel_ref = _rel(Y, ref)
    assert rel_ref < 2e-2, f"{act} bias={bias_on} vs fp32 rel={rel_ref:.3e}"

    # Oracle 2: the REAL Triton kernel_fwd (tl.dot), upcast f16->f32.
    rel_tri = None
    if _JIT is not None:
        yt = _triton_run(_JIT, A, W, bias, act, _FIXED)
        torch.cuda.synchronize()
        rel_tri = _rel(Y, yt.float())
        assert rel_tri < 2e-2, f"{act} bias={bias_on} vs triton rel={rel_tri:.3e}"
    tri = f"  vs-triton rel={rel_tri:.2e}" if rel_tri is not None else ""
    print(f"\nlinear {act} bias={bias_on} (512^3): vs-fp32 rel={rel_ref:.2e}{tri}")


# The honest head-to-head (STEP 2): mine vs the REAL tl.dot kernel_fwd (NOT
# cuBLAS), kernel-only median, no host sync in the timed region. The spec shapes:
# square 512^3, big 4096^3, K-bound 1024^2 x 8192.
@pytest.mark.skipif(_JIT is None, reason="flash_attn linear source not present")
@pytest.mark.parametrize("M,N,K", [(512, 512, 512), (1024, 1024, 8192),
                                   (4096, 4096, 4096)])
def test_linear_pk_perf(M, N, K):
    act, bias_on = "id", False
    go, Y, A, W, bias = _build_and_run(M, N, K, act, bias_on)
    rel_ref = _rel(Y, _ref(A, W, bias, act))
    assert rel_ref < 2e-2, f"({M},{N},{K}) vs fp32 rel={rel_ref:.3e}"

    best = _best_triton(_JIT, A, W, bias, act)
    assert best is not None, "no triton config compiled"
    tgo, tt, cfg = best
    rel_tri = _rel(Y, tgo().float())
    assert rel_tri < 2e-2, f"({M},{N},{K}) vs triton rel={rel_tri:.3e}"
    tm = _bench(go)
    flop = 2 * M * N * K
    print(f"\nPK ({M},{N},{K}): mine {tm*1e3:.1f}us ({flop/(tm/1e3)/1e9:.0f} GF/s)"
          f"  tl.dot {tt*1e3:.1f}us ({flop/(tt/1e3)/1e9:.0f} GF/s) cfg{cfg}  "
          f"mine/tl.dot {tm/tt:.2f}  rel(fp32)={rel_ref:.1e} rel(triton)={rel_tri:.1e}")
