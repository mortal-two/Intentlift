"""M7b RE-ATTACK of the residual coverage modules.

Each residual FlagGems module deferred by earlier tasks is RE-ATTACKED here by
showing the numeric mechanism is already expressible in IntentIR and verifying it
against BOTH the real flag_gems kernel and an independent torch reference:

  * ``scaled_softmax``      -> softmax(scale * x) over the last axis = a
                               row-reduction softmax with a scalar pre-scale (the
                               exact subgraph our flash fwd computes internally).
  * ``rwkv_ka_fusion``      -> an elementwise fusion + a per-head grouped L2
                               normalise (a reduce over the head-channel group).
                               NOT a carried-state recurrence -- the flag_gems
                               "rwkv" fused kernels are stateless maps/reduces.
  * ``rwkv_mm_sparsity``    -> k(1D) @ v(2D) = a mat-vec contraction; the
                               "sparsity" is a perf-only zero-row skip that does
                               not change the value.
  * ``bin_topk``  (DSA)     -> top-k INDICES over a row segment = our sort/topk
                               op. flag_gems' bucket-sort kernel does NOT compile
                               on sm_120a (LLVM "Cannot select v2f16" in its
                               uint16-histogram bitcast), so torch.topk is the
                               reference (set match).
  * ``attention`` / flash   -> the register-resident flash-mma cubin (already PK'd
                               vs torch SDPA) additionally cross-checked vs
                               flag_gems.scaled_dot_product_attention, causal +
                               non-causal.

The first four run through the IntentIR reference interpreter (the executable
Layer-C semantics); the last runs the real sm_120a flash cubin. flag_gems is fed
fp16 on the GPU, the interpreter fp32 on the host, compared at fp16 tolerance.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_residual_coverage_pk.py -q -s
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
flag_gems = pytest.importorskip("flag_gems")

from intentir.ir.model import IntentFunction  # noqa: E402
from intentir.verify.back.legacy_v0.interpret import interpret  # noqa: E402

_HAS_CUDA = torch.cuda.is_available()


def _run_intent(fn_json: dict, **inputs: np.ndarray) -> dict[str, np.ndarray]:
    return interpret(IntentFunction.from_json(fn_json), inputs)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# --------------------------------------------------------------------------- #
# scaled_softmax : softmax(scale * x) over the last axis of a 4D score tensor. #
# --------------------------------------------------------------------------- #
def _scaled_softmax_intent(scale: float) -> dict:
    S = ["B", "H", "Q", "K"]
    return {
        "name": "scaled_softmax",
        "inputs": [{"name": "x", "dtype": "f32", "shape": S}],
        "outputs": [{"name": "y", "dtype": "f32", "shape": S}],
        "body": [
            {"op": "const", "result": "sc", "operands": [],
             "attrs": {"value": scale, "shape": [], "dtype": "f32"}},
            {"op": "map_binary", "result": "s", "operands": ["x", "sc"],
             "attrs": {"fn": "mul"}},
            {"op": "reduce", "result": "m", "operands": ["s"],
             "attrs": {"fn": "max", "axes": [3], "assoc": "associative"}},
            {"op": "map_binary", "result": "c", "operands": ["s", "m"],
             "attrs": {"fn": "sub"}},
            {"op": "map_unary", "result": "e", "operands": ["c"],
             "attrs": {"fn": "exp"}},
            {"op": "reduce", "result": "l", "operands": ["e"],
             "attrs": {"fn": "sum", "axes": [3], "assoc": "float_nondeterministic"}},
            {"op": "map_binary", "result": "y", "operands": ["e", "l"],
             "attrs": {"fn": "div"}},
        ],
    }


@pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA GPU")
@pytest.mark.parametrize("B,H,Q,K", [(2, 3, 16, 64), (1, 4, 32, 128)])
def test_scaled_softmax_pk(B, H, Q, K):
    torch.manual_seed(0)
    scale = 0.3
    xf = torch.randn(B, H, Q, K, dtype=torch.float32)
    x16 = xf.half().cuda()

    fg = flag_gems.scaled_softmax_forward(x16, scale).float().cpu().numpy()
    mine = _run_intent(_scaled_softmax_intent(scale), x=xf.numpy())["y"]
    ref = torch.softmax(xf * scale, dim=-1).numpy()

    r_fg, r_ref = _rel(mine, fg), _rel(mine, ref)
    print(f"\nscaled_softmax ({B},{H},{Q},{K}): rel_flaggems={r_fg:.2e} "
          f"rel_torch={r_ref:.2e}")
    assert r_ref < 1e-5
    assert r_fg < 2e-2


# --------------------------------------------------------------------------- #
# rwkv_ka_fusion : elementwise fusion + per-head grouped L2 normalise.         #
# Inputs are fed pre-reshaped to the grouped [T,H,N] layout (reshape is a       #
# layout no-op); kk/ka broadcast over T as [1,H,N].                             #
# --------------------------------------------------------------------------- #
def _rwkv_ka_intent() -> dict:
    G = ["T", "H", "N"]
    B = ["1", "H", "N"]
    return {
        "name": "rwkv_ka_fusion",
        "inputs": [
            {"name": "k", "dtype": "f32", "shape": G},
            {"name": "kk", "dtype": "f32", "shape": B},
            {"name": "a", "dtype": "f32", "shape": G},
            {"name": "ka", "dtype": "f32", "shape": B},
        ],
        "outputs": [
            {"name": "ok", "dtype": "f32", "shape": G},
            {"name": "okk", "dtype": "f32", "shape": G},
            {"name": "okka", "dtype": "f32", "shape": G},
        ],
        "body": [
            {"op": "const", "result": "one", "operands": [],
             "attrs": {"value": 1.0, "shape": [], "dtype": "f32"}},
            {"op": "const", "result": "eps", "operands": [],
             "attrs": {"value": 1e-12, "shape": [], "dtype": "f32"}},
            # okk = (k*kk) / sqrt(sum_N (k*kk)^2 + eps)
            {"op": "map_binary", "result": "kt", "operands": ["k", "kk"],
             "attrs": {"fn": "mul"}},
            {"op": "map_binary", "result": "kt2", "operands": ["kt", "kt"],
             "attrs": {"fn": "mul"}},
            {"op": "reduce", "result": "n2", "operands": ["kt2"],
             "attrs": {"fn": "sum", "axes": [2], "assoc": "float_nondeterministic"}},
            {"op": "map_binary", "result": "n2e", "operands": ["n2", "eps"],
             "attrs": {"fn": "add"}},
            {"op": "map_unary", "result": "nrm", "operands": ["n2e"],
             "attrs": {"fn": "sqrt"}},
            {"op": "map_binary", "result": "okk", "operands": ["kt", "nrm"],
             "attrs": {"fn": "div"}},
            # ok = k * (1 + (a - 1) * ka)
            {"op": "map_binary", "result": "am1", "operands": ["a", "one"],
             "attrs": {"fn": "sub"}},
            {"op": "map_binary", "result": "g", "operands": ["am1", "ka"],
             "attrs": {"fn": "mul"}},
            {"op": "map_binary", "result": "g1", "operands": ["g", "one"],
             "attrs": {"fn": "add"}},
            {"op": "map_binary", "result": "ok", "operands": ["k", "g1"],
             "attrs": {"fn": "mul"}},
            # okka = okk * a
            {"op": "map_binary", "result": "okka", "operands": ["okk", "a"],
             "attrs": {"fn": "mul"}},
        ],
    }


@pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA GPU")
@pytest.mark.parametrize("T,H,N", [(4, 2, 4), (8, 3, 16)])
def test_rwkv_ka_fusion_pk(T, H, N):
    torch.manual_seed(0)
    C = H * N
    kf = torch.randn(T, C, dtype=torch.float32)
    kkf = torch.randn(C, dtype=torch.float32)
    af = torch.randn(T, C, dtype=torch.float32)
    kaf = torch.randn(C, dtype=torch.float32)

    ok_g, okk_g, okka_g = flag_gems.rwkv_ka_fusion(
        kf.half().cuda(), kkf.half().cuda(), af.half().cuda(), kaf.half().cuda(),
        H, N)

    out = _run_intent(
        _rwkv_ka_intent(),
        k=kf.numpy().reshape(T, H, N),
        kk=kkf.numpy().reshape(1, H, N),
        a=af.numpy().reshape(T, H, N),
        ka=kaf.numpy().reshape(1, H, N),
    )
    ok = out["ok"].reshape(T, C)
    okk = out["okk"].reshape(T, C)
    okka = out["okka"].reshape(T, C)

    r = [_rel(ok, ok_g.float().cpu().numpy()),
         _rel(okk, okk_g.float().cpu().numpy()),
         _rel(okka, okka_g.float().cpu().numpy())]
    print(f"\nrwkv_ka_fusion ({T},{H},{N}): rel_flaggems ok/okk/okka="
          f"{r[0]:.2e}/{r[1]:.2e}/{r[2]:.2e}")
    assert max(r) < 2e-2


# --------------------------------------------------------------------------- #
# rwkv_mm_sparsity : out[m] = sum_k k[k] * v[k,m] = mat-vec contraction.       #
# --------------------------------------------------------------------------- #
def _rwkv_mm_intent() -> dict:
    return {
        "name": "rwkv_mm_sparsity",
        "inputs": [
            {"name": "k", "dtype": "f32", "shape": ["K"]},
            {"name": "v", "dtype": "f32", "shape": ["K", "M"]},
        ],
        "outputs": [{"name": "o", "dtype": "f32", "shape": ["M"]}],
        "body": [
            {"op": "contraction", "result": "o", "operands": ["k", "v"],
             "attrs": {"subscripts": "k,km->m"}},
        ],
    }


@pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA GPU")
# NB: use power-of-2 K. flag_gems' kernel masks its loads against
# next_power_of_2(K), so for a non-pow2 K it reads (K_pow2 - K) elements PAST the
# real k/v tensors -- an out-of-bounds read in flag_gems itself that yields
# nondeterministic garbage. Our mechanism matches torch exactly at any K
# (rel_torch < 1e-5 below); pow2 K just avoids flag_gems' own OOB for the
# cross-check.
@pytest.mark.parametrize("K,M", [(1024, 32), (2048, 64)])
def test_rwkv_mm_sparsity_pk(K, M):
    torch.manual_seed(0)
    kf = torch.randn(K, dtype=torch.float32)
    kf[kf.abs() < 0.3] = 0.0  # inject sparsity (perf-only skip in flag_gems)
    vf = torch.randn(K, M, dtype=torch.float32)

    o_g = flag_gems.rwkv_mm_sparsity(kf.half().cuda(), vf.half().cuda())
    mine = _run_intent(_rwkv_mm_intent(), k=kf.numpy(), v=vf.numpy())["o"]
    ref = (kf @ vf).numpy()

    r_fg, r_ref = _rel(mine, o_g.float().cpu().numpy()), _rel(mine, ref)
    print(f"\nrwkv_mm_sparsity ({K},{M}): rel_flaggems={r_fg:.2e} rel_torch={r_ref:.2e}")
    assert r_ref < 1e-5
    assert r_fg < 2e-2


# --------------------------------------------------------------------------- #
# bin_topk (DSA) : top-k indices over a row segment = our sort/topk op.        #
# flag_gems' bucket-sort kernel fails LLVM sm_120a codegen, so torch is ref.   #
# --------------------------------------------------------------------------- #
def _topk_indices_intent(k: int) -> dict:
    return {
        "name": "bin_topk",
        "inputs": [{"name": "x", "dtype": "f32", "shape": ["B", "S"]}],
        "outputs": [{"name": "idx", "dtype": "i32", "shape": ["B", "K"]}],
        "body": [
            {"op": "sort", "result": "idx", "operands": ["x"],
             "attrs": {"axis": 1, "descending": True, "k": k, "return": "indices"}},
        ],
    }


@pytest.mark.parametrize("B,S,k", [(2, 512, 8), (4, 256, 16)])
def test_bin_topk_pk(B, S, k):
    torch.manual_seed(0)
    xf = torch.randn(B, S, dtype=torch.float32)
    mine = _run_intent(_topk_indices_intent(k), x=xf.numpy())["idx"]
    ref = torch.topk(xf, k, dim=-1).indices.numpy()
    # Bucket top-k returns the top-k SET (order is algorithm-defined, not by value).
    ok = all(set(mine[b].tolist()) == set(ref[b].tolist()) for b in range(B))
    print(f"\nbin_topk ({B},{S},k={k}): top-k set match vs torch.topk = {ok}")
    assert ok


# --------------------------------------------------------------------------- #
# attention : the register-resident flash-mma cubin cross-checked vs the real  #
# flag_gems.scaled_dot_product_attention, causal + non-causal.                 #
# --------------------------------------------------------------------------- #
try:
    from intentir.backend.cuda.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT
    _FLASH_READY = (pathlib.Path(INTENT_OPT).exists()
                    and pathlib.Path(MLIR_OPT).exists()
                    and pathlib.Path(CUDA_BIN).exists() and _HAS_CUDA)
except Exception:
    _FLASH_READY = False


@pytest.mark.skipif(not _FLASH_READY, reason="flash toolchain / GPU not present")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sq,sk,d", [(128, 128, 64), (256, 256, 64)])
def test_attention_flash_vs_flaggems_pk(sq, sk, d, causal):
    from test_flash_fwd_mma_pk import _FlashKernel, _flash_intent, _rel as _relt
    from test_flash_fwd_causal_pk import _FlashCausalKernel

    torch.manual_seed(0)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    scale = 1.0 / (d ** 0.5)

    kern = (_FlashCausalKernel if causal else _FlashKernel)(_flash_intent(sq, sk, d))
    o = torch.empty(sq, d, device="cuda", dtype=torch.float32).contiguous()
    kern.run(q, k, v, scale, o)
    torch.cuda.synchronize()

    # flag_gems' own SDPA (the "attention" module's public entry) as the reference.
    fg = flag_gems.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None], scale=scale, is_causal=causal
    )[0, 0]

    rel = _relt(o, fg)
    print(f"\nattention flash-mma ({sq},{sk},{d}) causal={causal}: "
          f"rel_flaggems_sdpa={rel:.3e}")
    assert rel < 2e-2
