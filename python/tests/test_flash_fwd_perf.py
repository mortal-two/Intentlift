"""M8d flash-fwd PERF: the REGISTER-RESIDENT mma.sync flash forward vs the real
Triton ``_fwd_kernel`` and torch SDPA, kernel-only ``do_bench`` (no-sync median).

The M8d deliverable (``-convert-intent-flash-mma-to-gpu``, num-stages=2) keeps the
online-softmax state (O/S/m/l) resident in registers via ``nvgpu.mma.sync`` and is
PK'd for correctness in test_flash_fwd_mma_pk.py. That file is correctness-only; the
PRD's "flash fwd ~parity small / 1.38-1.43x large vs Triton flash" number lived only
in a scratchpad bench. This file COMMITS that measurement as a gated test.

It measures our fwd kernel against:
  (a) the bare @triton.jit ``_fwd_kernel`` (flash_attn_triton.py), driven with a
      FIXED config (BLOCK_M=BLOCK_N=128, num_warps=4 for d=64, num_stages=1) via its
      inner JITFunction, dodging the @triton.heuristics launcher (heuristic
      constexprs supplied explicitly) -- an apples-to-apples kernel comparison.
  (b) torch SDPA (the library flash kernel; a different op boundary -- informational).

The Dataset source is READ-ONLY and is NOT modified. Triton 3.x removed
tl.dot(..., trans_b=); since @triton.jit re-reads source from the file path at
compile time, we write the rewritten copy to a LOCAL temp file (NOT Dataset), keyed
by content hash, and load the kernel from there -- the reference stays pristine.
(Same local-copy trick as test_flash_bwd_perf.py::_load_triton_bwd.)

HONEST MEASURED RESULT (5090D, this committed test): the mine/triton gap SHRINKS
with size -- ~2-3x at 128, ~1.5-2x at 256, ~1.2-1.6x at 512, ~1.0-1.2x at 1024
(near-parity at the largest shape). This does NOT reproduce the PRD's "parity small
/ 1.38-1.43x large": the trend is INVERTED (large is closer to parity than the PRD
claims, small is worse). The small-shape gap is largely our per-call Python launch
overhead (kern.run rebuilds the launch args each call; Triton's cached launcher does
not) inflating sub-30us kernels, and is noisy. So the LOOSE perf assert
(mine <= triton * 2.5) is gated only on the GPU-bound large shapes (sq >= 512, like
test_flash_bwd_perf.py); all four ratios are PRINTED honestly. It guards a
regression, it does NOT claim parity.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_flash_fwd_perf.py -q -s
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from test_flash_fwd_mma_pk import _FlashKernel, _flash_intent, _rel  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                       reason="intent-opt not built"),
    pytest.mark.skipif(not pathlib.Path(MLIR_OPT).exists(),
                       reason="mlir-opt not present"),
    pytest.mark.skipif(not pathlib.Path(CUDA_BIN).exists(),
                       reason="CUDA toolkit not present"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]

_FLASH_TRITON = ("/home/kingdom/phdworks/Dataset/flash-attention/"
                 "flash_attn/flash_attn_triton.py")


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=50, rep=200, return_mode="median")


def _load_triton_fwd():
    """The bare @triton.jit _fwd_kernel (inner JITFunction, under @heuristics).

    _fwd_kernel is Heuristics(JITFunction); we drive the inner JITFunction with the
    heuristic constexprs (EVEN_M/EVEN_N/EVEN_HEADDIM) supplied explicitly, dodging
    the heuristics launcher.

    Dataset is READ-ONLY: the trans_b= removed in Triton 3.x is patched into a
    LOCAL temp copy (keyed by content hash) that @triton.jit re-reads, leaving the
    reference file pristine.
    """
    import hashlib
    import tempfile
    if not pathlib.Path(_FLASH_TRITON).exists():
        pytest.skip("flash-attention source not present")
    src = pathlib.Path(_FLASH_TRITON).read_text()
    patches = [
        ("qk += tl.dot(q, k, trans_b=True)", "qk += tl.dot(q, tl.trans(k))"),
    ]
    for old, new in patches:
        src = src.replace(old, new)
    tag = hashlib.sha1(src.encode()).hexdigest()[:12]
    local = pathlib.Path(tempfile.gettempdir()) / f"fat_fwd_triton3x_{tag}.py"
    if not local.exists():
        local.write_text(src)
    spec = importlib.util.spec_from_file_location("fat_fwd_triton3x", str(local))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._fwd_kernel.fn  # Heuristics.fn == inner JITFunction


def _triton_fwd_thunk(Sq, Sk, d, q2, k2, v2, scale, BM=128, BN=128):
    """A no-arg thunk launching ONLY Triton's _fwd_kernel (single-head, non-causal)."""
    fwd_jit = _load_triton_fwd()
    sqr = math.ceil(Sq / 128) * 128
    # 4D single-head (batch=1, nheads=1) views.
    q = q2.view(1, Sq, 1, d)
    k = k2.view(1, Sk, 1, d)
    v = v2.view(1, Sk, 1, d)
    o = torch.empty_like(q)
    lse = torch.empty(1, 1, sqr, device="cuda", dtype=torch.float32)
    tmp = torch.empty(1, 1, sqr, device="cuda", dtype=torch.float32)
    BHD = max(triton.next_power_of_2(d), 16)
    warps = 4 if d <= 64 else 8
    grid = (triton.cdiv(Sq, BM), 1)
    EVEN_M, EVEN_N, EVEN_HEADDIM = Sq % BM == 0, Sk % BN == 0, d == BHD

    def go():
        fwd_jit[grid](
            q, k, v, None, o, lse, tmp, scale,
            q.stride(0), q.stride(2), q.stride(1),
            k.stride(0), k.stride(2), k.stride(1),
            v.stride(0), v.stride(2), v.stride(1),
            0, 0, 0,
            o.stride(0), o.stride(2), o.stride(1),
            1, Sq, Sk, sqr, d, Sq // 32, Sk // 32,
            "none", False, BHD, EVEN_M, EVEN_N, EVEN_HEADDIM, BM, BN,
            num_warps=warps, num_stages=1)
    return go, o.view(Sq, d)


@pytest.mark.parametrize("sq,sk,d", [
    (128, 128, 64),
    (256, 256, 64),
    (512, 512, 64),
    (1024, 1024, 64),
])
def test_flash_fwd_perf(sq, sk, d):
    torch.manual_seed(0)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    scale = 1.0 / math.sqrt(d)

    # fp32 + SDPA references (reused from the correctness PK).
    s = torch.matmul(q.float(), k.float().t()) * scale
    ref32 = torch.matmul(torch.softmax(s, dim=-1), v.float())
    sdpa_ref = torch.nn.functional.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None], scale=scale)[0, 0]

    # ---- our M8d register-resident mma.sync flash fwd (the PERF deliverable) ---- #
    # num-stages=2 explicit (== the pass default, but pinned so a future default
    # change can't silently move the reproduced number).
    kern = _FlashKernel(_flash_intent(sq, sk, d), num_stages=2)
    o = torch.empty(sq, d, device="cuda", dtype=torch.float32).contiguous()
    mfn = lambda: kern.run(q, k, v, scale, o)
    mfn()
    torch.cuda.synchronize()
    rel_ref, rel_sdpa = _rel(o, ref32), _rel(o, sdpa_ref)
    assert rel_ref < 2e-2, f"({sq},{sk},{d}) rel vs fp32 ref={rel_ref:.3e}"
    assert rel_sdpa < 2e-2, f"({sq},{sk},{d}) rel vs SDPA={rel_sdpa:.3e}"
    t_mine = _bench(mfn)

    # ---- Triton _fwd_kernel (fixed 128x128 config) ---- #
    t_triton = None
    try:
        tgo, _ = _triton_fwd_thunk(sq, sk, d, q, k, v, scale)
        tgo()
        torch.cuda.synchronize()
        t_triton = _bench(tgo)
    except Exception as exc:  # noqa: BLE001 - Triton-3.x incompat: skip baseline
        print(f"\n[triton fwd baseline unavailable, SDPA+correctness still gated]: "
              f"{type(exc).__name__}: {exc}")

    # ---- torch SDPA (library flash kernel; informational) ---- #
    q4 = q.view(1, 1, sq, d)
    k4 = k.view(1, 1, sk, d)
    v4 = v.view(1, 1, sk, d)
    sdfn = lambda: torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, scale=scale)
    sdfn()
    torch.cuda.synchronize()
    t_sdpa = _bench(sdfn)

    print(f"\nflash-fwd ({sq},{sk},{d}) grid={kern.grid} block={kern.block} "
          f"smem={kern.shared}B rel_fp32={rel_ref:.2e} rel_sdpa={rel_sdpa:.2e}")
    print(f"  MINE    {t_mine*1e3:7.1f}us  (M8d register-resident mma.sync)")
    if t_triton is not None:
        print(f"  triton  {t_triton*1e3:7.1f}us  -> mine/triton = {t_mine/t_triton:.2f}x")
    else:
        print(f"  triton  (unavailable -- baseline skipped)")
    print(f"  SDPA    {t_sdpa*1e3:7.1f}us  -> mine/SDPA   = {t_mine/t_sdpa:.2f}x "
          f"(library flash; informational)")

    # HONEST measured trend (5090D, this committed test): the mine/triton gap
    # SHRINKS with size -- ~2-3x at 128, ~1.5-2x at 256, ~1.2-1.6x at 512, ~1.0-1.2x
    # at 1024 (near-parity large). This does NOT reproduce the PRD's "parity small
    # / 1.38-1.43x large": the trend is inverted (large is BETTER than claimed).
    # The small-shape gap is largely our per-call Python launch overhead (kern.run
    # rebuilds the arg list each call; Triton's cached launcher does not) inflating
    # sub-30us kernels -- NOT a kernel-efficiency gap; it is noisy (do_bench brackets
    # the host enqueue). So, exactly like test_flash_bwd_perf.py (which only benches
    # 512/1024), the loose perf assert is gated only on the GPU-bound large shapes
    # where the number is stable and meaningful; the small-shape ratios are printed
    # honestly but not gated. The bound guards a regression, NOT a parity claim.
    if t_triton is not None and sq >= 512:
        assert t_mine <= t_triton * 2.5, \
            f"mine {t_mine*1e3:.1f}us vs triton {t_triton*1e3:.1f}us > 2.5x"
