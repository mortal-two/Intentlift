"""M9 flash-bwd PERF: the ATOMIC (key-block-parallel) mode vs serial, the real
Triton ``_bwd_kernel``, and torch SDPA backward, kernel-only ``do_bench``.

M9a measured the SERIAL DQ mode (dq-mode=serial, grid=(1,1,1), ONE block per
(b,h) owning the whole head via an outer scf.for over key tiles) at ~617us on the
5090D -- single-SM, a deliberately un-parallel correctness baseline. M9b added the
ATOMIC mode (dq-mode=atomic): grid.x = cdiv(Sk,BN) so one PROGRAM owns each key
block and accumulates its dQ contribution into pre-zeroed global DQ via
memref.atomic_rmw addf (-> red.global.add.f32); dK/dV are partitioned by key block
(no atomics). This parallelizes over key blocks -- 8 programs for Sk=512, 16 for
Sk=1024 -- the M9 PERF win.

This test measures the atomic mode against:
  (a) the bare @triton.jit ``_bwd_kernel`` (flash_attn_triton.py), called with a
      FIXED config (BM=128,BN=64,SEQUENCE_PARALLEL=True,num_warps=8) to dodge the
      autotune launcher; LSE+delta precomputed OUTSIDE the timed region (we time
      ONLY the main bwd loop -- our atomic mode likewise does NOT run a preprocess
      pass), so it is an apples-to-apples kernel comparison.
  (b) torch SDPA backward (the library flash kernel; includes its own forward
      recompute, a different op boundary -- informational).
  (c) the serial mode (the 617us M9a baseline).

HONEST RESIDUAL (5090D): atomic lands ~1.4-1.6x off the same-tile (64x64,8warp)
Triton config and ~1.8-1.9x off Triton's best (128x64) -- the inherited
wmma.load gap (M7) + the 5-matmul serial chain (barrier between each) + 4 warps
vs Triton's 8. Single-head grid is 8-16 blocks for ~170 SMs (Triton's BN=64 grid
is the SAME 8 blocks), so the residual is per-block efficiency, not grid size.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_flash_bwd_perf.py -q -s
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from test_flash_bwd_dualmode_pk import _bwd_intent, _FlashBwdKernel  # noqa: E402

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


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=50, rep=200, return_mode="median")


def _load_triton_bwd():
    """The bare @triton.jit _bwd_kernel (unwrap autotune+heuristics) + preprocess.

    flash_attn_triton.py's _bwd_kernel is Autotuner(Heuristics(JITFunction)); we
    drive the inner JITFunction with an explicit fixed config, dodging the
    autotune launcher.

    The Dataset source is READ-ONLY and is NOT modified. Triton 3.x removed
    tl.dot(..., trans_a=/trans_b=); since @triton.jit re-reads source from the
    file path at compile time, we write the rewritten copy to a LOCAL temp file
    (NOT Dataset) and load the kernel from there, leaving the reference pristine.
    """
    import hashlib
    import tempfile
    if not pathlib.Path(_FLASH_TRITON).exists():
        pytest.skip("flash-attention source not present")
    src = pathlib.Path(_FLASH_TRITON).read_text()
    patches = [
        ("qk += tl.dot(q, k, trans_b=True)", "qk += tl.dot(q, tl.trans(k))"),
        ("qk = tl.dot(q, k, trans_b=True)", "qk = tl.dot(q, tl.trans(k))"),
        ("dv += tl.dot(p.to(do.dtype), do, trans_a=True)",
         "dv += tl.dot(tl.trans(p.to(do.dtype)), do)"),
        ("dp = tl.dot(do, v, trans_b=True)", "dp = tl.dot(do, tl.trans(v))"),
        ("dk += tl.dot(ds, q, trans_a=True)", "dk += tl.dot(tl.trans(ds), q)"),
    ]
    for old, new in patches:
        src = src.replace(old, new)
    # Stable local path (keyed by content) so Triton's source re-read hits the
    # patched copy, not the pristine Dataset file.
    tag = hashlib.sha1(src.encode()).hexdigest()[:12]
    local = pathlib.Path(tempfile.gettempdir()) / f"fat_bwd_triton3x_{tag}.py"
    if not local.exists():
        local.write_text(src)
    spec = importlib.util.spec_from_file_location("fat_bwd_triton3x", str(local))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._bwd_kernel.fn.fn, mod._bwd_preprocess_do_o_dot


def _triton_bwd_thunk(Sq, Sk, d, q2, k2, v2, do2, lse_v, scale, BM=128, BN=64,
                      warps=8):
    """A no-arg thunk launching ONLY Triton's _bwd_kernel (delta precomputed)."""
    bwd_jit, pre_jit = _load_triton_bwd()
    sqr = math.ceil(Sq / 128) * 128
    # 4D single-head (batch=1, nheads=1) views.
    q = q2.view(1, Sq, 1, d)
    k = k2.view(1, Sk, 1, d)
    v = v2.view(1, Sk, 1, d)
    do = do2.view(1, Sq, 1, d)
    with torch.no_grad():
        o_det = (torch.softmax((q2.float() @ k2.float().t()) * scale, dim=-1)
                 @ v2.float()).to(torch.float16).view(1, Sq, 1, d).contiguous()
    lse = torch.zeros(1, 1, sqr, device="cuda", dtype=torch.float32)
    lse[0, 0, :Sq] = lse_v
    delta = torch.empty(1, 1, sqr, device="cuda", dtype=torch.float32)
    dq = torch.zeros(1, Sq, 1, d, device="cuda", dtype=torch.float32)
    dk = torch.empty(1, Sk, 1, d, device="cuda", dtype=torch.float32)
    dv = torch.empty(1, Sk, 1, d, device="cuda", dtype=torch.float32)
    BHD = max(triton.next_power_of_2(d), 16)
    pre_jit[(triton.cdiv(Sq, 128), 1)](
        o_det, do, delta, o_det.stride(0), o_det.stride(2), o_det.stride(1),
        do.stride(0), do.stride(2), do.stride(1), 1, Sq, sqr, d,
        BLOCK_M=128, BLOCK_HEADDIM=BHD)
    grid = (triton.cdiv(Sk, BN), 1)
    EVEN_M, EVEN_N, EVEN_HEADDIM = Sq % BM == 0, Sk % BN == 0, d == BHD

    def go():
        # No dq.zero_() in the timed region (parity with the atomic-mode thunk):
        # the cross-program atomic adds sum across do_bench reps -- values drift
        # but kernel timing is unaffected. Caller zeroes once for correctness.
        bwd_jit[grid](
            q, k, v, dq, do, dq, dk, dv, lse, delta, scale,
            q.stride(0), q.stride(2), q.stride(1),
            k.stride(0), k.stride(2), k.stride(1),
            v.stride(0), v.stride(2), v.stride(1), 0, 0, 0,
            do.stride(0), do.stride(2), do.stride(1),
            dq.stride(0), dq.stride(2), dq.stride(1),
            dk.stride(0), dk.stride(2), dk.stride(1),
            dv.stride(0), dv.stride(2), dv.stride(1),
            1, Sq, Sk, sqr, d, Sq // 32, Sk // 32,
            "none", False, BHD, True, EVEN_M, EVEN_N, EVEN_HEADDIM, BM, BN,
            num_warps=warps, num_stages=1)
    return go, (dq, dk, dv)


@pytest.mark.parametrize("sq,sk,d", [(512, 512, 64), (1024, 1024, 64)])
def test_flash_bwd_atomic_perf(sq, sk, d):
    torch.manual_seed(0)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    do = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    scale = 1.0 / math.sqrt(d)

    # fp32 autograd reference.
    qf = q.float().requires_grad_(True)
    kf = k.float().requires_grad_(True)
    vf = v.float().requires_grad_(True)
    o = torch.softmax((qf @ kf.t()) * scale, dim=-1) @ vf
    o.backward(do.float())
    with torch.no_grad():
        sd = (q.float() @ k.float().t()) * scale
        lse = torch.logsumexp(sd, dim=-1, keepdim=True).contiguous()
        lse_v = lse.squeeze(1)
        o_det = torch.softmax(sd, dim=-1) @ v.float()
        delta = (o_det * do.float()).sum(-1, keepdim=True).contiguous()

    intent = _bwd_intent(sq, sk, d)
    dq = torch.zeros(sq, d, device="cuda", dtype=torch.float32).contiguous()
    dk = torch.empty(sk, d, device="cuda", dtype=torch.float32).contiguous()
    dv = torch.empty(sk, d, device="cuda", dtype=torch.float32).contiguous()

    # ---- atomic mode (the M9 PERF deliverable) ---- #
    atomic = _FlashBwdKernel(intent, "atomic")
    assert atomic.grid[0] == sk // 64, "atomic grid.x == cdiv(Sk,BN)"
    dq.zero_()
    afn = lambda: atomic.run(q, k, v, do, lse, delta, scale, dq, dk, dv)
    afn()
    torch.cuda.synchronize()
    rel_dq, rel_dk, rel_dv = _rel(dq, qf.grad), _rel(dk, kf.grad), _rel(dv, vf.grad)
    assert rel_dv < 2e-2 and rel_dk < 2e-2 and rel_dq < 2e-2, \
        f"atomic rel dQ={rel_dq:.2e} dK={rel_dk:.2e} dV={rel_dv:.2e}"
    t_atomic = _bench(afn)

    # ---- serial mode (the M9a 617us baseline) ---- #
    serial = _FlashBwdKernel(intent, "serial")
    dq.zero_()
    sfn = lambda: serial.run(q, k, v, do, lse, delta, scale, dq, dk, dv)
    sfn()
    torch.cuda.synchronize()
    t_serial = _bench(sfn)

    # ---- Triton _bwd_kernel (best fixed config) ---- #
    tgo, _ = _triton_bwd_thunk(sq, sk, d, q, k, v, do, lse_v, scale)
    tgo()
    torch.cuda.synchronize()
    t_triton = _bench(tgo)

    # ---- torch SDPA backward (informational; includes recompute) ---- #
    q4 = q.view(1, 1, sq, d).clone().requires_grad_(True)
    k4 = k.view(1, 1, sk, d).clone().requires_grad_(True)
    v4 = v.view(1, 1, sk, d).clone().requires_grad_(True)
    do4 = do.view(1, 1, sq, d)
    osd = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, scale=scale)
    sdfn = lambda: torch.autograd.grad(osd, [q4, k4, v4], do4, retain_graph=True)
    sdfn()
    torch.cuda.synchronize()
    t_sdpa = _bench(sdfn)

    print(f"\nflash-bwd ({sq},{sk},{d}) atomic grid={atomic.grid} "
          f"smem={atomic.shared}B rel dQ={rel_dq:.2e} dK={rel_dk:.2e} dV={rel_dv:.2e}")
    print(f"  serial  {t_serial*1e3:7.1f}us  (M9a single-SM baseline)")
    print(f"  ATOMIC  {t_atomic*1e3:7.1f}us  -> {t_serial/t_atomic:5.1f}x over serial")
    print(f"  triton  {t_triton*1e3:7.1f}us  -> atomic/triton = {t_atomic/t_triton:.2f}x")
    print(f"  SDPA    {t_sdpa*1e3:7.1f}us  -> atomic/SDPA   = {t_atomic/t_sdpa:.2f}x "
          f"(incl. fwd recompute; informational)")

    # The atomic key-block parallelism must be a large win over the single-SM
    # serial baseline (measured 6.7x@512, 14.5x@1024).
    assert t_atomic < t_serial * 0.5, \
        f"atomic {t_atomic*1e3:.1f}us not >> serial {t_serial*1e3:.1f}us"
    # Honest residual vs the real Triton bwd kernel (measured ~1.8-1.9x); loose
    # bound guards against a regression, not a parity claim.
    assert t_atomic < t_triton * 3.0, \
        f"atomic {t_atomic*1e3:.1f}us vs triton {t_triton*1e3:.1f}us > 3x"
