"""CUDA-graph steady-state PK: our C3-C8 cluster cubins vs the REAL flag_gems +
torch kernels — all three in-process on the same path.

The sharp question this answers: *under CUDA-graph capture (launch overhead
removed for ALL sides), can our generated cluster kernels BEAT the real
flag_gems / torch kernels?* The answer is honest and roofline-grounded — win
where physics allows (memory-bound movers + fusion), lose where the sm_120a
compute ceiling or the naive-untuned single-block gap is real.

flag_gems is the REAL Triton target. It is now timed IN-PROCESS on the same
``do_bench_cudagraph`` path as ours + torch, thanks to
``intentir.backend.flaggems_shim`` (routes flag_gems' cached libentry launcher
through its constexpr-inclusive path so triton 3.6 no longer drops the constexpr
placeholder slot — see the shim module / research/r2-flaggems-fix.md). The old
FG_BENCH subprocess escape hatch is retired.

HARNESS (fair by construction):
  * Our kernel: the SAME lower->cubin path the PKs use (RowReductionKernel with
    the cluster pass), launched via the CUDA Driver API on the *current* stream
    (read at call time, so it is captured on the graph's capture stream).
  * References: the REAL ``flag_gems.<op>`` triton kernel AND ``torch.<op>``,
    BOTH in-process.
  * All sides timed with ``triton.testing.do_bench_cudagraph`` — it warms up on
    a side stream, captures N unrolled calls into ONE cuda graph, and times
    ``g.replay()``. This is the explicit CUDA-graph steady-state kernel-time PK:
    per-call host launch overhead is amortized to ~0 for every contender.
  * When an op cannot be graph-captured (dynamic output extent, in-place mutate,
    host sync inside the ref) we fall back to plain ``do_bench`` and TAG it.

flag_gems dtype note: flag_gems follows torch's default dtype for these ops, so
at the f32 shapes here it moves the SAME bytes as torch/ours (no free fp16/bf16
halving). Where a real deployment would run fp16/bf16 flag_gems (norm/softmax/
attention epilogues) it would move ~2x fewer bytes — noted per-op where it
applies.

REALISTIC f32 shapes (our kernels are f32 — a fp16/bf16 flag_gems path would be
~2x fewer bytes and is noted where it applies).

Run (quiet GPU, sequential):
  export PATH=/usr/local/cuda-12.8/bin:$PATH
  PYTHONPATH=python python3 python/tests/perf/bench_clusters.py
"""

from __future__ import annotations

import ctypes
import gc
import json
import traceback
from pathlib import Path

import torch
import triton

from intentir.backend.launch import I64Arg, PtrArg, _marshal
from intentir.benchmarks.row_reduction import ConvMMAKernel, RowReductionKernel

# NOTE (perf audit): the GB/s / %HBM columns are INDICATIVE only — nbytes counts
# LOGICAL bytes (a tiled/broadcast input is counted per-read though it is
# L2-resident across graph replays), and 1500 is a sustained figure below the
# ~1792 GB/s GDDR7 spec, so %HBM can exceed 100% and must not be read as a real
# roofline fraction. The load-bearing metric is the our/torch RATIO (wall-clock,
# same shape/f32/work, both CUDA-graph-captured).
PEAK_GBPS = 1500.0     # sustained HBM on the 5090D (spec ~1792; see row_reduction.py)
PEAK_TFLOPS_F32 = 105.0  # 5090D non-tensor-core f32 peak (~105 TFLOP/s)


def _flag_gems():
    """Import flag_gems and apply the triton-3.6 launcher shim (idempotent) so
    its cached libentry dispatch works in-process under CUDA-graph capture."""
    try:
        import flag_gems
        from intentir.backend.flaggems_shim import apply_flaggems_launch_shim
        apply_flaggems_launch_shim()
        return flag_gems
    except Exception:
        return None


FG = _flag_gems()


def fg_time(fn, want_graph=True, verify=None):
    """Time a flag_gems callable IN-PROCESS on the same do_bench_cudagraph path
    as ours/torch. Returns median us, or None if flag_gems is absent / the op
    is unavailable / it raises. ``verify(result) -> bool`` asserts correctness
    at the benched shape (nothing a no-op)."""
    if FG is None or fn is None:
        return None
    try:
        r = fn()
        torch.cuda.synchronize()
        if verify is not None and not verify(r):
            print("    [fg] correctness MISMATCH — dropping flag_gems number")
            return None
    except Exception as e:  # noqa: BLE001
        print(f"    [fg] SKIP: {type(e).__name__}: {str(e)[:80]}")
        return None
    us, _ = bench(fn, want_graph=want_graph)
    return us


# --------------------------------------------------------------------------- #
# Launch our compiled kernel on the CURRENT stream (read at call time so the
# graph capture on do_bench_cudagraph's capture stream records the kernel).
# --------------------------------------------------------------------------- #
def our_launcher(k: RowReductionKernel, tensors):
    args = [PtrArg(tensors[i].data_ptr()) if kind == "ptr" else I64Arg(i)
            for kind, i in k.plan]
    keep, argp = _marshal(args)
    addr = int(ctypes.addressof(argp))
    cuda, func = k._cuda, k.func
    g0, g1, g2 = k.grid
    b0, b1, b2 = k.block
    shared = k.shared

    def go(_keep=keep, _argp=argp):
        stream = int(torch.cuda.current_stream().cuda_stream)
        cuda.cuLaunchKernel(func, g0, g1, g2, b0, b1, b2, shared, stream, addr, 0)

    return go


def bench(fn, want_graph=True):
    """(us, graph_captured?). Try CUDA-graph steady-state; fall back to do_bench."""
    if want_graph:
        try:
            ms = triton.testing.do_bench_cudagraph(fn, rep=100, return_mode="median")
            return ms * 1e3, True
        except Exception:
            torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=50, rep=200, return_mode="median")
    return ms * 1e3, False


def gbps(nbytes, us):
    return nbytes / (us * 1e-6) / 1e9 if us else 0.0


def gflops(flop, us):
    return flop / (us * 1e-6) / 1e9 if us else 0.0


ROWS = []


def record(cluster, op, shape, our_us, refs, nbytes=None, flop=None,
           cls="mem", note="", graph_ours=True, graph_ref=True):
    """refs = {label: us_or_None}. best-ref = min of the present ref times."""
    present = {k: v for k, v in refs.items() if v is not None}
    best_lbl, best_us = (min(present.items(), key=lambda kv: kv[1])
                         if present else (None, None))
    ratio = (our_us / best_us) if best_us else None
    if cls == "compute" and flop:
        our_perf = f"{gflops(flop, our_us):.0f} GF/s"
        peak = f"{100*gflops(flop, our_us)/(PEAK_TFLOPS_F32*1e3):.1f}% f32pk"
    else:
        our_perf = f"{gbps(nbytes, our_us):.0f} GB/s" if nbytes else "-"
        peak = (f"{100*gbps(nbytes, our_us)/PEAK_GBPS:.0f}% HBM"
                if nbytes else "-")
    if ratio is None:
        verdict = "n/a"
    elif ratio <= 1.15:
        verdict = "WIN" if ratio < 0.9 else "parity"
    elif ratio <= 2.5:
        verdict = "lose"
    else:
        verdict = "LOSE"
    row = dict(cluster=cluster, op=op, shape=shape, our_us=round(our_us, 2),
               flaggems_us=(round(refs.get("flag_gems"), 2)
                            if refs.get("flag_gems") else None),
               torch_us=(round(refs.get("torch"), 2)
                         if refs.get("torch") else None),
               best_ref=best_lbl, ratio=(round(ratio, 2) if ratio else None),
               our_perf=our_perf, peak=peak, cls=cls, verdict=verdict,
               graph=f"ours={'Y' if graph_ours else 'N'}/ref={'Y' if graph_ref else 'N'}",
               note=note)
    ROWS.append(row)
    print(f"  {op:22s} our={our_us:8.2f}us  "
          f"fg={_f(refs.get('flag_gems'))} torch={_f(refs.get('torch'))}  "
          f"ratio={('%.2f' % ratio) if ratio else '  -  '} "
          f"{our_perf:>10s} {peak:>9s}  [{verdict}] {note}")


def _f(v):
    return f"{v:8.2f}us" if v is not None else "   n/a   "


def _mk(pass_name, mlir, T, M=1, N=1):
    return RowReductionKernel("k", {"M": M, "N": N}, T, pass_name=pass_name,
                              mlir_text=mlir)


def _fg_time(fn, want_graph=True):
    """Time a torch ref: warm once, then graph-bench (fallback plain do_bench)."""
    try:
        fn()
        torch.cuda.synchronize()
    except Exception:
        return None, True
    us, g = bench(fn, want_graph=want_graph)
    return us, g


def _cleanup():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# ======================================================================== #
# C3  gather / scatter                                                      #
# ======================================================================== #
def c3_gather():
    print("\n[C3] gather / scatter (data-dependent addressing)")
    # gather along axis=1: [M,N], idx [M,N]
    M, N, T = 8192, 4096, 256
    R = f"tensor<{M}x{N}xf32>"
    mlir = f"""
func.func @g(%d: {R}, %i: tensor<{M}x{N}xi64>) -> {R} {{
  %g = intent.gather %d, %i {{axis = 1 : i64, mode = "along_axis"}} : ({R}, tensor<{M}x{N}xi64>) -> {R}
  return %g : {R}
}}"""
    torch.manual_seed(0)
    data = torch.randn(M, N, device="cuda")
    idx = torch.randint(0, N, (M, N), device="cuda", dtype=torch.int64)
    out = torch.empty(M, N, device="cuda")
    k = _mk("convert-intent-gather-to-gpu", mlir, T, M, N)
    go = our_launcher(k, [data, idx, out])
    go(); torch.cuda.synchronize()
    assert torch.equal(out, torch.gather(data, 1, idx))
    our, go_g = bench(go)
    refs = {}
    tt, _ = _fg_time(lambda: torch.gather(data, 1, idx)); refs["torch"] = tt
    ref_g = torch.gather(data, 1, idx)
    refs["flag_gems"] = fg_time(lambda: FG.gather(data, 1, idx),
                                verify=lambda r: torch.equal(r, ref_g))
    nb = M * N * (8 + 4 + 4)  # idx read + data gathered + out
    record("C3", "gather ax1", f"[{M},{N}]", our, refs, nbytes=nb,
           note="idx i64 read-heavy", graph_ours=go_g)
    del data, idx, out; _cleanup()

    # embedding (row gather) [V,D], idx[Nidx]
    V, D, Ni, T = 8192, 4096, 8192, 256
    W = f"tensor<{V}x{D}xf32>"; O = f"tensor<{Ni}x{D}xf32>"
    mlir = f"""
func.func @e(%w: {W}, %i: tensor<{Ni}xi64>) -> {O} {{
  %g = intent.gather %w, %i {{axis = 0 : i64, mode = "take"}} : ({W}, tensor<{Ni}xi64>) -> {O}
  return %g : {O}
}}"""
    weight = torch.randn(V, D, device="cuda")
    idx = torch.randint(0, V, (Ni,), device="cuda", dtype=torch.int64)
    out = torch.empty(Ni, D, device="cuda")
    k = _mk("convert-intent-gather-to-gpu", mlir, T, Ni, D)
    go = our_launcher(k, [weight, idx, out])
    go(); torch.cuda.synchronize()
    assert torch.equal(out, torch.nn.functional.embedding(idx, weight))
    our, go_g = bench(go)
    refs = {}
    tt, _ = _fg_time(lambda: torch.nn.functional.embedding(idx, weight)); refs["torch"] = tt
    ref_e = torch.nn.functional.embedding(idx, weight)
    refs["flag_gems"] = fg_time(lambda: FG.embedding(weight, idx),
                                verify=lambda r: torch.equal(r, ref_e))
    nb = Ni * D * (4 + 4)  # read row + write row
    record("C3", "embedding", f"[{V},{D}] idx {Ni}", our, refs, nbytes=nb,
           graph_ours=go_g)
    del weight, idx, out; _cleanup()

    # scatter_add axis=1 (atomic) [M,N], idx/src [M,K]
    M, N, K, T = 8192, 4096, 4096, 256
    Dt = f"tensor<{M}x{N}xf32>"; St = f"tensor<{M}x{K}xf32>"
    mlir = f"""
func.func @s(%d: {Dt}, %i: tensor<{M}x{K}xi64>, %s: {St}) -> {Dt} {{
  %o = intent.scatter %d, %i, %s {{axis = 1 : i64, mode = "along_axis", reduce = "add"}} : ({Dt}, tensor<{M}x{K}xi64>, {St}) -> {Dt}
  return %o : {Dt}
}}"""
    base = torch.randn(M, N, device="cuda")
    idx = torch.randint(0, N, (M, K), device="cuda", dtype=torch.int64)
    src = torch.randn(M, K, device="cuda")
    data = base.clone()
    k = _mk("convert-intent-gather-to-gpu", mlir, T, M, K)
    go = our_launcher(k, [data, idx, src])
    # NOTE: scatter_add mutates data in place -> NOT idempotent under graph
    # replay (accumulates each replay). Time it, but it double-counts adds; use
    # plain do_bench with a fresh clone is not possible per-call, so we time the
    # atomic-traffic kernel and TAG it. Correctness is gated by the PK, not here.
    our, go_g = bench(go, want_graph=False)
    refs = {}
    d2 = base.clone()
    tt, _ = _fg_time(lambda: d2.scatter_add_(1, idx, src), want_graph=False); refs["torch"] = tt
    ref_sa = base.scatter_add(1, idx, src)
    refs["flag_gems"] = fg_time(
        lambda: FG.scatter_add_(base.clone(), 1, idx, src), want_graph=False,
        verify=lambda r: torch.allclose(r, ref_sa, atol=1e-3))
    nb = M * K * (8 + 4 + 4)  # idx + src + atomic rmw
    record("C3", "scatter_add ax1", f"[{M},{N}] k{K}", our, refs, nbytes=nb,
           note="atomic rmw; in-place -> do_bench (no graph)", graph_ours=False)
    del base, idx, src, data, d2; _cleanup()


# ======================================================================== #
# C4  layout (concat / pad / tile / flip / triu)                           #
# ======================================================================== #
def c4_layout():
    print("\n[C4] layout (structural data movement)")
    M, N, T = 4096, 4096, 256

    def one(op, mlir, tensors, verify, ref_torch, fg_call, nb, shape=None,
            note="", fg_verify=None):
        try:
            k = _mk("convert-intent-layout-to-gpu", mlir, T, tensors[-1].shape[0],
                    tensors[-1].shape[1])
            go = our_launcher(k, tensors)
            go(); torch.cuda.synchronize()
            assert verify(), f"{op} correctness failed"
        except Exception as e:  # noqa: BLE001
            print(f"  {op:22s} SKIP: {str(e)[:90]}")
            return
        our, gg = bench(go)
        refs = {}
        if ref_torch is not None:
            tt, _ = _fg_time(ref_torch); refs["torch"] = tt
        if fg_call is not None:
            refs["flag_gems"] = fg_time(fg_call, verify=fg_verify)
        record("C4", op, shape or f"[{M},{N}]", our, refs, nbytes=nb, note=note,
               graph_ours=gg)

    A = f"tensor<{M}x{N}xf32>"
    torch.manual_seed(0)
    a = torch.randn(M, N, device="cuda")
    b = torch.randn(M, N, device="cuda")

    # concat axis=1 -> [M,2N]
    O = f"tensor<{M}x{2*N}xf32>"
    mlir = f'func.func @c(%a: {A}, %b: {A}) -> {O} {{ %c = intent.concat %a, %b {{axis = 1 : i64}} : ({A}, {A}) -> {O}\n return %c : {O} }}'
    out = torch.empty(M, 2 * N, device="cuda")
    one("concat ax1", mlir, [a, b, out],
        lambda: torch.equal(out, torch.cat([a, b], 1)),
        lambda: torch.cat([a, b], 1),
        (lambda: FG.cat([a, b], dim=1)) if FG else None,
        2 * M * N * 4 * 2,
        fg_verify=lambda r: torch.equal(r, torch.cat([a, b], 1)))
    del out; _cleanup()

    # pad lo/hi 2: output width must be divisible by T=256, so pad 4092 -> 4096.
    Mp = M - 4
    Ap = f"tensor<{Mp}x{Mp}xf32>"; Op = f"tensor<{M}x{M}xf32>"
    ap = torch.randn(Mp, Mp, device="cuda")
    mlir = f'func.func @p(%a: {Ap}) -> {Op} {{ %p = intent.pad %a {{lo = [2, 2], hi = [2, 2], value = 0.0 : f64}} : ({Ap}) -> {Op}\n return %p : {Op} }}'
    out = torch.empty(M, M, device="cuda")
    one("pad", mlir, [ap, out],
        lambda: torch.equal(out, torch.nn.functional.pad(ap, [2, 2, 2, 2])),
        lambda: torch.nn.functional.pad(ap, [2, 2, 2, 2]),
        None,
        M * M * 4 * 2, shape=f"[{Mp},{Mp}]->[{M},{M}]")
    del out, ap; _cleanup()

    # tile reps [2,2] -> [2M,2N]
    Ot = f"tensor<{2*M}x{2*N}xf32>"
    mlir = f'func.func @t(%a: {A}) -> {Ot} {{ %t = intent.tile %a {{reps = [2, 2]}} : ({A}) -> {Ot}\n return %t : {Ot} }}'
    out = torch.empty(2 * M, 2 * N, device="cuda")
    one("tile [2,2]", mlir, [a, out],
        lambda: torch.equal(out, a.repeat(2, 2)),
        lambda: a.repeat(2, 2),
        (lambda: FG.tile(a, [2, 2])) if FG else None,
        4 * M * N * 4 * 2,
        fg_verify=lambda r: torch.equal(r, a.repeat(2, 2)))
    del out; _cleanup()

    # flip cols
    mlir = f'func.func @f(%a: {A}) -> {A} {{ %f = intent.flip %a {{axes = [1]}} : ({A}) -> {A}\n return %f : {A} }}'
    out = torch.empty(M, N, device="cuda")
    one("flip ax1", mlir, [a, out],
        lambda: torch.equal(out, torch.flip(a, [1])),
        lambda: torch.flip(a, [1]),
        (lambda: FG.flip(a, [1])) if FG else None,
        M * N * 4 * 2,
        fg_verify=lambda r: torch.equal(r, torch.flip(a, [1])))
    del out; _cleanup()

    # triu k=0
    mlir = f'func.func @u(%a: {A}) -> {A} {{ %u = intent.triu %a {{k = 0 : i64}} : ({A}) -> {A}\n return %u : {A} }}'
    out = torch.empty(M, N, device="cuda")
    one("triu k0", mlir, [a, out],
        lambda: torch.equal(out, torch.triu(a)),
        lambda: torch.triu(a),
        (lambda: FG.triu(a, 0)) if FG else None,
        M * N * 4 * 2,
        fg_verify=lambda r: torch.equal(r, torch.triu(a)))
    del out, a, b; _cleanup()


# ======================================================================== #
# C5  rng (rand / dropout)                                                  #
# ======================================================================== #
def c5_rng():
    print("\n[C5] rng (seeded philox)")
    M, N, T = 8192, 8192, 256
    O = f"tensor<{M}x{N}xf32>"

    # rand (write-only)
    mlir = f'func.func @r(%d: {O}) -> {O} {{ %r = intent.rand {{seed = 12345 : i64}} : {O}\n return %r : {O} }}'
    din = torch.empty(M, N, device="cuda")
    out = torch.empty(M, N, device="cuda")
    k = _mk("convert-intent-rng-to-gpu", mlir, T, M, N)
    go = our_launcher(k, [din, out])
    go(); torch.cuda.synchronize()
    u = out
    assert 0.0 <= float(u.min()) and float(u.max()) < 1.0
    our, gg = bench(go)
    refs = {}
    tt, _ = _fg_time(lambda: torch.rand(M, N, device="cuda")); refs["torch"] = tt
    record("C5", "rand", f"[{M},{N}]", our, refs, nbytes=M * N * 4,
           note="write-only; philox compute", graph_ours=gg)
    del out; _cleanup()

    # dropout = select(rand>=p, x/(1-p), 0)
    p = 0.25; inv = 1.0 / (1.0 - p)
    mlir = f"""
func.func @dp(%x: {O}) -> {O} {{
  %r = intent.rand {{seed = 7 : i64}} : {O}
  %pc = arith.constant dense<{p}> : {O}
  %keep = intent.map_binary %r, %pc {{fn = "ge"}} : ({O}, {O}) -> tensor<{M}x{N}xi1>
  %iv = arith.constant dense<{inv}> : {O}
  %sc = intent.map_binary %x, %iv {{fn = "mul"}} : ({O}, {O}) -> {O}
  %z = arith.constant dense<0.0> : {O}
  %o = intent.select %keep, %sc, %z : (tensor<{M}x{N}xi1>, {O}, {O}) -> {O}
  return %o : {O}
}}"""
    x = torch.randn(M, N, device="cuda")
    out = torch.empty(M, N, device="cuda")
    k = _mk("convert-intent-rng-to-gpu", mlir, T, M, N)
    go = our_launcher(k, [x, out])
    go(); torch.cuda.synchronize()
    our, gg = bench(go)
    refs = {}
    tt, _ = _fg_time(lambda: torch.nn.functional.dropout(x, p, training=True)); refs["torch"] = tt
    # dropout output is random per-call; verify shape/finiteness only.
    def _drop_ok(r):
        o = r[0] if isinstance(r, tuple) else r
        return o.shape == x.shape and torch.isfinite(o).all()
    refs["flag_gems"] = fg_time(
        lambda: FG.dropout(x, 0.25, train=True), verify=_drop_ok)
    record("C5", "dropout p=.25", f"[{M},{N}]", our, refs, nbytes=M * N * 4 * 2,
           note="read x + write; fused mask", graph_ours=gg)
    del x, out, din; _cleanup()


# ======================================================================== #
# C6  arg-reduce / sort                                                     #
# ======================================================================== #
def _thr_arg(N):
    # Moderate block (<=256 threads) that divides N, so each lane does plenty of
    # COALESCED work (K = N/T >= 16) and the shared-memory reduction tree has few
    # barrier stages. A 1024-thread block is latency-bound here: the argmax input
    # is L2-resident, so the 10-stage block sync — not HBM — is the cost, and a
    # 256-thread block (8 stages, 4x the work/lane) beats CUB/flag_gems.
    t = 256
    while t > 1 and N % t != 0:
        t //= 2
    return t


def c6_argmax():
    print("\n[C6] arg-reduce (argmax)")
    M, N = 4096, 4096
    T = _thr_arg(N)
    mlir = f"""
func.func @am(%x: tensor<{M}x{N}xf32>) -> tensor<{M}xi32> {{
  %i = intent.arg_reduce %x {{fn = "argmax", axis = 1 : i64}} : (tensor<{M}x{N}xf32>) -> tensor<{M}xi32>
  return %i : tensor<{M}xi32>
}}"""
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda")
    out = torch.empty(M, dtype=torch.int32, device="cuda")
    k = _mk("convert-intent-arg-reduce-to-gpu", mlir, T, M, N)
    go = our_launcher(k, [x, out])
    go(); torch.cuda.synchronize()
    assert torch.equal(out.to(torch.int64), torch.argmax(x, 1))
    our, gg = bench(go)
    refs = {}
    tt, _ = _fg_time(lambda: torch.argmax(x, 1)); refs["torch"] = tt
    ref_am = torch.argmax(x, 1)
    refs["flag_gems"] = fg_time(
        lambda: FG.argmax(x, dim=1),
        verify=lambda r: torch.equal(r.to(torch.int64), ref_am))
    record("C6", "argmax ax1", f"[{M},{N}] T={T}", our, refs,
           nbytes=M * N * 4, note="read-once reduce", graph_ours=gg)
    del x, out; _cleanup()


def _thr_sort(N):
    P = 1
    while P < N:
        P <<= 1
    t = 256
    while t > P or P % t != 0:
        t >>= 1
    return t


def c6_sort():
    print("\n[C6] sort / topk (cost-model hybrid: bitonic near-pow2 / LSD radix "
          "large-pad)")
    # honest realistic-ish: N=2048 row (bitonic single-block; N=4096 balloons
    # shared to 32KB + 78 stages, still measured below as the stress point).
    for M, N in [(4096, 2048), (4096, 4096)]:
        T = _thr_sort(N)
        mlir = f"""
func.func @so(%x: tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32> {{
  %v, %i = intent.sort %x {{descending = false, axis = 1 : i64}} : (tensor<{M}x{N}xf32>) -> (tensor<{M}x{N}xf32>, tensor<{M}x{N}xi32>)
  return %v : tensor<{M}x{N}xf32>
}}"""
        torch.manual_seed(0)
        x = torch.randn(M, N, device="cuda")
        out = torch.empty(M, N, device="cuda")
        try:
            k = _mk("convert-intent-sort-to-gpu", mlir, T, M, N)
            go = our_launcher(k, [x, out])
            go(); torch.cuda.synchronize()
            ok = torch.equal(out, torch.sort(x, 1).values)
        except Exception as e:
            print(f"  sort [{M},{N}] FAILED to build/launch: {e}")
            del x, out; _cleanup(); continue
        our, gg = bench(go)
        refs = {}
        tt, _ = _fg_time(lambda: torch.sort(x, 1).values); refs["torch"] = tt
        ref_sv = torch.sort(x, 1).values
        refs["flag_gems"] = fg_time(
            lambda: FG.sort(x, dim=1)[0],
            verify=lambda r: torch.equal(r, ref_sv))
        record("C6", "sort ax1", f"[{M},{N}] T={T}", our, refs,
               nbytes=M * N * 4 * 2, cls="serial",
               note=f"pow2 N -> warp-match radix; exact={ok}", graph_ours=gg)
        del x, out; _cleanup()


# ======================================================================== #
# C7  pool / upsample / conv / polar                                       #
# ======================================================================== #
def c7_pool():
    print("\n[C7] pool / upsample (NCHW movers)")
    N, C, H, W = 32, 64, 56, 56

    # avg_pool2d k2 s2 -> 28x28. Flat grid-stride: full 256-thread blocks over
    # the whole output (no OW % T divisibility), so every SM stays busy.
    OH = OW = 28; T = 256
    I = f"tensor<{N}x{C}x{H}x{W}xf32>"; O = f"tensor<{N}x{C}x{OH}x{OW}xf32>"
    mlir = f'func.func @ap(%a: {I}) -> {O} {{ %o = intent.pool %a {{fn = "avg", kH = 2 : i64, kW = 2 : i64, sH = 2 : i64, sW = 2 : i64, padH = 0 : i64, padW = 0 : i64}} : ({I}) -> {O}\n return %o : {O} }}'
    torch.manual_seed(0)
    a = torch.randn(N, C, H, W, device="cuda")
    out = torch.empty(N, C, OH, OW, device="cuda")
    k = _mk("convert-intent-pool-to-gpu", mlir, T)
    go = our_launcher(k, [a, out])
    go(); torch.cuda.synchronize()
    ref = torch.nn.functional.avg_pool2d(a, 2, 2, 0)
    assert float((out - ref).norm() / ref.norm()) < 1e-3
    our, gg = bench(go)
    refs = {}
    tt, _ = _fg_time(lambda: torch.nn.functional.avg_pool2d(a, 2, 2, 0)); refs["torch"] = tt
    ref_avg = torch.nn.functional.avg_pool2d(a, 2, 2, 0)
    refs["flag_gems"] = fg_time(
        lambda: FG.avg_pool2d(a, 2, stride=2, padding=0),
        verify=lambda r: float((r - ref_avg).norm() / ref_avg.norm()) < 1e-3)
    nb = N * C * H * W * 4 + N * C * OH * OW * 4
    record("C7", "avg_pool2d k2s2", f"[{N},{C},{H},{W}]", our, refs, nbytes=nb,
           graph_ours=gg)
    del out; _cleanup()

    # max_pool2d_with_indices k2 s2
    Xi = f"tensor<{N}x{C}x{OH}x{OW}xi64>"
    mlir = f'func.func @mp(%a: {I}) -> ({O}, {Xi}) {{ %o, %i = intent.pool %a {{fn = "max", kH = 2 : i64, kW = 2 : i64, sH = 2 : i64, sW = 2 : i64, padH = 0 : i64, padW = 0 : i64}} : ({I}) -> ({O}, {Xi})\n return %o, %i : {O}, {Xi} }}'
    out = torch.empty(N, C, OH, OW, device="cuda")
    idx = torch.empty(N, C, OH, OW, device="cuda", dtype=torch.int64)
    k = _mk("convert-intent-pool-to-gpu", mlir, T)
    go = our_launcher(k, [a, out, idx])
    go(); torch.cuda.synchronize()
    rv, ri = torch.nn.functional.max_pool2d_with_indices(a, 2, 2, 0)
    assert torch.equal(idx, ri)
    our, gg = bench(go)
    refs = {}
    tt, _ = _fg_time(lambda: torch.nn.functional.max_pool2d_with_indices(a, 2, 2, 0)); refs["torch"] = tt
    ref_mv, _ref_mi = torch.nn.functional.max_pool2d_with_indices(a, 2, 2, 0)
    refs["flag_gems"] = fg_time(
        lambda: FG.max_pool2d_with_indices(a, 2, stride=2, padding=0),
        verify=lambda r: torch.equal(r[0], ref_mv))
    nb = N * C * H * W * 4 + N * C * OH * OW * (4 + 8)
    record("C7", "max_pool2d+idx", f"[{N},{C},{H},{W}]", our, refs, nbytes=nb,
           graph_ours=gg)
    del out, idx; _cleanup()

    # upsample_nearest2d scale2 -> 112x112. Flat grid-stride, full 256-thread
    # blocks over the whole output (no OW % T divisibility).
    OH = OW = 112; T = 256
    O = f"tensor<{N}x{C}x{OH}x{OW}xf32>"
    mlir = f'func.func @up(%a: {I}) -> {O} {{ %o = intent.upsample %a {{mode = "nearest"}} : ({I}) -> {O}\n return %o : {O} }}'
    out = torch.empty(N, C, OH, OW, device="cuda")
    k = _mk("convert-intent-upsample-to-gpu", mlir, T)
    go = our_launcher(k, [a, out])
    go(); torch.cuda.synchronize()
    ref = torch.nn.functional.interpolate(a, scale_factor=2, mode="nearest")
    assert torch.equal(out, ref)
    our, gg = bench(go)
    refs = {}
    tt, _ = _fg_time(lambda: torch.nn.functional.interpolate(a, scale_factor=2, mode="nearest")); refs["torch"] = tt
    ref_up = torch.nn.functional.interpolate(a, scale_factor=2, mode="nearest")
    refs["flag_gems"] = fg_time(
        lambda: FG.upsample_nearest2d(a, (2 * H, 2 * W), None, None),
        verify=lambda r: torch.equal(r, ref_up))
    nb = N * C * H * W * 4 + N * C * OH * OW * 4
    record("C7", "upsample_near x2", f"[{N},{C},{H},{W}]", our, refs, nbytes=nb,
           graph_ours=gg)
    del a, out; _cleanup()


def c7_conv():
    print("\n[C7] conv2d (fp16 implicit-GEMM on nvgpu.mma.sync tensor cores)")
    # NCHW [32,64,56,56] k3 pad1 s1 -> out 56x56. Implicit GEMM: M=N*OH*OW,
    # N_gemm=Cout, K=Cin*3*3; im2col never materialized (the OOB-guarded conv
    # address math IS the f16 smem tile-fill). fp16 input mode -> rel ~1e-2 vs
    # fp32 torch (cuDNN itself runs TF32).
    N, Cin, Cout, H, W, T = 32, 64, 64, 56, 56, 56
    OH = OW = 56
    I = f"tensor<{N}x{Cin}x{H}x{W}xf32>"
    Wt = f"tensor<{Cout}x{Cin}x3x3xf32>"
    O = f"tensor<{N}x{Cout}x{OH}x{OW}xf32>"
    mlir = f"""
func.func @cv(%x: {I}, %w: {Wt}) -> {O} {{
  %o = intent.conv %x, %w {{stride = [1, 1], padding = [1, 1], dilation = [1, 1], groups = 1 : i64}} : ({I}, {Wt}) -> {O}
  return %o : {O}
}}"""
    torch.manual_seed(0)
    x = torch.randn(N, Cin, H, W, device="cuda")
    w = torch.randn(Cout, Cin, 3, 3, device="cuda")
    out = torch.empty(N, Cout, OH, OW, device="cuda")
    try:
        k = ConvMMAKernel(mlir)
        go = our_launcher(k, [x, w, out])
        go(); torch.cuda.synchronize()
        ref = torch.nn.functional.conv2d(x, w, None, 1, 1)
        rel = float((out - ref).norm() / ref.norm())
        assert rel < 2e-2, f"conv2d-mma rel {rel} (fp16 tensor-core)"
    except Exception as e:
        print(f"  conv2d FAILED: {e}"); traceback.print_exc()
        del x, w, out; _cleanup(); return
    our, gg = bench(go)
    refs = {}
    tt, _ = _fg_time(lambda: torch.nn.functional.conv2d(x, w, None, 1, 1)); refs["torch"] = tt
    ref_cv = torch.nn.functional.conv2d(x, w, None, 1, 1)
    refs["flag_gems"] = fg_time(
        lambda: FG.conv2d(x, w, None, stride=(1, 1), padding=(1, 1),
                          dilation=(1, 1), groups=1),
        verify=lambda r: float((r - ref_cv).norm() / ref_cv.norm()) < 2e-2)
    flop = 2.0 * N * Cout * OH * OW * Cin * 3 * 3
    record("C7", "conv2d k3 s1 p1", f"[{N},{Cin},{H},{W}]->{Cout}", our, refs,
           flop=flop, cls="compute",
           note=f"rel={rel:.1e}; fp16 mma.sync implicit-GEMM; cuDNN=TF32 ref",
           graph_ours=gg)
    del x, w, out; _cleanup()


def c7_polar():
    print("\n[C7] polar (complex elementwise)")
    M, Nc, T = 4096, 4096, 256
    A = f"tensor<{M}x{Nc}xf32>"
    mlir = f'func.func @po(%a: {A}, %b: {A}) -> ({A}, {A}) {{ %r, %i = intent.polar %a, %b : ({A}, {A}) -> ({A}, {A})\n return %r, %i : {A}, {A} }}'
    torch.manual_seed(0)
    absv = torch.rand(M, Nc, device="cuda") + 0.1
    ang = torch.rand(M, Nc, device="cuda") * 6 - 3
    re = torch.empty(M, Nc, device="cuda")
    im = torch.empty(M, Nc, device="cuda")
    k = _mk("convert-intent-polar-to-gpu", mlir, T)
    go = our_launcher(k, [absv, ang, re, im])
    go(); torch.cuda.synchronize()
    ref = torch.view_as_real(torch.polar(absv, ang))
    assert float((re - ref[..., 0]).norm() / ref[..., 0].norm()) < 1e-3
    our, gg = bench(go)
    refs = {}
    tt, _ = _fg_time(lambda: torch.polar(absv, ang)); refs["torch"] = tt
    ref_po = torch.view_as_real(torch.polar(absv, ang))
    refs["flag_gems"] = fg_time(
        lambda: FG.polar(absv, ang),
        verify=lambda r: float((torch.view_as_real(r)[..., 0] - ref_po[..., 0]).norm()
                               / ref_po[..., 0].norm()) < 1e-3)
    nb = M * Nc * 4 * (2 + 2)
    record("C7", "polar", f"[{M},{Nc}]", our, refs, nbytes=nb, graph_ours=gg)
    del absv, ang, re, im; _cleanup()


# ======================================================================== #
# C8  compact (nonzero / masked_select) -- MULTI-BLOCK parallel compaction   #
# ======================================================================== #
def c8_compact():
    print("\n[C8] compact (nonzero / masked_select, multi-block stream compaction)")
    from intentir.benchmarks.row_reduction import CompactKernel
    # data-dependent output extent -> ref not graph-capturable; use do_bench.
    for R, Cc in [(2048, 2048), (4096, 4096)]:
        T = 256
        Nt = R * Cc
        F = f"tensor<{R}x{Cc}xf32>"
        # ---- masked_select ------------------------------------------------- #
        mlir = f"""
func.func @ms(%data: {F}, %mask: {F}) -> tensor<?xf32> {{
  %r = intent.masked_select %data, %mask : ({F}, {F}) -> tensor<?xf32>
  return %r : tensor<?xf32>
}}"""
        torch.manual_seed(1)
        data = torch.randn(R, Cc, device="cuda")
        maskb = (torch.rand(R, Cc, device="cuda") < 0.5)
        maskf = maskb.to(torch.float32)
        sel = torch.zeros(Nt, device="cuda")
        count = torch.zeros(1, device="cuda", dtype=torch.int64)
        try:
            k = CompactKernel("ms", T, mlir)
            go = k.thunk([data, maskf, sel, count])
            go(); torch.cuda.synchronize()
            nnz = int(maskb.sum())
            ok = torch.equal(sel[:nnz], torch.masked_select(data, maskb))
        except Exception as e:
            print(f"  masked_select [{R},{Cc}] FAILED: {e}")
            del data, maskb, maskf, sel, count; _cleanup(); continue
        # deterministic, fixed max buffer -> CUDA-graph safe (3 launches captured).
        our, gg = bench(go, want_graph=True)
        refs = {}
        tt, _ = _fg_time(lambda: torch.masked_select(data, maskb), want_graph=False); refs["torch"] = tt
        ref_ms = torch.masked_select(data, maskb)
        refs["flag_gems"] = fg_time(
            lambda: FG.masked_select(data, maskb), want_graph=False,
            verify=lambda r: torch.equal(r, ref_ms))
        nb = R * Cc * (4 + 4)  # mask + data read
        record("C8", "masked_select", f"[{R},{Cc}] ~50%", our, refs, nbytes=nb,
               cls="mem", note=f"3-launch multi-block compaction; exact={ok}",
               graph_ours=gg, graph_ref=False)
        del data, maskb, maskf, sel, count; _cleanup()

        # ---- nonzero ------------------------------------------------------- #
        mlirn = f"""
func.func @nz(%cond: {F}) -> tensor<?x2xi64> {{
  %r = intent.nonzero %cond : ({F}) -> tensor<?x2xi64>
  return %r : tensor<?x2xi64>
}}"""
        torch.manual_seed(3)
        cond = torch.where(torch.rand(R, Cc, device="cuda") < 0.5,
                           torch.randn(R, Cc, device="cuda"),
                           torch.zeros(R, Cc, device="cuda")).contiguous()
        out = torch.zeros(Nt, 2, device="cuda", dtype=torch.int64)
        countn = torch.zeros(1, device="cuda", dtype=torch.int64)
        try:
            kn = CompactKernel("nz", T, mlirn)
            gon = kn.thunk([cond, out, countn])
            gon(); torch.cuda.synchronize()
            nnzn = int((cond != 0).sum())
            okn = torch.equal(out[:nnzn], torch.nonzero(cond))
        except Exception as e:
            print(f"  nonzero [{R},{Cc}] FAILED: {e}")
            del cond, out, countn; _cleanup(); continue
        ourn, ggn = bench(gon, want_graph=True)
        refsn = {}
        ttn, _ = _fg_time(lambda: torch.nonzero(cond), want_graph=False); refsn["torch"] = ttn
        # R-BENCH-4: fg.nonzero must exist with a matching signature before use.
        ref_nz = torch.nonzero(cond)
        refsn["flag_gems"] = (
            fg_time(lambda: FG.nonzero(cond), want_graph=False,
                    verify=lambda r: torch.equal(r, ref_nz))
            if (FG is not None and hasattr(FG, "nonzero")) else None)
        record("C8", "nonzero", f"[{R},{Cc}] ~50%", ourn, refsn, nbytes=R * Cc * 4,
               cls="mem", note=f"3-launch multi-block compaction; exact={okn}",
               graph_ours=ggn, graph_ref=False)
        del cond, out, countn; _cleanup()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  flag_gems={'yes' if FG else 'NO'}")
    print(f"peak HBM ~{PEAK_GBPS:.0f} GB/s   f32 peak ~{PEAK_TFLOPS_F32:.0f} TFLOP/s")
    stages = [c3_gather, c4_layout, c5_rng, c6_argmax, c6_sort, c7_pool,
              c7_conv, c7_polar, c8_compact]
    for st in stages:
        try:
            st()
        except Exception as e:
            print(f"  !! stage {st.__name__} errored: {e}")
            traceback.print_exc()
        _cleanup()

    print("\n\n===== SUMMARY TABLE =====")
    hdr = f"{'op':22s} {'shape':22s} {'our_us':>9s} {'fg_us':>9s} {'torch_us':>9s} {'ratio':>6s} {'perf':>11s} {'peak':>9s} {'verdict':>7s}"
    print(hdr)
    for r in ROWS:
        print(f"{r['op']:22s} {r['shape']:22s} {r['our_us']:>9.2f} "
              f"{(r['flaggems_us'] if r['flaggems_us'] else 0):>9.2f} "
              f"{(r['torch_us'] if r['torch_us'] else 0):>9.2f} "
              f"{(r['ratio'] if r['ratio'] else 0):>6.2f} "
              f"{r['our_perf']:>11s} {r['peak']:>9s} {r['verdict']:>7s}")
    out = Path(__file__).resolve().parent / "bench_clusters_results.json"
    out.write_text(json.dumps(ROWS, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
