"""M5 autotuner demonstration: the search FINDS the optimum, no hand-tuning.

For each re-tuned op we report the FIXED-schedule ratio (the baked constant /
one-line heuristic the backend shipped) vs the AUTOTUNED ratio (the argmin config
the driver picked by ``lower -> do_bench -> argmin``). ratio = our_us / faster
reference (torch / flag_gems); <1 wins, lower is better.

The point is twofold:
  1. the autotuner reproduces M1's hand-set argmax T=256 AUTOMATICALLY (the fixed
     1024-thread default is latency-bound; the driver measures and rejects it);
  2. it can search PAST the shipped default (conv/contraction tile sweep).

Run (quiet GPU):
  export PATH=/usr/local/cuda-12.8/bin:$PATH
  PYTHONPATH=python python3 python/tests/perf/autotune_demo.py
"""

from __future__ import annotations

import ctypes
import json
import traceback
from pathlib import Path

import torch
import triton

from intentir.backend.cuda.autotune import AutotuneDriver, TuneTask
from intentir.backend.cuda.launch import I64Arg, PtrArg, _marshal
from intentir.backend.cuda.kernel_driver import (ConvMMAKernel, MatmulKernel,
                                               RowReductionKernel)


def _flag_gems():
    try:
        import flag_gems
        from intentir.benchmarks.flaggems_shim import apply_flaggems_launch_shim
        apply_flaggems_launch_shim()
        return flag_gems
    except Exception:
        return None


FG = _flag_gems()


def our_launcher(k, tensors):
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
    if want_graph:
        try:
            ms = triton.testing.do_bench_cudagraph(fn, rep=100, return_mode="median")
            return ms * 1e3
        except Exception:
            torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=50, rep=200, return_mode="median")
    return ms * 1e3


def ref_us(fn):
    try:
        fn(); torch.cuda.synchronize()
    except Exception:
        return None
    return bench(fn)


def fg_us(fn, verify):
    if FG is None or fn is None:
        return None
    try:
        r = fn(); torch.cuda.synchronize()
        if not verify(r):
            return None
    except Exception:
        return None
    return bench(fn)


ROWS = []


def report(op, shape, fixed_cfg, fixed_us, tuned_cfg, tuned_us, best_ref, ref_lbl):
    fr = fixed_us / best_ref if best_ref else None
    tr = tuned_us / best_ref if best_ref else None
    speedup = fixed_us / tuned_us if tuned_us else None
    ROWS.append(dict(op=op, shape=shape, fixed_cfg=fixed_cfg,
                     fixed_us=round(fixed_us, 2), fixed_ratio=round(fr, 3) if fr else None,
                     tuned_cfg=tuned_cfg, tuned_us=round(tuned_us, 2),
                     tuned_ratio=round(tr, 3) if tr else None,
                     ref=ref_lbl, ref_us=round(best_ref, 2) if best_ref else None,
                     self_speedup=round(speedup, 3) if speedup else None))
    print(f"\n  {op}  {shape}")
    print(f"    FIXED    {fixed_cfg}  -> {fixed_us:8.2f} us  ratio={fr:.3f}"
          if fr else f"    FIXED    {fixed_cfg} -> {fixed_us:.2f} us")
    print(f"    AUTOTUNE {tuned_cfg}  -> {tuned_us:8.2f} us  ratio={tr:.3f}"
          if tr else f"    AUTOTUNE {tuned_cfg} -> {tuned_us:.2f} us")
    print(f"    ref[{ref_lbl}]={best_ref:.2f}us   fixed->tuned self-speedup="
          f"{speedup:.3f}x" if best_ref else f"    self-speedup={speedup:.3f}x")


# ------------------------------------------------------------------------- #
# 1. argmax — the headline T-sweep (M1's story, reproduced automatically)
# ------------------------------------------------------------------------- #
def demo_argmax(drv):
    print("\n[1] argmax ax1  — block-threads T sweep")
    M, N = 4096, 4096
    mlir = f"""
func.func @am(%x: tensor<{M}x{N}xf32>) -> tensor<{M}xi32> {{
  %i = intent.arg_reduce %x {{fn = "argmax", axis = 1 : i64}} : (tensor<{M}x{N}xf32>) -> tensor<{M}xi32>
  return %i : tensor<{M}xi32>
}}"""
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda")
    out = torch.empty(M, dtype=torch.int32, device="cuda")
    ref = torch.argmax(x, 1)

    def build(cfg):
        k = RowReductionKernel("am", {"M": M, "N": N}, cfg["T"],
                               pass_name="convert-intent-arg-reduce-to-gpu",
                               mlir_text=mlir)
        go = our_launcher(k, [x, out])
        go(); torch.cuda.synchronize()
        ok = torch.equal(out.to(torch.int64), ref)
        return go, ok

    # FIXED schedule = the bad 1024-thread default M1 diagnosed.
    fixed_cfg = {"T": 1024}
    fgo, fok = build(fixed_cfg)
    assert fok
    fixed_us = bench(fgo)

    # AUTOTUNE
    task = TuneTask("argmax", f"[{M},{N}]", "f32", "mover", {"divisor": N}, build)
    tuned_cfg = drv.tune(task, force=True, verbose=True)
    tgo, tok = build(tuned_cfg)
    assert tok, "tuned config broke correctness"
    tuned_us = bench(tgo)

    t = ref_us(lambda: torch.argmax(x, 1))
    fg = fg_us(lambda: FG.argmax(x, dim=1),
               lambda r: torch.equal(r.to(torch.int64), ref))
    best_ref, lbl = _best(t, fg)
    report("argmax ax1", f"[{M},{N}]", fixed_cfg, fixed_us, tuned_cfg, tuned_us,
           best_ref, lbl)
    del x, out


# ------------------------------------------------------------------------- #
# 2. gather — a mover T-sweep
# ------------------------------------------------------------------------- #
def demo_gather(drv):
    print("\n[2] gather ax1  — mover block-threads T sweep")
    M, N = 8192, 4096
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
    ref = torch.gather(data, 1, idx)

    def build(cfg):
        k = RowReductionKernel("g", {"M": M, "N": N}, cfg["T"],
                               pass_name="convert-intent-gather-to-gpu",
                               mlir_text=mlir)
        go = our_launcher(k, [data, idx, out])
        go(); torch.cuda.synchronize()
        ok = torch.equal(out, ref)
        return go, ok

    # FIXED = the shipped autoThreads default (caps at 256 for divisible N).
    fixed_cfg = {"T": 256}
    fgo, fok = build(fixed_cfg)
    assert fok
    fixed_us = bench(fgo)

    task = TuneTask("gather", f"[{M},{N}]", "f32", "mover", {"divisor": N}, build)
    tuned_cfg = drv.tune(task, force=True, verbose=True)
    tgo, tok = build(tuned_cfg)
    assert tok
    tuned_us = bench(tgo)

    t = ref_us(lambda: torch.gather(data, 1, idx))
    fg = fg_us(lambda: FG.gather(data, 1, idx), lambda r: torch.equal(r, ref))
    best_ref, lbl = _best(t, fg)
    report("gather ax1", f"[{M},{N}]", fixed_cfg, fixed_us, tuned_cfg, tuned_us,
           best_ref, lbl)
    del data, idx, out


# ------------------------------------------------------------------------- #
# 3. contraction (matmul) — BM/BN/BK/numStages tile sweep on mma.sync
# ------------------------------------------------------------------------- #
def demo_contraction(drv):
    print("\n[3] contraction mk,nk->mn  — BM/BN/BK/numStages tile sweep")
    M, N, K = 512, 512, 512
    A = f"tensor<{M}x{K}xf16>"; Wt = f"tensor<{N}x{K}xf16>"; Y = f"tensor<{M}x{N}xf32>"
    mlir = f"""
func.func @mm(%a: {A}, %w: {Wt}) -> {Y} {{
  %y = intent.contraction %a, %w {{subscripts = "mk,nk->mn"}} : ({A}, {Wt}) -> {Y}
  return %y : {Y}
}}"""
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)
    y = torch.empty(M, N, device="cuda", dtype=torch.float32)
    ref = (a.float() @ w.float().t())

    def build(cfg):
        k = MatmulKernel(mlir, mma=True, block_m=cfg["BM"], block_n=cfg["BN"],
                         block_k=cfg["BK"], num_stages=cfg["numStages"],
                         smem_pad=cfg.get("smemPadK", 0))
        go = our_launcher(k, [a, w, y])  # stream read INSIDE go (graph-captured)
        go(); torch.cuda.synchronize()
        rel = float((y - ref).norm() / ref.norm())
        return go, (rel < 2e-2)

    # FIXED = the shipped mma default (BM=64,BN=128,BK=32,stages=3).
    fixed_cfg = {"BM": 64, "BN": 128, "BK": 32, "numStages": 3}
    fgo, fok = build(fixed_cfg)
    assert fok, "fixed matmul correctness"
    fixed_us = bench(fgo)

    task = TuneTask("contraction", f"[{M},{N},{K}]", "f16", "contraction",
                    {"M": M, "N": N, "K": K}, build)
    tuned_cfg = drv.tune(task, force=True, verbose=True)
    tgo, tok = build(tuned_cfg)
    assert tok
    tuned_us = bench(tgo)

    t = ref_us(lambda: torch.matmul(a, w.t()))
    best_ref, lbl = _best(t, None)
    report("matmul", f"[{M},{N},{K}]", fixed_cfg, fixed_us, tuned_cfg, tuned_us,
           best_ref, lbl)
    del a, w, y


# ------------------------------------------------------------------------- #
# 4. conv2d — implicit-GEMM BM/BN tile sweep
# ------------------------------------------------------------------------- #
def demo_conv(drv):
    print("\n[4] conv2d k3 s1 p1  — implicit-GEMM BM/BN tile sweep")
    Nb, Cin, Cout, H, W = 32, 64, 64, 56, 56
    OH = OW = 56
    I = f"tensor<{Nb}x{Cin}x{H}x{W}xf32>"
    Wt = f"tensor<{Cout}x{Cin}x3x3xf32>"
    O = f"tensor<{Nb}x{Cout}x{OH}x{OW}xf32>"
    mlir = f"""
func.func @cv(%x: {I}, %w: {Wt}) -> {O} {{
  %o = intent.conv %x, %w {{stride = [1, 1], padding = [1, 1], dilation = [1, 1], groups = 1 : i64}} : ({I}, {Wt}) -> {O}
  return %o : {O}
}}"""
    torch.manual_seed(0)
    x = torch.randn(Nb, Cin, H, W, device="cuda")
    wt = torch.randn(Cout, Cin, 3, 3, device="cuda")
    out = torch.empty(Nb, Cout, OH, OW, device="cuda")
    ref = torch.nn.functional.conv2d(x, wt, None, 1, 1)
    # implicit-GEMM dims: M = Nb*OH*OW, N = Cout, K = Cin*3*3
    Mg, Ng = Nb * OH * OW, Cout

    def build(cfg):
        k = ConvMMAKernel(mlir, block_m=cfg["BM"], block_n=cfg["BN"])
        go = our_launcher(k, [x, wt, out])  # stream read INSIDE go (captured)
        go(); torch.cuda.synchronize()
        rel = float((out - ref).norm() / ref.norm())
        return go, (rel < 2e-2)

    fixed_cfg = {"BM": 64, "BN": 64}
    fgo, fok = build(fixed_cfg)
    assert fok, "fixed conv correctness"
    fixed_us = bench(fgo)

    task = TuneTask("conv2d", f"[{Nb},{Cin},{H},{W}]->{Cout}", "f16", "conv",
                    {"M": Mg, "N": Ng}, build)
    tuned_cfg = drv.tune(task, force=True, verbose=True)
    tgo, tok = build(tuned_cfg)
    assert tok
    tuned_us = bench(tgo)

    t = ref_us(lambda: torch.nn.functional.conv2d(x, wt, None, 1, 1))
    fg = fg_us(lambda: FG.conv2d(x, wt, None, stride=(1, 1), padding=(1, 1),
                                 dilation=(1, 1), groups=1),
               lambda r: float((r - ref).norm() / ref.norm()) < 2e-2)
    best_ref, lbl = _best(t, fg)
    report("conv2d k3s1p1", f"[{Nb},{Cin},{H},{W}]->{Cout}", fixed_cfg, fixed_us,
           tuned_cfg, tuned_us, best_ref, lbl)
    del x, wt, out


def _best(t, fg):
    present = {k: v for k, v in {"torch": t, "flag_gems": fg}.items() if v}
    if not present:
        return None, None
    lbl, v = min(present.items(), key=lambda kv: kv[1])
    return v, lbl


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  flag_gems={'yes' if FG else 'NO'}")
    # Fresh cache for the demo so the search is exercised end-to-end.
    cache = Path(__file__).resolve().parent / "autotune_demo_cache.json"
    if cache.exists():
        cache.unlink()
    drv = AutotuneDriver(cache_path=cache)
    for fn in (demo_argmax, demo_gather, demo_contraction, demo_conv):
        try:
            fn(drv)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {fn.__name__} errored: {e}")
            traceback.print_exc()
        gc_sync()

    print("\n\n===== AUTOTUNER SUMMARY (fixed -> tuned) =====")
    hdr = (f"{'op':16s} {'shape':22s} {'fixed_cfg':34s} {'fx_us':>8s} "
           f"{'fx_ratio':>8s} {'tuned_cfg':34s} {'tn_us':>8s} {'tn_ratio':>8s} "
           f"{'self':>6s}")
    print(hdr)
    for r in ROWS:
        print(f"{r['op']:16s} {r['shape']:22s} {str(r['fixed_cfg']):34s} "
              f"{r['fixed_us']:>8.2f} {(r['fixed_ratio'] or 0):>8.3f} "
              f"{str(r['tuned_cfg']):34s} {r['tuned_us']:>8.2f} "
              f"{(r['tuned_ratio'] or 0):>8.3f} {(r['self_speedup'] or 0):>6.3f}")
    outp = Path(__file__).resolve().parent / "autotune_demo_results.json"
    outp.write_text(json.dumps(ROWS, indent=2))
    print(f"\nwrote {outp}")
    print(f"cache persisted at {cache}")
    # Show cache reuse (resumable): the build path consults the cache first and
    # gets the winner WITHOUT re-searching (falls back to occupancy default for
    # cold shapes). This is the runtime-consult hook.
    print("\n--- cache reuse / runtime-consult hook (no re-search) ---")
    drv2 = AutotuneDriver(cache_path=cache)  # reload from disk
    print("  argmax [4096,4096] f32 ->",
          drv2.config_for("argmax", "[4096,4096]", "f32", "mover", {"divisor": 4096}))
    print("  COLD shape [7000,3000] f32 -> default",
          drv2.config_for("argmax", "[7000,3000]", "f32", "mover", {"divisor": 3000}))
    print("  cached keys:", list(json.loads(cache.read_text()).keys()))


def gc_sync():
    import gc
    gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
