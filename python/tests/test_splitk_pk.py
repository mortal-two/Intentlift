"""PK the xformers split-K attention forward + reduce lowerings.

Two M8 deliverables, lowered by C++ OpRewritePatterns to sm_120a cubins:

  (1) split-K FORWARD (`-convert-intent-loop-contraction-to-gpu`, the M8a
      fused-flash pass GENERALIZED): the online-softmax flash loop with an OUTER
      split-K axis. The split dim Z is the leading dim of K/V/O/LSE (batched-RHS
      contractions "qd,zkd->zqk" / "zqk,zkd->zqd"; Q stays rank-2, shared). It
      lowers to ONE tensor-core gpu.launch with grid.y = Z; each (q-tile, chunk)
      program runs M8a's body over its key chunk and writes a PARTIAL O[chunk]
      (= softmax(chunk)@V_chunk) + LSE[chunk] (= m + log(l)). Reuses the M8a Q
      stage / two WMMA matmuls / cooperative shared-memory online softmax + alpha
      rescale verbatim; only the global addressing (z-offset) and the LSE
      epilogue are new.

  (2) split-K REDUCE (`-convert-intent-splitk-reduce-to-gpu`, NEW pass): the
      online-LSE merge across chunks, NO matmul. One intent.loop scans the split
      axis of the packed partials, carrying (lse_max, sumexp, O_acc); lowers to a
      block-cooperative gpu.launch (one block per output row, one thread per
      head-dim element, scf.for over the split axis, no shared memory). Memory-
      bound -> parity is the honest floor.

Correctness (rel, f16 ~2e-2) vs an fp32 reference AND the composed cross-check
that the task demands: splitK_fwd partials fed through splitK_reduce == the M8a
full flash == torch SDPA. fp32/f16, do_bench kernel-only median, no host sync in
the timed region.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_splitk_pk.py -q -s
"""

from __future__ import annotations

import os
import pathlib

import pytest

from intentir.backend.compile import (CUDA_BIN, INTENT_OPT, MLIR_OPT, _run,
                                       lower_memref_module_to_kernels)
from intentir.backend.launch import load_module
from intentir.backend.orchestrate import (_F32Arg, _I64Arg, _PtrArg, _launch,
                                          _host_func_text,
                                          _map_input_aligned_ptrs,
                                          _parse_constants, _parse_launches,
                                          _parse_signature_blockargs,
                                          _resolve_const_index)

torch = pytest.importorskip("torch")

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


# --------------------------------------------------------------------------- #
# A generic bare-ptr launch harness: compile one func through `pass_name` +
# lower_memref_module_to_kernels(scf_to_cf=True), resolve the single gpu.launch.
# run(tensors, scalar) passes memrefs in signature order; an f32-by-value arg
# (the scale) is filled from `scalar`.
# --------------------------------------------------------------------------- #
class _Kernel:
    def __init__(self, mlir_text: str, pass_name: str):
        rr = _run([INTENT_OPT, pass_name], stdin=mlir_text)
        kernels = lower_memref_module_to_kernels(rr, scf_to_cf=True)
        if len(kernels) != 1:
            raise RuntimeError(f"expected one kernel, got {len(kernels)}")
        ck = kernels[0]
        self.func = load_module(ck.cubin).function(ck.kernel_name)

        _n, ftext = _host_func_text(ck.binary_mlir)
        _s, _o, ptr_blockargs = _parse_signature_blockargs(ftext)
        aligned = _map_input_aligned_ptrs(ptr_blockargs)
        consts = _parse_constants(ftext)
        sites = _parse_launches(ftext)
        if len(sites) != 1:
            raise RuntimeError(f"expected one gpu.launch_func, got {len(sites)}")
        site = sites[0]

        class _Shim:
            constants = consts

        self.grid = tuple(_resolve_const_index(_Shim, s) for s in site.grid_ssa)
        self.block = tuple(_resolve_const_index(_Shim, s) for s in site.block_ssa)
        self.shared = 0
        if site.dynamic_shared_ssa is not None:
            self.shared = int(consts[site.dynamic_shared_ssa][0])
            from intentir.backend.launch import _driver
            cuda = _driver()
            attr = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            cuda.cuFuncSetAttribute(self.func, attr, self.shared)

        self.plan: list[tuple] = []
        for op in site.args:
            if op.ty.startswith("!llvm.ptr"):
                self.plan.append(("ptr", aligned[op.ssa]))
            elif op.ty in ("i64", "index"):
                self.plan.append(("i64", int(consts[op.ssa][0])))
            elif op.ty == "f32":
                self.plan.append(("scalar",))
            else:
                raise RuntimeError(f"unhandled launch-arg {op.ty}")

    def run(self, tensors, scalar=None):
        call_args = []
        for entry in self.plan:
            if entry[0] == "ptr":
                call_args.append(_PtrArg(tensors[entry[1]].data_ptr()))
            elif entry[0] == "scalar":
                call_args.append(_F32Arg(float(scalar)))
            elif entry[0] == "i64":
                call_args.append(_I64Arg(entry[1]))
        _launch(self.func, self.grid, self.block, call_args, shared=self.shared)


def _splitk_fwd_mlir(z: int, sq: int, skc: int, d: int) -> str:
    QT = f"tensor<{sq}x{d}xf16>"
    KT = f"tensor<{z}x{skc}x{d}xf16>"
    Kj = f"tensor<{z}x{skc}x{d}xf16>"
    M1 = f"tensor<{z}x{sq}x1xf32>"
    OT = f"tensor<{z}x{sq}x{d}xf32>"   # accumulator / PV / O output
    ST = f"tensor<{z}x{sq}x{skc}xf32>"  # scores S = Q@K^T (last dim = chunk keys)
    return f"""
func.func @splitk_fwd(%Q: {QT}, %K: {KT}, %V: {KT}, %scale: f32)
    -> ({OT}, {M1}) {{
  %m0 = arith.constant dense<0xFF800000> : {M1}
  %l0 = arith.constant dense<0.0> : {M1}
  %O0 = arith.constant dense<0.0> : {OT}
  %r:3 = intent.loop scan(%K, %V : {KT}, {KT})
                     init(%m0, %l0, %O0 : {M1}, {M1}, {OT})
                     {{scan_axes = [1, 1], tile_hint = 64 : i64}}
                     -> {M1}, {M1}, {OT} do {{
  ^bb0(%Kj: {Kj}, %Vj: {Kj}, %m: {M1}, %l: {M1}, %Oacc: {OT}):
    %S = intent.contraction %Q, %Kj {{subscripts = "qd,zkd->zqk"}} : ({QT}, {Kj}) -> {ST}
    %scsplat = tensor.splat %scale : {ST}
    %Ss = intent.map_binary %S, %scsplat {{fn = "mul"}} : ({ST}, {ST}) -> {ST}
    %mb = intent.reduce %Ss {{fn = "max", axes = [2], assoc = "associative"}} : ({ST}) -> {M1}
    %mn = intent.map_binary %m, %mb {{fn = "max"}} : ({M1}, {M1}) -> {M1}
    %Sc = intent.map_binary %Ss, %mn {{fn = "sub"}} : ({ST}, {M1}) -> {ST}
    %P = intent.map_unary %Sc {{fn = "exp"}} : ({ST}) -> {ST}
    %co = intent.map_binary %m, %mn {{fn = "sub"}} : ({M1}, {M1}) -> {M1}
    %al = intent.map_unary %co {{fn = "exp"}} : ({M1}) -> {M1}
    %ps = intent.reduce %P {{fn = "sum", axes = [2], assoc = "float_nondeterministic"}} : ({ST}) -> {M1}
    %ls = intent.map_binary %l, %al {{fn = "mul"}} : ({M1}, {M1}) -> {M1}
    %ln = intent.map_binary %ls, %ps {{fn = "add"}} : ({M1}, {M1}) -> {M1}
    %PV = intent.contraction %P, %Vj {{subscripts = "zqk,zkd->zqd"}} : ({ST}, {Kj}) -> {OT}
    %Os = intent.map_binary %Oacc, %al {{fn = "mul"}} : ({OT}, {M1}) -> {OT}
    %On = intent.map_binary %Os, %PV {{fn = "add"}} : ({OT}, {OT}) -> {OT}
    intent.yield %mn, %ln, %On : {M1}, {M1}, {OT}
  }}
  %O = intent.map_binary %r#2, %r#1 {{fn = "div"}} : ({OT}, {M1}) -> {OT}
  %logl = intent.map_unary %r#1 {{fn = "log"}} : ({M1}) -> {M1}
  %lse = intent.map_binary %r#0, %logl {{fn = "add"}} : ({M1}, {M1}) -> {M1}
  return %O, %lse : {OT}, {M1}
}}
"""


def _splitk_reduce_mlir(s: int, mq: int, d: int) -> str:
    OSK = f"tensor<{s}x{mq}x{d}xf32>"
    LSK = f"tensor<{s}x{mq}x1xf32>"
    S1 = f"tensor<{mq}x1xf32>"
    OT = f"tensor<{mq}x{d}xf32>"
    OS = f"tensor<{mq}x{d}xf32>"
    return f"""
func.func @splitk_reduce(%Osk: {OSK}, %Lsk: {LSK}) -> {OT} {{
  %m0 = arith.constant dense<0xFF800000> : {S1}
  %se0 = arith.constant dense<0.0> : {S1}
  %oa0 = arith.constant dense<0.0> : {OT}
  %r:3 = intent.loop scan(%Osk, %Lsk : {OSK}, {LSK})
                     init(%m0, %se0, %oa0 : {S1}, {S1}, {OT})
                     {{scan_axes = [0, 0], tile_hint = 1 : i64}}
                     -> {S1}, {S1}, {OT} do {{
  ^bb0(%Os: {OS}, %Ls: {S1}, %mmax: {S1}, %se: {S1}, %oa: {OT}):
    %mn = intent.map_binary %mmax, %Ls {{fn = "max"}} : ({S1}, {S1}) -> {S1}
    %cs = intent.map_binary %mmax, %mn {{fn = "sub"}} : ({S1}, {S1}) -> {S1}
    %corr = intent.map_unary %cs {{fn = "exp"}} : ({S1}) -> {S1}
    %ws = intent.map_binary %Ls, %mn {{fn = "sub"}} : ({S1}, {S1}) -> {S1}
    %w = intent.map_unary %ws {{fn = "exp"}} : ({S1}) -> {S1}
    %se1 = intent.map_binary %se, %corr {{fn = "mul"}} : ({S1}, {S1}) -> {S1}
    %senew = intent.map_binary %se1, %w {{fn = "add"}} : ({S1}, {S1}) -> {S1}
    %oa1 = intent.map_binary %oa, %corr {{fn = "mul"}} : ({OT}, {S1}) -> {OT}
    %wO = intent.map_binary %Os, %w {{fn = "mul"}} : ({OS}, {S1}) -> {OT}
    %oanew = intent.map_binary %oa1, %wO {{fn = "add"}} : ({OT}, {OT}) -> {OT}
    intent.yield %mn, %senew, %oanew : {S1}, {S1}, {OT}
  }}
  %O = intent.map_binary %r#2, %r#1 {{fn = "div"}} : ({OT}, {S1}) -> {OT}
  return %O : {OT}
}}
"""


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def _bench(fn):
    import triton
    return triton.testing.do_bench(fn, warmup=50, rep=200, return_mode="median")


# --------------------------------------------------------------------------- #
# (1) split-K FORWARD: per-chunk partial O + LSE, vs an fp32 per-chunk reference.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("z,sq,skc,d", [
    (2, 128, 64, 64),   # one K-tile per chunk (numKT == 1)
    (4, 128, 64, 64),
    (2, 128, 128, 64),  # multi-K-tile chunk (numKT == 2) x batched z-offset
])
def test_splitk_fwd_pk(z, sq, skc, d):
    torch.manual_seed(0)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(z, skc, d, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(z, skc, d, device="cuda", dtype=torch.float16).contiguous()
    scale = 1.0 / (d ** 0.5)

    kern = _Kernel(_splitk_fwd_mlir(z, sq, skc, d),
                   "-convert-intent-loop-contraction-to-gpu")
    o = torch.empty(z, sq, d, device="cuda", dtype=torch.float32).contiguous()
    lse = torch.empty(z, sq, 1, device="cuda", dtype=torch.float32).contiguous()
    kern.run([q, k, v, o, lse], scalar=scale)
    torch.cuda.synchronize()

    # fp32 per-chunk reference: each chunk is a local attention over its keys.
    qf = q.float()
    rel_o, rel_lse = 0.0, 0.0
    for zi in range(z):
        s = torch.matmul(qf, k[zi].float().t()) * scale          # [Sq, Skc]
        m = s.max(dim=1, keepdim=True).values                    # [Sq,1]
        e = torch.exp(s - m)
        l = e.sum(dim=1, keepdim=True)                           # [Sq,1]
        o_ref = torch.matmul(e, v[zi].float()) / l              # [Sq,D]
        lse_ref = (m + torch.log(l)).squeeze(1)                  # [Sq]
        rel_o = max(rel_o, _rel(o[zi], o_ref))
        rel_lse = max(rel_lse, _rel(lse[zi, :, 0], lse_ref))

    print(f"\nsplitk_fwd z={z} sq={sq} skc={skc} d={d} grid={kern.grid} "
          f"block={kern.block} smem={kern.shared}B: rel_O={rel_o:.3e} "
          f"rel_LSE={rel_lse:.3e}")
    assert rel_o < 2e-2, f"partial O rel={rel_o:.3e}"
    assert rel_lse < 2e-2, f"partial LSE rel={rel_lse:.3e}"


# --------------------------------------------------------------------------- #
# (2) split-K REDUCE: merge packed partials, vs an fp32 LSE-merge reference.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s,mq,d", [(2, 128, 64), (8, 256, 64)])
def test_splitk_reduce_pk(s, mq, d):
    torch.manual_seed(1)
    # Random but realistic partials: O in [-1,1], LSE spread over a few units.
    osk = torch.randn(s, mq, d, device="cuda", dtype=torch.float32).contiguous()
    lsk = (torch.randn(s, mq, 1, device="cuda", dtype=torch.float32) * 2.0
           ).contiguous()

    kern = _Kernel(_splitk_reduce_mlir(s, mq, d),
                   "-convert-intent-splitk-reduce-to-gpu")
    out = torch.empty(mq, d, device="cuda", dtype=torch.float32).contiguous()
    kern.run([osk, lsk, out])
    torch.cuda.synchronize()

    # fp32 reference merge: lse_max over splits; w_s = exp(LSE_s - lse_max).
    lse_max = lsk.max(dim=0).values                              # [Mq,1]
    w = torch.exp(lsk - lse_max[None])                          # [S,Mq,1]
    sumexp = w.sum(dim=0)                                        # [Mq,1]
    num = (osk * w).sum(dim=0)                                   # [Mq,D]
    ref = num / sumexp
    rel = _rel(out, ref)
    print(f"\nsplitk_reduce s={s} mq={mq} d={d} grid={kern.grid} "
          f"block={kern.block}: rel={rel:.3e}")
    assert rel < 1e-3, f"reduce rel={rel:.3e}"


_XF_SPLITK = ("/home/kingdom/phdworks/Dataset/xformers/xformers/ops/fmha/"
              "_triton/splitk_kernels.py")


def _load_xformers_splitk():
    """Load the raw xformers split-K kernels (source file, no install).

    The module top does ``from xformers.triton.vararg_kernel import
    unroll_varargs, VAR_ARGS_ARRAY``; xformers is not pip-installed, so stub that
    module (the plain ``_splitK_reduce`` we exercise does not call unroll_varargs
    or use VAR_ARGS_ARRAY -- only the *_varargs siblings do).
    """
    import importlib.util
    import sys
    import types
    names = ("xformers", "xformers.triton", "xformers.triton.vararg_kernel")
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    try:
        # Keep the source-only import shim local to this helper. Leaving these
        # stubs in sys.modules makes later tests mistake the lambda below for
        # xformers' real unroll_varargs implementation, corrupting the Triton
        # JIT signature in an order-dependent way.
        sys.modules["xformers"] = types.ModuleType("xformers")
        sys.modules["xformers.triton"] = types.ModuleType("xformers.triton")
        vk = types.ModuleType("xformers.triton.vararg_kernel")
        vk.unroll_varargs = lambda fn, N=1: fn
        vk.VAR_ARGS_ARRAY = list
        sys.modules["xformers.triton.vararg_kernel"] = vk

        spec = importlib.util.spec_from_file_location("xf_splitk", _XF_SPLITK)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


# Real-xformers head-to-head for the REDUCE deliverable: the plain @triton.jit
# `_splitK_reduce` consumes exactly our packed [S,Mq,D] / [S,Mq] layout (single
# attention -> B=G=H=1). Correctness rel < 1e-3 vs our cubin + the fp32 merge,
# plus a kernel-only median latency datapoint. Memory-bound -> parity floor.
@pytest.mark.parametrize("s,mq,d", [(2, 128, 64), (8, 256, 64)])
def test_splitk_reduce_vs_xformers(s, mq, d):
    triton = pytest.importorskip("triton")
    try:
        xf = _load_xformers_splitk()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"xformers split-K source not importable (flagged): {e}")

    torch.manual_seed(1)
    osk = torch.randn(s, mq, d, device="cuda", dtype=torch.float32).contiguous()
    lsk = (torch.randn(s, mq, 1, device="cuda", dtype=torch.float32) * 2.0
           ).contiguous()

    # ours
    kern = _Kernel(_splitk_reduce_mlir(s, mq, d),
                   "-convert-intent-splitk-reduce-to-gpu")
    out = torch.empty(mq, d, device="cuda", dtype=torch.float32).contiguous()
    kern.run([osk, lsk, out])
    torch.cuda.synchronize()

    # real xformers _splitK_reduce. Layouts: Out_splitK[B,G,H,split_k,Mq,K],
    # LSE_splitK[B,G,H,split_k,Mq], Out[B,H,M,K] (single attention: B=G=H=1).
    lsk2 = lsk.squeeze(-1).contiguous()                  # [S, Mq]
    out_xf = torch.empty(mq, d, device="cuda", dtype=torch.float32).contiguous()
    lse_xf = torch.empty(mq, device="cuda", dtype=torch.float32).contiguous()
    splitK_pow2 = triton.next_power_of_2(s)
    head_dim_pow_2 = triton.next_power_of_2(d)

    def xformers():
        xf._splitK_reduce[(mq, 1, 1)](
            osk, lsk2, out_xf, lse_xf,
            split_k=s, splitK_pow2=splitK_pow2,
            stride_osk_z=s * mq * d, stride_osk_g=s * mq * d,
            stride_osk_h=s * mq * d, stride_osk_s=mq * d, stride_osk_m=d,
            stride_osk_k=1,
            stride_lsek_z=s * mq, stride_lsek_g=s * mq, stride_lsek_h=s * mq,
            stride_lsek_s=mq, stride_lsek_m=1,
            stride_oz=mq * d, stride_og=mq * d, stride_oh=mq * d,
            stride_om=d, stride_ok=1,
            stride_lse_z=mq, stride_lse_g=mq, stride_lse_h=mq, stride_lse_m=1,
            head_dim=d, head_dim_pow_2=head_dim_pow_2, H=1, G=1, WRITE_LSE=False)

    xformers()
    torch.cuda.synchronize()

    rel_xf = _rel(out, out_xf)
    lse_max = lsk.max(dim=0).values
    w = torch.exp(lsk - lse_max[None])
    ref = (osk * w).sum(dim=0) / w.sum(dim=0)
    rel_ref = _rel(out, ref)

    tm, tx = _bench(lambda: kern.run([osk, lsk, out])), _bench(xformers)
    gb = (s * mq * d + s * mq + mq * d) * 4 / 1e9  # read partials + LSE, write O
    # HONEST perf framing (no gate): correct + runs + bit-faithful, but NOT at
    # parity here. The reduce is memory-bound; at these tiny working sets
    # (< 1 MB) do_bench medians are launch/occupancy-dominated and our minimal
    # one-block-per-row x D-thread geometry (Mq*D threads ~= a few % of SM
    # capacity) underfills the GPU. Closing it = fold the (b,h,g) decode axis
    # into the grid like real split-K + cp.async (the shared M8b tuning path).
    print(f"\nsplitk_reduce vs xformers s={s} mq={mq} d={d}: rel_xf={rel_xf:.3e} "
          f"rel_fp32={rel_ref:.3e}  mine {tm*1e3:.1f}us ({gb/(tm/1e3):.0f} GB/s) "
          f"xformers {tx*1e3:.1f}us ({gb/(tx/1e3):.0f} GB/s)  mine/xf {tm/tx:.2f} "
          f"[correct+runs; perf gap, untuned geometry -> M8b]")
    assert rel_xf < 1e-3, f"vs xformers={rel_xf:.3e}"
    assert rel_ref < 1e-3, f"vs fp32 ref={rel_ref:.3e}"


# --------------------------------------------------------------------------- #
# THE CROSS-CHECK the task demands: splitK_fwd partials fed through splitK_reduce
# == full attention == torch SDPA. Plus a perf datapoint (kernel-only median).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sq,d,split_k,skc", [
    (128, 64, 2, 64),    # numKT == 1 per chunk
    (256, 64, 4, 64),
    (128, 64, 2, 128),   # numKT == 2 per chunk x batched z-offset (sk_full=256)
])
def test_splitk_fwd_reduce_cross_check(sq, d, split_k, skc):
    sk_full = skc * split_k         # divide the full key sequence into split_k chunks
    scale = 1.0 / (d ** 0.5)

    torch.manual_seed(2)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    k_full = torch.randn(sk_full, d, device="cuda", dtype=torch.float16).contiguous()
    v_full = torch.randn(sk_full, d, device="cuda", dtype=torch.float16).contiguous()
    # Chunk the keys/values along the sequence -> the split-K leading dim.
    k = k_full.view(split_k, skc, d).contiguous()
    v = v_full.view(split_k, skc, d).contiguous()

    fwd = _Kernel(_splitk_fwd_mlir(split_k, sq, skc, d),
                  "-convert-intent-loop-contraction-to-gpu")
    red = _Kernel(_splitk_reduce_mlir(split_k, sq, d),
                  "-convert-intent-splitk-reduce-to-gpu")

    o_part = torch.empty(split_k, sq, d, device="cuda", dtype=torch.float32)
    lse_part = torch.empty(split_k, sq, 1, device="cuda", dtype=torch.float32)
    out = torch.empty(sq, d, device="cuda", dtype=torch.float32)

    def run_both():
        fwd.run([q, k, v, o_part, lse_part], scalar=scale)
        red.run([o_part, lse_part, out])

    run_both()
    torch.cuda.synchronize()

    # (a) fp32 reference full attention over the concatenated keys.
    s = torch.matmul(q.float(), k_full.float().t()) * scale
    p = torch.softmax(s, dim=-1)
    ref32 = torch.matmul(p, v_full.float())
    rel_ref = _rel(out, ref32)

    # (b) torch SDPA (the library kernel).
    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q[None, None], k_full[None, None], v_full[None, None], scale=scale)[0, 0]
    rel_sdpa = _rel(out, sdpa)

    # (c) the M8a full flash (one chunk = the whole sequence), same lowering.
    full = _Kernel(_splitk_fwd_mlir(1, sq, sk_full, d),
                   "-convert-intent-loop-contraction-to-gpu")
    o_full = torch.empty(1, sq, d, device="cuda", dtype=torch.float32)
    lse_full = torch.empty(1, sq, 1, device="cuda", dtype=torch.float32)
    full.run([q, k_full.view(1, sk_full, d).contiguous(),
              v_full.view(1, sk_full, d).contiguous(), o_full, lse_full],
             scalar=scale)
    torch.cuda.synchronize()
    rel_m8a = _rel(out, o_full[0])

    tm = _bench(run_both)
    print(f"\nsplitk fwd+reduce sq={sq} d={d} split_k={split_k}: "
          f"rel_fp32={rel_ref:.3e} rel_sdpa={rel_sdpa:.3e} rel_M8a={rel_m8a:.3e} "
          f"fwd+reduce kernel-only {tm*1e3:.1f}us")
    assert rel_ref < 2e-2, f"vs fp32 ref={rel_ref:.3e}"
    assert rel_sdpa < 2e-2, f"vs SDPA={rel_sdpa:.3e}"
    assert rel_m8a < 2e-2, f"vs M8a full flash={rel_m8a:.3e}"
