"""PK the xformers split-K reduce BACKWARD lowering (M9b deliverable).

`_splitK_reduce_varargs_backward` is the GRADIENT of the online log-sum-exp merge
that combines the split-K partial outputs. Given the packed partials
(Out_splitK[S,Mq,D], LSE_splitK[S,Mq,1]) and the merged-output gradients
(Out[Mq,D], LSE[Mq,1], DOut[Mq,D], DLSE[Mq,1]), it redistributes them back to
each split chunk's gradient (dOut_splitK[S,Mq,D], dLSE_splitK[S,Mq,1]):

    lse_max = max_s LSE_splitK[s]
    w_s            = exp(LSE_splitK[s] - lse_max) / exp(LSE - lse_max)
    dOut_splitK[s] = w_s * DOut
    dLSE_splitK[s] = w_s * DLSE + sum_d((Out_splitK[s] - Out) * w_s * DOut)

Lowered by `-convert-intent-splitk-reduce-backward-to-gpu` (a straight-line
map/reduce DAG -> ONE block-cooperative gpu.launch, grid=(Mq,1,1), block=(D,1,1),
two scf.for over the split axis, the corr head-dim sum a gpu.all_reduce). All
f32 -> rel < 1e-3 (typically ~1e-6) vs an fp32 reference AND, if importable, the
real xformers `_splitK_reduce_varargs_backward` kernel.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_splitk_reduce_bwd_pk.py -q -s
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


class _Kernel:
    """Compile one func through `pass_name` + lower_memref_module_to_kernels,
    resolve a bare-ptr launch. run(tensors) passes memrefs in signature order."""

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


def _bwd_mlir(s: int, mq: int, d: int) -> str:
    OSK = f"tensor<{s}x{mq}x{d}xf32>"
    LSK = f"tensor<{s}x{mq}x1xf32>"
    LMX = f"tensor<1x{mq}x1xf32>"
    OUT = f"tensor<{mq}x{d}xf32>"
    L1 = f"tensor<{mq}x1xf32>"
    DSK = f"tensor<{s}x{mq}x1xf32>"
    return f"""
func.func @splitk_reduce_bwd(
    %Osk: {OSK}, %Lsk: {LSK}, %Out: {OUT}, %Lse: {L1},
    %dOut: {OUT}, %dLse: {L1}) -> ({OSK}, {DSK}) {{
  %lmax = intent.reduce %Lsk {{fn = "max", axes = [0], assoc = "associative"}} : ({LSK}) -> {LMX}
  %a = intent.map_binary %Lsk, %lmax {{fn = "sub"}} : ({LSK}, {LMX}) -> {LSK}
  %ea = intent.map_unary %a {{fn = "exp"}} : ({LSK}) -> {LSK}
  %b = intent.map_binary %Lse, %lmax {{fn = "sub"}} : ({L1}, {LMX}) -> {LMX}
  %eb = intent.map_unary %b {{fn = "exp"}} : ({LMX}) -> {LMX}
  %w = intent.map_binary %ea, %eb {{fn = "div"}} : ({LSK}, {LMX}) -> {LSK}
  %doutsk = intent.map_binary %w, %dOut {{fn = "mul"}} : ({LSK}, {OUT}) -> {OSK}
  %diff = intent.map_binary %Osk, %Out {{fn = "sub"}} : ({OSK}, {OUT}) -> {OSK}
  %t = intent.map_binary %diff, %doutsk {{fn = "mul"}} : ({OSK}, {OSK}) -> {OSK}
  %corr = intent.reduce %t {{fn = "sum", axes = [2], assoc = "float_nondeterministic"}} : ({OSK}) -> {DSK}
  %wdl = intent.map_binary %w, %dLse {{fn = "mul"}} : ({LSK}, {L1}) -> {DSK}
  %dlsesk = intent.map_binary %wdl, %corr {{fn = "add"}} : ({DSK}, {DSK}) -> {DSK}
  return %doutsk, %dlsesk : {OSK}, {DSK}
}}
"""


def _rel(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-30))


def _fp32_reference(osk, lsk, out, lse, dout, dlse):
    """The split-K reduce backward, exactly as the kernel computes it (fp32)."""
    lse_max = lsk.max(dim=0, keepdim=True).values            # [1,Mq,1]
    denom = torch.exp(lse - lse_max[0])                       # [Mq,1] == sumexp
    w = torch.exp(lsk - lse_max) / denom[None]               # [S,Mq,1]
    dout_sk = w * dout[None]                                  # [S,Mq,D]
    corr = ((osk - out[None]) * dout_sk).sum(dim=2, keepdim=True)  # [S,Mq,1]
    dlse_sk = w * dlse[None] + corr                          # [S,Mq,1]
    return dout_sk, dlse_sk


def _make_inputs(s, mq, d, seed):
    torch.manual_seed(seed)
    osk = torch.randn(s, mq, d, device="cuda", dtype=torch.float32).contiguous()
    lsk = (torch.randn(s, mq, 1, device="cuda", dtype=torch.float32) * 2.0
           ).contiguous()
    # merged forward outputs (the realistic inputs the backward consumes).
    lse_max = lsk.max(dim=0, keepdim=True).values
    wf = torch.exp(lsk - lse_max)
    sumexp = wf.sum(dim=0)
    out = ((osk * wf).sum(dim=0) / sumexp).contiguous()       # [Mq,D]
    lse = (lse_max[0] + torch.log(sumexp)).contiguous()       # [Mq,1]
    dout = torch.randn(mq, d, device="cuda", dtype=torch.float32).contiguous()
    dlse = torch.randn(mq, 1, device="cuda", dtype=torch.float32).contiguous()
    return osk, lsk, out, lse, dout, dlse


# --------------------------------------------------------------------------- #
# (1) vs fp32 reference.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s,mq,d", [(2, 128, 64), (4, 128, 64), (8, 256, 64)])
def test_splitk_reduce_bwd_pk(s, mq, d):
    osk, lsk, out, lse, dout, dlse = _make_inputs(s, mq, d, seed=3)
    kern = _Kernel(_bwd_mlir(s, mq, d),
                   "-convert-intent-splitk-reduce-backward-to-gpu")
    dout_sk = torch.zeros(s, mq, d, device="cuda", dtype=torch.float32).contiguous()
    dlse_sk = torch.zeros(s, mq, 1, device="cuda", dtype=torch.float32).contiguous()
    kern.run([osk, lsk, out, lse, dout, dlse, dout_sk, dlse_sk])
    torch.cuda.synchronize()

    ref_do, ref_dl = _fp32_reference(osk, lsk, out, lse, dout, dlse)
    rel_do = _rel(dout_sk, ref_do)
    rel_dl = _rel(dlse_sk, ref_dl)
    print(f"\nsplitk_reduce_bwd s={s} mq={mq} d={d} grid={kern.grid} "
          f"block={kern.block}: rel_dOut_splitK={rel_do:.3e} "
          f"rel_dLSE_splitK={rel_dl:.3e}")
    assert rel_do < 1e-3, f"dOut_splitK rel={rel_do:.3e}"
    assert rel_dl < 1e-3, f"dLSE_splitK rel={rel_dl:.3e}"


# --------------------------------------------------------------------------- #
# (2) vs the REAL xformers _splitK_reduce_varargs_backward (single attention,
# B=G=H=1; each split chunk is one tensor in the varargs list). Needs the real
# xformers.triton.vararg_kernel.unroll_varargs codegen; if not importable, this
# test skips with a flag and (1) remains the correctness anchor.
# --------------------------------------------------------------------------- #
_XF_SPLITK = ("/home/kingdom/phdworks/Dataset/xformers/xformers/ops/fmha/"
              "_triton/splitk_kernels.py")


def _load_xformers_splitk_real():
    """Import the xformers split-K kernels WITH the real vararg_kernel module."""
    import importlib.util
    import sys
    xf_root = "/home/kingdom/phdworks/Dataset/xformers"
    if xf_root not in sys.path:
        sys.path.insert(0, xf_root)
    # Import the real vararg_kernel (provides unroll_varargs + VAR_ARGS_ARRAY).
    import importlib
    vk = importlib.import_module("xformers.triton.vararg_kernel")
    spec = importlib.util.spec_from_file_location("xf_splitk_real", _XF_SPLITK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, vk


@pytest.mark.parametrize("s,mq,d", [(2, 128, 64), (4, 128, 64)])
def test_splitk_reduce_bwd_vs_xformers(s, mq, d):
    triton = pytest.importorskip("triton")
    try:
        xf, vk = _load_xformers_splitk_real()
        bwd = vk.unroll_varargs(xf._splitK_reduce_varargs_backward, N=s)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"xformers varargs backward not runnable (flagged): {e}")

    osk, lsk, out, lse, dout, dlse = _make_inputs(s, mq, d, seed=3)

    # ours
    kern = _Kernel(_bwd_mlir(s, mq, d),
                   "-convert-intent-splitk-reduce-backward-to-gpu")
    dout_sk = torch.zeros(s, mq, d, device="cuda", dtype=torch.float32).contiguous()
    dlse_sk = torch.zeros(s, mq, 1, device="cuda", dtype=torch.float32).contiguous()
    kern.run([osk, lsk, out, lse, dout, dlse, dout_sk, dlse_sk])
    torch.cuda.synchronize()

    # real xformers: per-split tensors as varargs lists. Layouts (B=G=H=1):
    # Out_splitK[i] -> [Mq,D], LSE_splitK[i] -> [Mq]; merged Out[Mq,D], LSE[Mq].
    out_sk_list = [osk[i].contiguous() for i in range(s)]
    lse_sk_list = [lsk[i, :, 0].contiguous() for i in range(s)]
    dout_sk_xf = [torch.zeros(mq, d, device="cuda", dtype=torch.float32) for _ in range(s)]
    dlse_sk_xf = [torch.zeros(mq, device="cuda", dtype=torch.float32) for _ in range(s)]
    out_x = out.contiguous()
    lse_x = lse[:, 0].contiguous()
    dout_x = dout.contiguous()
    dlse_x = dlse[:, 0].contiguous()

    args = []
    args += out_sk_list + lse_sk_list + dout_sk_xf + dlse_sk_xf
    args += [out_x, lse_x, dout_x, dlse_x]
    # strides for each chunk (contiguous [Mq,D] / [Mq]): z/g/h collapse to 0.
    args += [0] * s + [0] * s + [0] * s + [d] * s + [1] * s  # osk z,g,h,m,k
    args += [0] * s + [0] * s + [0] * s + [1] * s            # lsek z,g,h,m
    args += [0, 0, 0, d, 1]                                  # out z,g,h,m,k
    args += [0, 0, 0, 1]                                     # lse z,g,h,m
    args += [0, 0, 0, d, 1]                                  # dout z,g,h,m,k
    args += [0, 0, 0, 1]                                     # dlse z,g,h,m
    bwd[(mq, 1, 1)](*args, BLOCK_SIZE=d, H=1, G=1)
    torch.cuda.synchronize()

    rel_do = max(_rel(dout_sk[i], dout_sk_xf[i]) for i in range(s))
    rel_dl = max(_rel(dlse_sk[i, :, 0], dlse_sk_xf[i]) for i in range(s))
    ref_do, ref_dl = _fp32_reference(osk, lsk, out, lse, dout, dlse)
    rel_do_ref = _rel(dout_sk, ref_do)
    rel_dl_ref = _rel(dlse_sk, ref_dl)
    print(f"\nsplitk_reduce_bwd vs xformers s={s} mq={mq} d={d}: "
          f"rel_dOut(xf)={rel_do:.3e} rel_dLSE(xf)={rel_dl:.3e}  "
          f"rel_dOut(fp32)={rel_do_ref:.3e} rel_dLSE(fp32)={rel_dl_ref:.3e}")
    assert rel_do < 1e-3, f"vs xformers dOut={rel_do:.3e}"
    assert rel_dl < 1e-3, f"vs xformers dLSE={rel_dl:.3e}"
