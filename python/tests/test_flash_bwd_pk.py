"""PK the fused FlashAttention BACKWARD lowering vs torch autograd (SDPA backward).

The M9a deliverable: the FlashAttention backward expressed as ONE ``intent.loop``
scanning the QUERY axis (scan Q/dO/LSE/Delta, carry the per-key-row gradients
dk[Sk,D], dv[Sk,D]; body = recompute ``P = exp(scale*S - LSE)`` + the 5
contractions S=Q@K^T, dV=P^T@dO, dP=dO@V^T, dK=dS^T@Q, dQ=dS@K) lowered by
``-convert-intent-attention-backward-to-gpu`` to ONE tiled tensor-core
``gpu.launch`` (BM=BN=64, dk/dv + recompute-P/dS resident in dynamic shared, the
5 matmuls on WMMA, dV/dK on the AOp-transpose path). SERIAL DQ: one block per
(b,h) read-modify-writes pre-zeroed global DQ across an outer key loop.

M9a bar = CORRECT + RUNS spill-free on sm_120a: dQ/dK/dV rel < 2e-2 vs torch
autograd (the f16 inputs upcast to an fp32 master, ``O=softmax(scale*Q@K^T)@V``,
``O.backward(dO)``), on >= 1 shape. Perf-tuning / atomic-DQ / split-K / causal is
M9b; a vs-Triton datapoint here is informational only.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_flash_bwd_pk.py -q -s
"""

from __future__ import annotations

import pathlib

import pytest

from intentir.backend.cuda.compile import (CUDA_BIN, INTENT_OPT, MLIR_OPT, _run,
                                       lower_memref_module_to_kernels)
from intentir.backend.cuda.launch import load_module
from intentir.backend.cuda.orchestrate import (_F32Arg, _I64Arg, _PtrArg, _launch,
                                          _host_func_text,
                                          _map_input_aligned_ptrs,
                                          _parse_constants, _parse_launches,
                                          _parse_signature_blockargs,
                                          _resolve_const_index)

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


def _bwd_intent(sq: int, sk: int, d: int) -> str:
    """The fused flash backward as one intent.loop, parameterized on (sq,sk,d).

    Mirrors test/lower-flash-bwd-to-gpu.mlir. Scans Q/dO/LSE/Delta (query axis),
    carries dk/dv; body recomputes P from LSE + the 5 contractions.
    """
    QT = f"tensor<{sq}x{d}xf16>"
    KT = f"tensor<{sk}x{d}xf16>"
    Qt = f"tensor<64x{d}xf16>"
    M1q = f"tensor<{sq}x1xf32>"
    M1t = "tensor<64x1xf32>"
    ST = f"tensor<64x{sk}xf32>"     # S/P/dP/dS tile [BM, Sk]
    GK = f"tensor<{sk}x{d}xf32>"    # dk/dv [Sk, D]
    GQ = f"tensor<64x{d}xf32>"      # dq tile [BM, D]
    return f"""
func.func @flash_bwd(%Q: {QT}, %K: {KT}, %V: {KT}, %dO: {QT},
                     %LSE: {M1q}, %Delta: {M1q}, %scale: f32)
    -> ({GK}, {GK}) {{
  %dk0 = arith.constant dense<0.0> : {GK}
  %dv0 = arith.constant dense<0.0> : {GK}
  %r:2 = intent.loop scan(%Q, %dO, %LSE, %Delta : {QT}, {QT}, {M1q}, {M1q})
                     init(%dk0, %dv0 : {GK}, {GK})
                     {{scan_axes = [0, 0, 0, 0], tile_hint = 64 : i64}}
                     -> {GK}, {GK} do {{
  ^bb0(%Qt: {Qt}, %dOt: {Qt}, %LSEt: {M1t}, %Deltat: {M1t}, %dk: {GK}, %dv: {GK}):
    %S = intent.contraction %Qt, %K {{subscripts = "qd,kd->qk"}} : ({Qt}, {KT}) -> {ST}
    %scsplat = tensor.splat %scale : {ST}
    %Ss = intent.map_binary %S, %scsplat {{fn = "mul"}} : ({ST}, {ST}) -> {ST}
    %Sm = intent.map_binary %Ss, %LSEt {{fn = "sub"}} : ({ST}, {M1t}) -> {ST}
    %P = intent.map_unary %Sm {{fn = "exp"}} : ({ST}) -> {ST}
    %dVc = intent.contraction %P, %dOt {{subscripts = "qk,qd->kd"}} : ({ST}, {Qt}) -> {GK}
    %dvn = intent.map_binary %dv, %dVc {{fn = "add"}} : ({GK}, {GK}) -> {GK}
    %dP = intent.contraction %dOt, %V {{subscripts = "qd,kd->qk"}} : ({Qt}, {KT}) -> {ST}
    %dPd = intent.map_binary %dP, %Deltat {{fn = "sub"}} : ({ST}, {M1t}) -> {ST}
    %Pds = intent.map_binary %P, %dPd {{fn = "mul"}} : ({ST}, {ST}) -> {ST}
    %dS = intent.map_binary %Pds, %scsplat {{fn = "mul"}} : ({ST}, {ST}) -> {ST}
    %dKc = intent.contraction %dS, %Qt {{subscripts = "qk,qd->kd"}} : ({ST}, {Qt}) -> {GK}
    %dkn = intent.map_binary %dk, %dKc {{fn = "add"}} : ({GK}, {GK}) -> {GK}
    %dQc = intent.contraction %dS, %K {{subscripts = "qk,kd->qd"}} : ({ST}, {KT}) -> {GQ}
    intent.yield %dkn, %dvn : {GK}, {GK}
  }}
  return %r#0, %r#1 : {GK}, {GK}
}}
"""


class _FlashBwdKernel:
    """Compile the flash-bwd intent through -convert-intent-attention-backward-to-gpu
    + lower_memref_module_to_kernels(scf_to_cf=True), resolve a bare-ptr launch
    plan. Input ptr order: Q,K,V,dO,LSE,Delta, then appended dQ,dK,dV (scale by
    value). Host pre-zeros dQ.
    """

    def __init__(self, mlir_text: str):
        rr = _run([INTENT_OPT, "-convert-intent-attention-backward-to-gpu"],
                  stdin=mlir_text)
        kernels = lower_memref_module_to_kernels(rr, scf_to_cf=True)
        if len(kernels) != 1:
            raise RuntimeError(f"flash-bwd: expected one kernel, got {len(kernels)}")
        ck = kernels[0]
        self.func = load_module(ck.cubin).function(ck.kernel_name)

        _n, ftext = _host_func_text(ck.binary_mlir)
        _s, _o, ptr_blockargs = _parse_signature_blockargs(ftext)
        aligned = _map_input_aligned_ptrs(ptr_blockargs)
        consts = _parse_constants(ftext)
        sites = _parse_launches(ftext)
        if len(sites) != 1:
            raise RuntimeError(f"flash-bwd: expected one launch, got {len(sites)}")
        site = sites[0]

        class _Shim:
            constants = consts

        self.grid = tuple(_resolve_const_index(_Shim, s) for s in site.grid_ssa)
        self.block = tuple(_resolve_const_index(_Shim, s) for s in site.block_ssa)
        self.shared = 0
        if site.dynamic_shared_ssa is not None:
            self.shared = int(consts[site.dynamic_shared_ssa][0])
            from intentir.backend.cuda.launch import _driver
            cuda = _driver()
            attr = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            cuda.cuFuncSetAttribute(self.func, attr, self.shared)

        # ptr -> input idx (Q,K,V,dO,LSE,Delta,dQ,dK,dV); a block-arg-derived f32
        # is %scale; an f32/i64 const is a baked literal.
        self.plan: list[tuple] = []
        for op in site.args:
            if op.ty.startswith("!llvm.ptr"):
                self.plan.append(("ptr", aligned[op.ssa]))
            elif op.ty in ("i64", "index"):
                self.plan.append(("i64", int(consts[op.ssa][0])))
            elif op.ty == "f32":
                self.plan.append(("scale",))
            else:
                raise RuntimeError(f"flash-bwd: unhandled launch-arg {op.ty}")

    def run(self, q, k, v, do, lse, delta, scale, dq, dk, dv):
        inputs = [q, k, v, do, lse, delta, dq, dk, dv]  # ptr idx order
        call_args = []
        for entry in self.plan:
            kind = entry[0]
            if kind == "ptr":
                call_args.append(_PtrArg(inputs[entry[1]].data_ptr()))
            elif kind == "scale":
                call_args.append(_F32Arg(float(scale)))
            elif kind == "i64":
                call_args.append(_I64Arg(entry[1]))
        _launch(self.func, self.grid, self.block, call_args, shared=self.shared)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("sq,sk,d", [
    (128, 128, 64),
    (256, 128, 64),
])
def test_flash_bwd_pk(sq, sk, d):
    torch.manual_seed(0)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    do = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    scale = 1.0 / (d ** 0.5)

    # fp32 master = the f16 inputs upcast (isolates the f16-matmul error). LSE and
    # Delta are computed from the SAME forward the kernel recomputes P against.
    qf = q.float().requires_grad_(True)
    kf = k.float().requires_grad_(True)
    vf = v.float().requires_grad_(True)
    s = (qf @ kf.t()) * scale
    p = torch.softmax(s, dim=-1)
    o = p @ vf
    o.backward(do.float())

    with torch.no_grad():
        s_det = (q.float() @ k.float().t()) * scale
        lse = torch.logsumexp(s_det, dim=-1, keepdim=True).contiguous()
        p_det = torch.softmax(s_det, dim=-1)
        o_det = p_det @ v.float()
        delta = (o_det * do.float()).sum(-1, keepdim=True).contiguous()

    kern = _FlashBwdKernel(_bwd_intent(sq, sk, d))
    dq = torch.zeros(sq, d, device="cuda", dtype=torch.float32).contiguous()
    dk = torch.empty(sk, d, device="cuda", dtype=torch.float32).contiguous()
    dv = torch.empty(sk, d, device="cuda", dtype=torch.float32).contiguous()
    kern.run(q, k, v, do, lse, delta, scale, dq, dk, dv)
    torch.cuda.synchronize()

    rel_dq = _rel(dq, qf.grad)
    rel_dk = _rel(dk, kf.grad)
    rel_dv = _rel(dv, vf.grad)
    print(f"\nflash-bwd ({sq},{sk},{d}) grid={kern.grid} block={kern.block} "
          f"smem={kern.shared}B: rel_dQ={rel_dq:.3e} rel_dK={rel_dk:.3e} "
          f"rel_dV={rel_dv:.3e}")

    assert rel_dv < 2e-2, f"({sq},{sk},{d}) rel dV={rel_dv:.3e}"
    assert rel_dk < 2e-2, f"({sq},{sk},{d}) rel dK={rel_dk:.3e}"
    assert rel_dq < 2e-2, f"({sq},{sk},{d}) rel dQ={rel_dq:.3e}"
