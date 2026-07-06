"""M9b HARD GATE: flash-bwd dual-mode DQ equivalence (serial == atomic == ref).

M9a lowered the FlashAttention backward with SERIAL DQ (one block per (b,h),
read-modify-writing pre-zeroed global DQ across an outer key loop). M9b adds the
ATOMIC mode (SEQUENCE_PARALLEL=true): grid parallel over key blocks
(grid.x = cdiv(Sk,BN)); each program owns ONE key block and accumulates its dQ
contribution into pre-zeroed global DQ via memref.atomic_rmw addf
(-> red.global.add.f32 on sm_120a), unlocking parallelism over key blocks. The
mode is selected by the pass option `dq-mode={serial,atomic}`.

THE HARD GATE (PRD M9): serial-DQ == atomic-DQ == fp32 torch-autograd reference
to rel < 1e-3, across >= 3 seeds. The float reassociation of the cross-program
atomic adds must stay in tolerance. dK/dV (partitioned by key block, no atomics)
must also match across modes and the reference.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_flash_bwd_dualmode_pk.py -q -s
"""

from __future__ import annotations

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
    QT = f"tensor<{sq}x{d}xf16>"
    KT = f"tensor<{sk}x{d}xf16>"
    Qt = f"tensor<64x{d}xf16>"
    M1q = f"tensor<{sq}x1xf32>"
    M1t = "tensor<64x1xf32>"
    ST = f"tensor<64x{sk}xf32>"
    GK = f"tensor<{sk}x{d}xf32>"
    GQ = f"tensor<64x{d}xf32>"
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
    def __init__(self, mlir_text: str, dq_mode: str):
        pass_arg = "-convert-intent-attention-backward-to-gpu"
        if dq_mode != "serial":
            pass_arg = f"-convert-intent-attention-backward-to-gpu=dq-mode={dq_mode}"
        rr = _run([INTENT_OPT, pass_arg], stdin=mlir_text)
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
                self.plan.append(("scale",))
            else:
                raise RuntimeError(f"flash-bwd: unhandled launch-arg {op.ty}")

        # R-FLASH-1: cached launcher (see test_flash_fwd_mma_pk). Correctness
        # harnesses synchronize explicitly after run().
        from intentir.backend.orchestrate import CachedLauncher
        self._cached = CachedLauncher(self.func, self.grid, self.block,
                                      self.plan, shared=self.shared)

    def run(self, q, k, v, do, lse, delta, scale, dq, dk, dv):
        self._cached.launch([q, k, v, do, lse, delta, dq, dk, dv], scale)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def _run_mode(kern, sq, sk, d, q, k, v, do, lse, delta, scale):
    dq = torch.zeros(sq, d, device="cuda", dtype=torch.float32).contiguous()
    dk = torch.empty(sk, d, device="cuda", dtype=torch.float32).contiguous()
    dv = torch.empty(sk, d, device="cuda", dtype=torch.float32).contiguous()
    kern.run(q, k, v, do, lse, delta, scale, dq, dk, dv)
    torch.cuda.synchronize()
    return dq, dk, dv


@pytest.mark.parametrize("sq,sk,d", [
    (128, 128, 64),   # numKT = 2 key blocks -> atomic contention across 2 progs
    (256, 256, 64),   # numKT = 4 key blocks
])
def test_flash_bwd_dualmode_gate(sq, sk, d):
    intent = _bwd_intent(sq, sk, d)
    serial = _FlashBwdKernel(intent, "serial")
    atomic = _FlashBwdKernel(intent, "atomic")
    scale = 1.0 / (d ** 0.5)

    print(f"\n[dual-mode gate] ({sq},{sk},{d}) serial grid={serial.grid} "
          f"atomic grid={atomic.grid}")
    assert serial.grid[0] == 1, "serial: one block per head"
    assert atomic.grid[0] == sk // 64, "atomic: grid.x == cdiv(Sk,BN)"

    max_sa_dq = max_sa_dk = max_sa_dv = 0.0
    max_ref_dq = max_ref_dk = max_ref_dv = 0.0
    seeds = [0, 1, 2, 7, 13]
    for seed in seeds:
        torch.manual_seed(seed)
        q = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
        k = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
        v = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
        do = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()

        # fp32 autograd reference (f16 inputs upcast to an fp32 master).
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

        sdq, sdk, sdv = _run_mode(serial, sq, sk, d, q, k, v, do, lse, delta, scale)
        adq, adk, adv = _run_mode(atomic, sq, sk, d, q, k, v, do, lse, delta, scale)

        # serial == atomic (the reassociation gate).
        sa_dq, sa_dk, sa_dv = _rel(adq, sdq), _rel(adk, sdk), _rel(adv, sdv)
        # both == fp32 reference.
        ref_dq = max(_rel(sdq, qf.grad), _rel(adq, qf.grad))
        ref_dk = max(_rel(sdk, kf.grad), _rel(adk, kf.grad))
        ref_dv = max(_rel(sdv, vf.grad), _rel(adv, vf.grad))

        max_sa_dq, max_sa_dk, max_sa_dv = (max(max_sa_dq, sa_dq),
                                           max(max_sa_dk, sa_dk),
                                           max(max_sa_dv, sa_dv))
        max_ref_dq, max_ref_dk, max_ref_dv = (max(max_ref_dq, ref_dq),
                                              max(max_ref_dk, ref_dk),
                                              max(max_ref_dv, ref_dv))
        print(f"  seed={seed}: serial==atomic dQ={sa_dq:.2e} dK={sa_dk:.2e} "
              f"dV={sa_dv:.2e} | vs fp32-ref dQ={ref_dq:.2e} dK={ref_dk:.2e} "
              f"dV={ref_dv:.2e}")

    print(f"  WORST serial==atomic: dQ={max_sa_dq:.2e} dK={max_sa_dk:.2e} "
          f"dV={max_sa_dv:.2e}")
    print(f"  WORST vs fp32-ref:    dQ={max_ref_dq:.2e} dK={max_ref_dk:.2e} "
          f"dV={max_ref_dv:.2e}")

    # HARD GATE: serial == atomic to rel < 1e-3 (reassociation must stay in tol).
    assert max_sa_dq < 1e-3, f"serial!=atomic dQ={max_sa_dq:.3e}"
    assert max_sa_dk < 1e-3, f"serial!=atomic dK={max_sa_dk:.3e}"
    assert max_sa_dv < 1e-3, f"serial!=atomic dV={max_sa_dv:.3e}"
    # Both modes correct vs fp32 autograd reference (< 1e-3 on these shapes).
    assert max_ref_dq < 1e-3, f"vs fp32-ref dQ={max_ref_dq:.3e}"
    assert max_ref_dk < 1e-3, f"vs fp32-ref dK={max_ref_dk:.3e}"
    assert max_ref_dv < 1e-3, f"vs fp32-ref dV={max_ref_dv:.3e}"
