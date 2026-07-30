"""PK the REGISTER-RESIDENT FlashAttention FORWARD (M8c) vs fp32/SDPA/scalar.

The M8c deliverable: the same fused-flash intent.loop as M8a, but lowered by
``-convert-intent-flash-mma-to-gpu`` to ONE tensor-core ``gpu.launch`` that keeps
the online-softmax state (O/S/m/l) RESIDENT IN REGISTERS via ``nvgpu.mma.sync``
(only Q/K/V touch shared). The per-row rowmax/rowsum are warp ``gpu.shuffle``
reductions aligned to the m16n8 acc lane layout; P is assembled into the PV A
operand straight from the S accumulators (no shared P round-trip).

Bar = CORRECT + RUNS on sm_120a: rel < 2e-2 vs (a) fp32 softmax(Q@K^T*scale)@V,
(b) torch SDPA, (c) the scalar online-softmax flash, on >= 2 shapes. This file
mirrors test_flash_fwd_pk.py (which PK's the M8a shared-resident path, kept 100%
intact) so both can be run side by side.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_flash_fwd_mma_pk.py -q -s
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


def _flash_intent(sq: int, sk: int, d: int) -> str:
    QT = f"tensor<{sq}x{d}xf16>"
    KT = f"tensor<{sk}x{d}xf16>"
    Kj = f"tensor<64x{d}xf16>"
    M1 = f"tensor<{sq}x1xf32>"
    OT = f"tensor<{sq}x{d}xf32>"
    SC = f"tensor<{sq}x{d}xf32>"
    return f"""
func.func @flash_fwd(%Q: {QT}, %K: {KT}, %V: {KT}, %scale: f32) -> {OT} {{
  %m0 = arith.constant dense<0xFF800000> : {M1}
  %l0 = arith.constant dense<0.0> : {M1}
  %O0 = arith.constant dense<0.0> : {OT}
  %r:3 = intent.loop scan(%K, %V : {KT}, {KT})
                     init(%m0, %l0, %O0 : {M1}, {M1}, {OT})
                     {{scan_axes = [0, 0], tile_hint = 64 : i64}}
                     -> {M1}, {M1}, {OT} do {{
  ^bb0(%Kj: {Kj}, %Vj: {Kj}, %m: {M1}, %l: {M1}, %Oacc: {OT}):
    %S = intent.contraction %Q, %Kj {{subscripts = "qd,kd->qk"}} : ({QT}, {Kj}) -> {SC}
    %scsplat = tensor.splat %scale : {SC}
    %Ss = intent.map_binary %S, %scsplat {{fn = "mul"}} : ({SC}, {SC}) -> {SC}
    %mb = intent.reduce %Ss {{fn = "max", axes = [1], assoc = "associative"}} : ({SC}) -> {M1}
    %mn = intent.map_binary %m, %mb {{fn = "max"}} : ({M1}, {M1}) -> {M1}
    %Sc = intent.map_binary %Ss, %mn {{fn = "sub"}} : ({SC}, {M1}) -> {SC}
    %P = intent.map_unary %Sc {{fn = "exp"}} : ({SC}) -> {SC}
    %co = intent.map_binary %m, %mn {{fn = "sub"}} : ({M1}, {M1}) -> {M1}
    %al = intent.map_unary %co {{fn = "exp"}} : ({M1}) -> {M1}
    %ps = intent.reduce %P {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({SC}) -> {M1}
    %ls = intent.map_binary %l, %al {{fn = "mul"}} : ({M1}, {M1}) -> {M1}
    %ln = intent.map_binary %ls, %ps {{fn = "add"}} : ({M1}, {M1}) -> {M1}
    %PV = intent.contraction %P, %Vj {{subscripts = "qk,kd->qd"}} : ({SC}, {Kj}) -> {SC}
    %Os = intent.map_binary %Oacc, %al {{fn = "mul"}} : ({OT}, {M1}) -> {OT}
    %On = intent.map_binary %Os, %PV {{fn = "add"}} : ({OT}, {SC}) -> {OT}
    intent.yield %mn, %ln, %On : {M1}, {M1}, {OT}
  }}
  %O = intent.map_binary %r#2, %r#1 {{fn = "div"}} : ({OT}, {M1}) -> {OT}
  return %O : {OT}
}}
"""


class _FlashKernel:
    """Compile the flash intent through -convert-intent-flash-mma-to-gpu +
    lower_memref_module_to_kernels(scf_to_cf=True). Inputs order: Q,K,V,scale,O.
    """

    def __init__(self, mlir_text: str, num_stages: int | None = None):
        flash_pass = "-convert-intent-flash-mma-to-gpu"
        if num_stages is not None:
            flash_pass += f"=num-stages={num_stages}"
        rr = _run([INTENT_OPT, flash_pass], stdin=mlir_text)
        kernels = lower_memref_module_to_kernels(rr, scf_to_cf=True)
        if len(kernels) != 1:
            raise RuntimeError(f"flash: expected one kernel, got {len(kernels)}")
        ck = kernels[0]
        self.cubin = ck.cubin
        self.kernel_name = ck.kernel_name
        self.func = load_module(ck.cubin).function(ck.kernel_name)

        _n, ftext = _host_func_text(ck.binary_mlir)
        _s, _o, ptr_blockargs = _parse_signature_blockargs(ftext)
        aligned = _map_input_aligned_ptrs(ptr_blockargs)
        consts = _parse_constants(ftext)
        sites = _parse_launches(ftext)
        if len(sites) != 1:
            raise RuntimeError(f"flash: expected one gpu.launch_func, got {len(sites)}")
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

        self.plan: list[tuple] = []
        for op in site.args:
            if op.ty.startswith("!llvm.ptr"):
                self.plan.append(("ptr", aligned[op.ssa]))
            elif op.ty in ("i64", "index"):
                self.plan.append(("i64", int(consts[op.ssa][0])))
            elif op.ty == "f32":
                self.plan.append(("scale",))
            else:
                raise RuntimeError(f"flash: unhandled launch-arg {op.ty}")

        # R-FLASH-1: a CACHED launcher (pre-marshalled arg array, updated in
        # place, no per-call rebuild, no per-call synchronize) -- Triton-style.
        # This kills the small-shape launch-overhead gap so flash-fwd reaches
        # parity at ALL shapes, not just the GPU-bound large ones. Correctness
        # harnesses synchronize explicitly after run().
        from intentir.backend.cuda.orchestrate import PreparedLauncher
        self._cached = PreparedLauncher(self.func, self.grid, self.block,
                                      self.plan, shared=self.shared)

    def run(self, q, k, v, scale: float, o):
        self._cached.launch([q, k, v, o], scale)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("sq,sk,d", [
    (128, 128, 64),
    (256, 256, 64),
])
def test_flash_fwd_mma_pk(sq, sk, d):
    torch.manual_seed(0)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    scale = 1.0 / (d ** 0.5)

    kern = _FlashKernel(_flash_intent(sq, sk, d))
    o = torch.empty(sq, d, device="cuda", dtype=torch.float32).contiguous()
    kern.run(q, k, v, scale, o)
    torch.cuda.synchronize()

    s = torch.matmul(q.float(), k.float().t()) * scale
    p = torch.softmax(s, dim=-1)
    ref32 = torch.matmul(p, v.float())
    rel_ref = _rel(o, ref32)

    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None], scale=scale
    )[0, 0]
    rel_sdpa = _rel(o, sdpa)

    from intentir.backend.cuda.flash import run_fused_flash
    scalar = run_fused_flash(q.float().contiguous(), k.float().contiguous(),
                             v.float().contiguous(), scale, bk=16)
    rel_scalar = _rel(o, scalar)

    print(f"\nflash-mma ({sq},{sk},{d}) grid={kern.grid} block={kern.block} "
          f"smem={kern.shared}B: rel_fp32={rel_ref:.3e} rel_sdpa={rel_sdpa:.3e} "
          f"rel_scalar={rel_scalar:.3e}")

    assert rel_ref < 2e-2, f"({sq},{sk},{d}) rel vs fp32 ref={rel_ref:.3e}"
    assert rel_sdpa < 2e-2, f"({sq},{sk},{d}) rel vs SDPA={rel_sdpa:.3e}"
    assert rel_scalar < 2e-2, f"({sq},{sk},{d}) rel vs scalar flash={rel_scalar:.3e}"
