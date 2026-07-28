"""R-FLASH-3: the register-resident flash FORWARD with the LSE (log-sum-exp)
output extension (``-convert-intent-flash-mma-to-gpu=emit-lse=true``).

FlashAttention's backward -- and flag_gems' flash_api / attention modules -- need
the per-query log-sum-exp statistic LSE = m + log(l) (the log-sum-exp of the
SCALED scores) to recompute the softmax without re-reducing. The online-softmax
forward already carries m (rowmax) and l (rowsum) register-resident per row, so
LSE is emitted essentially for free as an appended [Sq,1] f32 destination -- a
loop-shape-only extension of the existing fwd (NOT a new kernel).

Bar: the O output is UNCHANGED (bit-for-bit identical to emit-lse=false, i.e. the
extension is non-perturbing) AND LSE matches the fp32 reference
``logsumexp(scale * Q@K^T, dim=-1)`` at rel < 2e-2, on >= 2 shapes.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_flash_fwd_lse_pk.py -q -s
"""

from __future__ import annotations

import pathlib

import pytest

from intentir.backend.cuda.compile import (CUDA_BIN, INTENT_OPT, MLIR_OPT, _run,
                                       lower_memref_module_to_kernels)
from intentir.backend.cuda.launch import load_module
from intentir.backend.cuda.orchestrate import (PreparedLauncher, _host_func_text,
                                           _map_input_aligned_ptrs,
                                           _parse_constants, _parse_launches,
                                           _parse_signature_blockargs,
                                           _resolve_const_index)

torch = pytest.importorskip("torch")

from test_flash_fwd_mma_pk import _flash_intent, _rel  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                       reason="intent-opt not built"),
    pytest.mark.skipif(not pathlib.Path(MLIR_OPT).exists(),
                       reason="mlir-opt not present"),
    pytest.mark.skipif(not pathlib.Path(CUDA_BIN).exists(),
                       reason="CUDA toolkit not present"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]


class _FlashLseKernel:
    """Compile the flash intent through the emit-lse variant. Inputs order:
    Q,K,V,scale,O,LSE (LSE the appended [Sq,1] f32 destination)."""

    def __init__(self, mlir_text: str, num_stages: int = 2):
        flash_pass = (f"-convert-intent-flash-mma-to-gpu="
                      f"num-stages={num_stages} emit-lse=true")
        rr = _run([INTENT_OPT, flash_pass], stdin=mlir_text)
        kernels = lower_memref_module_to_kernels(rr, scf_to_cf=True)
        if len(kernels) != 1:
            raise RuntimeError(f"flash-lse: expected one kernel, got {len(kernels)}")
        ck = kernels[0]
        self.func = load_module(ck.cubin).function(ck.kernel_name)

        _n, ftext = _host_func_text(ck.binary_mlir)
        _s, _o, ptr_blockargs = _parse_signature_blockargs(ftext)
        aligned = _map_input_aligned_ptrs(ptr_blockargs)
        consts = _parse_constants(ftext)
        sites = _parse_launches(ftext)
        assert len(sites) == 1
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
                raise RuntimeError(f"flash-lse: unhandled launch-arg {op.ty}")
        self._cached = PreparedLauncher(self.func, self.grid, self.block,
                                      self.plan, shared=self.shared)

    def run(self, q, k, v, scale, o, lse):
        self._cached.launch([q, k, v, o, lse], scale)


@pytest.mark.parametrize("sq,sk,d", [(128, 128, 64), (256, 256, 64)])
def test_flash_fwd_lse_pk(sq, sk, d):
    torch.manual_seed(0)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    scale = 1.0 / (d ** 0.5)

    kern = _FlashLseKernel(_flash_intent(sq, sk, d))
    o = torch.empty(sq, d, device="cuda", dtype=torch.float32).contiguous()
    lse = torch.empty(sq, 1, device="cuda", dtype=torch.float32).contiguous()
    kern.run(q, k, v, scale, o, lse)
    torch.cuda.synchronize()

    # References.
    s = torch.matmul(q.float(), k.float().t()) * scale
    ref_o = torch.matmul(torch.softmax(s, dim=-1), v.float())
    ref_lse = torch.logsumexp(s, dim=-1, keepdim=True)

    rel_o = _rel(o, ref_o)
    rel_lse = _rel(lse, ref_lse)
    print(f"\nflash-lse ({sq},{sk},{d}) grid={kern.grid} block={kern.block}: "
          f"rel_O={rel_o:.3e} rel_LSE={rel_lse:.3e}")

    assert rel_o < 2e-2, f"O rel={rel_o:.3e}"
    assert rel_lse < 2e-2, f"LSE rel={rel_lse:.3e}"
