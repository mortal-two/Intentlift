"""R-FLASH-3: the register-resident flash FORWARD with CAUSAL masking
(``-convert-intent-flash-mma-to-gpu=causal=true``) -- the flag_gems _attn_fwd
STAGE=3 diagonal mask expressed as a loop-shape extension of the existing fwd.

Each register-resident S accumulator lane knows its (query-row, key-col) from the
m16n8 PTX layout, so masking score[q,k]=-inf for key>query (lower-triangular) is a
per-element register select in the online softmax -- no new kernel, no shared
round-trip. This is the CORRECTNESS extension: every key tile is still visited and
masked (the STAGE=1 fully-masked-tile skip is not applied), so it is not faster
than non-causal, but it is exact.

Bar: rel < 2e-2 vs (a) fp32 causal reference softmax(mask(scale*Q@K^T))@V and
(b) torch SDPA is_causal=True, on >= 2 shapes (Sq==Sk diagonal).

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_flash_fwd_causal_pk.py -q -s
"""

from __future__ import annotations

import pathlib

import pytest

from intentir.backend.compile import (CUDA_BIN, INTENT_OPT, MLIR_OPT, _run,
                                       lower_memref_module_to_kernels)
from intentir.backend.launch import load_module
from intentir.backend.orchestrate import (CachedLauncher, _host_func_text,
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


class _FlashCausalKernel:
    """Compile the flash intent through the causal variant. Inputs: Q,K,V,scale,O."""

    def __init__(self, mlir_text: str, num_stages: int = 2):
        flash_pass = (f"-convert-intent-flash-mma-to-gpu="
                      f"num-stages={num_stages} causal=true")
        rr = _run([INTENT_OPT, flash_pass], stdin=mlir_text)
        kernels = lower_memref_module_to_kernels(rr, scf_to_cf=True)
        assert len(kernels) == 1
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
                raise RuntimeError(f"flash-causal: unhandled launch-arg {op.ty}")
        self._cached = CachedLauncher(self.func, self.grid, self.block,
                                      self.plan, shared=self.shared)

    def run(self, q, k, v, scale, o):
        self._cached.launch([q, k, v, o], scale)


@pytest.mark.parametrize("sq,sk,d", [(128, 128, 64), (256, 256, 64)])
def test_flash_fwd_causal_pk(sq, sk, d):
    torch.manual_seed(0)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    scale = 1.0 / (d ** 0.5)

    kern = _FlashCausalKernel(_flash_intent(sq, sk, d))
    o = torch.empty(sq, d, device="cuda", dtype=torch.float32).contiguous()
    kern.run(q, k, v, scale, o)
    torch.cuda.synchronize()

    # fp32 causal reference (lower-triangular mask on the scaled scores).
    s = torch.matmul(q.float(), k.float().t()) * scale
    mask = torch.tril(torch.ones(sq, sk, device="cuda")) == 0
    s = s.masked_fill(mask, float("-inf"))
    ref32 = torch.matmul(torch.softmax(s, dim=-1), v.float())

    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None], scale=scale, is_causal=True
    )[0, 0]

    rel_ref, rel_sdpa = _rel(o, ref32), _rel(o, sdpa)
    print(f"\nflash-causal ({sq},{sk},{d}) grid={kern.grid} block={kern.block}: "
          f"rel_fp32={rel_ref:.3e} rel_sdpa={rel_sdpa:.3e}")

    assert rel_ref < 2e-2, f"causal rel vs fp32={rel_ref:.3e}"
    assert rel_sdpa < 2e-2, f"causal rel vs SDPA={rel_sdpa:.3e}"
