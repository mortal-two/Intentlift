"""PK the C8 dynamic-shape-output lowering vs torch + the REAL flag_gems.

nonzero / masked_select / masked_scatter are lowered from atomic IntentIR
(`intent.nonzero` / `intent.masked_select` / `intent.masked_scatter`) through the
C++ ``-convert-intent-compact-to-gpu`` pass to a block-cooperative prefix-sum
sm_120a cubin, then launched and compared head-to-head against (1) a torch
reference and (2) the real ``flag_gems.<op>`` kernel.

The NEW mechanism vs every prior cluster is a data-dependent output EXTENT: the
number of true mask / nonzero elements (`nnz`) is unknown at compile time. The
compaction is an EXCLUSIVE PREFIX-SUM of the flattened 0/1 mask (T contiguous
segments; thread t counts the trues in all prior segments = its base slot, then
re-scans its own segment emitting each true element at the running slot), which
preserves EXACT row-major order (matches torch.nonzero / torch.masked_select).

How the runtime extent flows: the kernel writes into a MAX-sized (`Ntotal`)
destination and the last thread stores the total `nnz` into a `count` buffer; the
host reads `count` and slices `[:nnz]` (single-launch). We ALSO cross-check the
two-step form (host reduces `nnz` from the mask up front). Either way the device
buffer's leading dim is concrete at launch — no dynamically-sized allocation.

The intent MLIR is hand-written here (same shape as test/lower-compact-to-gpu.mlir),
not extracted from the corpus. Masks are passed as f32 (0.0/1.0) to the kernel
(truth test is `!= 0`); torch/flag_gems use the bool mask.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_dynamic_shape_pk.py -q -s
"""

from __future__ import annotations

import os
import pathlib

import pytest

from intentir.backend.cuda.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

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

_PASS = "convert-intent-compact-to-gpu"


def _kernel(name, T, mlir_text):
    # The compact pass now emits THREE gpu.launches (count, scan, scatter) that
    # share a device scratch; CompactKernel loads all three cubins, allocates the
    # scratch, and launches them in stream order.
    from intentir.backend.cuda.kernel_driver import CompactKernel
    return CompactKernel(name, T, mlir_text)


def _flag_gems():
    try:
        import flag_gems
        return flag_gems
    except Exception:
        return None


def _sorted_rows(t):
    """Lexicographically sort the rows of a [K, ndim] int tensor (value-set cmp)."""
    a = t.detach().cpu()
    if a.numel() == 0:
        return a
    keys = a[:, 0]
    for j in range(1, a.shape[1]):
        keys = keys * (a[:, j].max().item() + 1) + a[:, j]
    order = torch.argsort(keys, stable=True)
    return a[order]


# ---------------------------------------------------------------------------- #
# nonzero (torch.nonzero / numpy.argwhere): [R,C] -> [nnz, 2] row-major indices.
# ---------------------------------------------------------------------------- #

def test_nonzero():
    R, C, T = 64, 64, 256
    Nt = R * C
    F = f"tensor<{R}x{C}xf32>"
    mlir = f"""
func.func @nz(%cond: {F}) -> tensor<?x2xi64> {{
  %r = intent.nonzero %cond : ({F}) -> tensor<?x2xi64>
  return %r : tensor<?x2xi64>
}}
"""
    torch.manual_seed(0)
    # ~40% nonzero, a known-count structured mask + random zeros.
    cond = torch.randn(R, C, device="cuda")
    cond = torch.where(torch.rand(R, C, device="cuda") < 0.4, cond,
                       torch.zeros_like(cond)).contiguous()
    nnz = int((cond != 0).sum().item())

    out = torch.zeros(Nt, 2, device="cuda", dtype=torch.int64).contiguous()
    count = torch.zeros(1, device="cuda", dtype=torch.int64).contiguous()

    k = _kernel("nz", T, mlir)
    k.thunk([cond, out, count])()
    torch.cuda.synchronize()

    # Runtime extent flows through `count`; host slices [:nnz].
    assert int(count.item()) == nnz
    got = out[:nnz]

    ref = torch.nonzero(cond)  # [nnz, 2], row-major
    assert torch.equal(got, ref)  # EXACT row-major order

    fg = _flag_gems()
    if fg is not None:
        yz = fg.nonzero(cond)
        torch.cuda.synchronize()
        assert torch.equal(_sorted_rows(got), _sorted_rows(yz))  # index set
    print(f"\nnonzero ({R},{C}) nnz={nnz} T={T}: EXACT row-major vs torch"
          + ("" if fg is None else " + flag_gems (index set)"))


# ---------------------------------------------------------------------------- #
# masked_select (torch.masked_select / data[mask]): [R,C] -> [nnz] row-major.
# ---------------------------------------------------------------------------- #

def test_masked_select():
    R, C, T = 64, 64, 256
    Nt = R * C
    F = f"tensor<{R}x{C}xf32>"
    mlir = f"""
func.func @msel(%data: {F}, %mask: {F}) -> tensor<?xf32> {{
  %r = intent.masked_select %data, %mask : ({F}, {F}) -> tensor<?xf32>
  return %r : tensor<?xf32>
}}
"""
    torch.manual_seed(1)
    data = torch.randn(R, C, device="cuda").contiguous()
    maskb = (torch.rand(R, C, device="cuda") < 0.35)
    maskf = maskb.to(torch.float32).contiguous()  # kernel truth = != 0
    nnz = int(maskb.sum().item())

    sel = torch.zeros(Nt, device="cuda", dtype=torch.float32).contiguous()
    count = torch.zeros(1, device="cuda", dtype=torch.int64).contiguous()

    k = _kernel("msel", T, mlir)
    k.thunk([data, maskf, sel, count])()
    torch.cuda.synchronize()

    assert int(count.item()) == nnz
    got = sel[:nnz]

    ref = torch.masked_select(data, maskb)  # [nnz], row-major
    assert torch.equal(got, ref)  # EXACT row-major order

    fg = _flag_gems()
    if fg is not None:
        ys = fg.masked_select(data, maskb)
        torch.cuda.synchronize()
        # value set (sorted) — flag_gems order is row-major too, but compare as a
        # set to stay robust to any ordering choice.
        assert torch.equal(torch.sort(got).values, torch.sort(ys).values)
    print(f"\nmasked_select ({R},{C}) nnz={nnz} T={T}: EXACT row-major vs torch"
          + ("" if fg is None else " + flag_gems (value set)"))


# ---------------------------------------------------------------------------- #
# masked_scatter (torch.Tensor.masked_scatter): data-shaped, src filled in order.
# ---------------------------------------------------------------------------- #

def test_masked_scatter():
    R, C, T = 64, 64, 256
    Nt = R * C
    F = f"tensor<{R}x{C}xf32>"
    S = f"tensor<{Nt}xf32>"
    mlir = f"""
func.func @mscat(%data: {F}, %mask: {F}, %src: {S}) -> {F} {{
  %r = intent.masked_scatter %data, %mask, %src : ({F}, {F}, {S}) -> {F}
  return %r : {F}
}}
"""
    torch.manual_seed(2)
    data = torch.randn(R, C, device="cuda").contiguous()
    maskb = (torch.rand(R, C, device="cuda") < 0.5)
    maskf = maskb.to(torch.float32).contiguous()
    src = torch.arange(Nt, device="cuda", dtype=torch.float32).contiguous()

    out = torch.zeros(R, C, device="cuda", dtype=torch.float32).contiguous()

    k = _kernel("mscat", T, mlir)
    k.thunk([data, maskf, src, out])()
    torch.cuda.synchronize()

    ref = data.clone().masked_scatter(maskb, src)  # src consumed row-major
    rel = float((out - ref).norm() / ref.norm())
    assert rel < 1e-3
    assert torch.equal(out, ref)  # exact (pure movement)

    fg = _flag_gems()
    if fg is not None:
        yg = fg.masked_scatter(data, maskb, src)
        torch.cuda.synchronize()
        assert float((out - yg).norm() / yg.norm()) < 1e-3
    print(f"\nmasked_scatter ({R},{C}) nnz={int(maskb.sum())} T={T}: "
          "exact vs torch" + ("" if fg is None else " + flag_gems"))
