"""PK the C3 data-dependent-addressing lowering vs torch + the REAL flag_gems.

gather / index_select / embedding / scatter / scatter_add are lowered from atomic
IntentIR (`intent.gather` / `intent.scatter`) through the C++
``-convert-intent-gather-to-gpu`` pass to a block-cooperative sm_120a cubin, then
launched and compared head-to-head against (1) a torch reference and (2) the real
``flag_gems.<op>`` triton kernel. The index stream is exact, so the bar is
bit-faithful (rel < 1e-3; gather/select/embedding are byte-identical).

The new mechanism is a RUNTIME load/store coordinate on `data`: the index value
is read from the integer index memref, index_cast'd, and used as a data
`memref.load` (gather) / `memref.store` / `memref.atomic_rmw addf` (scatter_add)
coordinate. These are read-once / write-once memory-bound kernels — parity is the
honest floor — but this suite is a CORRECTNESS gate (C3 is coverage, not perf).

The intent MLIR is hand-written here (same shape as test/lower-gather-to-gpu.mlir),
not extracted from the corpus. Scatter is IN-PLACE on the (caller-cloned) data
buffer, so the "output" is the mutated data tensor.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_gather_scatter_pk.py -q -s
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

_PASS = "convert-intent-gather-to-gpu"


def _rel(a, b):
    b = b.float()
    n = b.norm()
    return float((a.float() - b).norm() / n) if n > 0 else float((a - b).abs().max())


def _kernel(name, M, N, T, mlir_text):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    return RowReductionKernel(name, {"M": M, "N": N}, T,
                              pass_name=_PASS, mlir_text=mlir_text)


def _flag_gems():
    try:
        import flag_gems
        return flag_gems
    except Exception:
        return None


# ---------------------------------------------------------------------------- #
# gather (torch.gather / numpy take_along_axis), axis=1.
# ---------------------------------------------------------------------------- #

def test_gather_along_axis():
    M, N, T = 2048, 1024, 256
    R = f"tensor<{M}x{N}xf32>"
    mlir = f"""
func.func @gather(%data: {R}, %index: tensor<{M}x{N}xi64>) -> {R} {{
  %g = intent.gather %data, %index {{axis = 1 : i64, mode = "along_axis"}} : ({R}, tensor<{M}x{N}xi64>) -> {R}
  return %g : {R}
}}
"""
    torch.manual_seed(0)
    data = torch.randn(M, N, device="cuda").contiguous()
    index = torch.randint(0, N, (M, N), device="cuda", dtype=torch.int64).contiguous()
    out = torch.empty(M, N, device="cuda")

    k = _kernel("gather", M, N, T, mlir)
    k.thunk([data, index, out])()
    torch.cuda.synchronize()

    ref = torch.gather(data, 1, index)
    assert _rel(out, ref) < 1e-3
    assert torch.equal(out, ref)  # exact

    fg = _flag_gems()
    if fg is not None:
        yg = fg.gather(data, 1, index)
        torch.cuda.synchronize()
        assert _rel(out, yg) < 1e-3
    print(f"\ngather ({M},{N}) T={T}: exact vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# index_select (numpy take), axis=1: out[M,K] = data[:, index].
# ---------------------------------------------------------------------------- #

def test_index_select():
    M, N, K, T = 2048, 2048, 1024, 256
    D = f"tensor<{M}x{N}xf32>"
    O = f"tensor<{M}x{K}xf32>"
    mlir = f"""
func.func @isel(%data: {D}, %index: tensor<{K}xi64>) -> {O} {{
  %g = intent.gather %data, %index {{axis = 1 : i64, mode = "take"}} : ({D}, tensor<{K}xi64>) -> {O}
  return %g : {O}
}}
"""
    torch.manual_seed(0)
    data = torch.randn(M, N, device="cuda").contiguous()
    index = torch.randint(0, N, (K,), device="cuda", dtype=torch.int64).contiguous()
    out = torch.empty(M, K, device="cuda")

    k = _kernel("isel", M, K, T, mlir)
    k.thunk([data, index, out])()
    torch.cuda.synchronize()

    ref = torch.index_select(data, 1, index)
    assert _rel(out, ref) < 1e-3
    assert torch.equal(out, ref)

    fg = _flag_gems()
    if fg is not None:
        ys = fg.index_select(data, 1, index)
        torch.cuda.synchronize()
        assert _rel(out, ys) < 1e-3
    print(f"\nindex_select ({M},{N})->({M},{K}): exact vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# embedding (row gather), take axis=0: out[N,D] = weight[index, :].
# ---------------------------------------------------------------------------- #

def test_embedding():
    V, Dm, Nidx, T = 4096, 512, 2048, 128
    W = f"tensor<{V}x{Dm}xf32>"
    O = f"tensor<{Nidx}x{Dm}xf32>"
    mlir = f"""
func.func @emb(%weight: {W}, %index: tensor<{Nidx}xi64>) -> {O} {{
  %g = intent.gather %weight, %index {{axis = 0 : i64, mode = "take"}} : ({W}, tensor<{Nidx}xi64>) -> {O}
  return %g : {O}
}}
"""
    torch.manual_seed(0)
    weight = torch.randn(V, Dm, device="cuda").contiguous()
    index = torch.randint(0, V, (Nidx,), device="cuda", dtype=torch.int64).contiguous()
    out = torch.empty(Nidx, Dm, device="cuda")

    k = _kernel("emb", Nidx, Dm, T, mlir)
    k.thunk([weight, index, out])()
    torch.cuda.synchronize()

    ref = torch.nn.functional.embedding(index, weight)
    assert _rel(out, ref) < 1e-3
    assert torch.equal(out, ref)

    fg = _flag_gems()
    if fg is not None:
        ye = fg.embedding(weight, index)
        torch.cuda.synchronize()
        assert _rel(out, ye) < 1e-3
    print(f"\nembedding V={V} D={Dm} N={Nidx}: exact vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# scatter (torch.scatter_ / numpy put_along_axis), axis=1, reduce=none.
# In-place on the caller-cloned data buffer. Per-row PERMUTATION index (unique)
# so reduce=none is deterministic (torch's contract for duplicate indices is
# nondeterministic).
# ---------------------------------------------------------------------------- #

def test_scatter_set():
    M, N, T = 2048, 1024, 256
    D = f"tensor<{M}x{N}xf32>"
    mlir = f"""
func.func @scat(%data: {D}, %index: tensor<{M}x{N}xi64>, %src: {D}) -> {D} {{
  %o = intent.scatter %data, %index, %src {{axis = 1 : i64, mode = "along_axis", reduce = "none"}} : ({D}, tensor<{M}x{N}xi64>, {D}) -> {D}
  return %o : {D}
}}
"""
    torch.manual_seed(0)
    base = torch.randn(M, N, device="cuda").contiguous()
    # A random per-row permutation of [0,N) -> unique indices (deterministic set).
    index = torch.argsort(torch.rand(M, N, device="cuda"), dim=1).to(torch.int64).contiguous()
    src = torch.randn(M, N, device="cuda").contiguous()

    data = base.clone().contiguous()  # in-place target
    k = _kernel("scat", M, N, T, mlir)
    k.thunk([data, index, src])()
    torch.cuda.synchronize()

    ref = base.clone().scatter_(1, index, src)
    assert _rel(data, ref) < 1e-3
    assert torch.equal(data, ref)

    fg = _flag_gems()
    if fg is not None:
        ys = fg.scatter(base, 1, index, src)
        torch.cuda.synchronize()
        assert _rel(data, ys) < 1e-3
    print(f"\nscatter ({M},{N}) reduce=none: exact vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# scatter_add (torch.scatter_add_), axis=1, reduce=add -> atomic_rmw addf.
# Random (duplicate-heavy) index -> exercises the atomic accumulation.
# ---------------------------------------------------------------------------- #

def test_scatter_add():
    M, N, K, T = 2048, 1024, 256, 128
    D = f"tensor<{M}x{N}xf32>"
    S = f"tensor<{M}x{K}xf32>"
    mlir = f"""
func.func @scadd(%data: {D}, %index: tensor<{M}x{K}xi64>, %src: {S}) -> {D} {{
  %o = intent.scatter %data, %index, %src {{axis = 1 : i64, mode = "along_axis", reduce = "add"}} : ({D}, tensor<{M}x{K}xi64>, {S}) -> {D}
  return %o : {D}
}}
"""
    torch.manual_seed(0)
    base = torch.randn(M, N, device="cuda").contiguous()
    index = torch.randint(0, N, (M, K), device="cuda", dtype=torch.int64).contiguous()
    src = torch.randn(M, K, device="cuda").contiguous()

    data = base.clone().contiguous()
    k = _kernel("scadd", M, K, T, mlir)  # launch geometry over src cols K
    k.thunk([data, index, src])()
    torch.cuda.synchronize()

    ref = base.clone().scatter_add_(1, index, src)
    assert _rel(data, ref) < 1e-3  # float add reassociation -> tolerance, not exact

    fg = _flag_gems()
    if fg is not None:
        d2 = base.clone().contiguous()
        fg.scatter_add_(d2, 1, index, src)
        torch.cuda.synchronize()
        assert _rel(data, d2) < 1e-3
    print(f"\nscatter_add ({M},{N}) K={K} reduce=add: rel<1e-3 vs torch"
          + ("" if fg is None else " + flag_gems"))
