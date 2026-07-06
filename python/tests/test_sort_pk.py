"""PK the C6 HARD-tier block-bitonic sort lowering vs torch AND the real
flag_gems kernels.

The last order-stats gap: a single-block BLOCK BITONIC SORT
(`-convert-intent-sort-to-gpu`). One block per row loads its row into a
`[P]`-element dynamic-shared value buffer (P = next_pow2(N)), padding N..P-1 with
a `+inf` (ascending) / `-inf` (descending) SENTINEL so the pad sorts to the END,
then runs the bitonic network — log2(P)*(log2(P)+1)/2 barrier-separated
compare-exchange stages, each a `gpu.barrier` then a paired compare-swap of
partners i and i^j under the direction bit (i & k)==0. For argsort a parallel
i32 index buffer is swapped in lockstep. topk = a NARROWER output extent (first
k sorted entries). All derived from the one `intent.sort` primitive.

Verify (correctness-only):
  sort     values rel == 0 (exact order) vs torch.sort AND flag_gems.sort.
  argsort  the index PERMUTATION reproduces the sorted values (ties unstable,
           like torch's default sort); checked against torch.argsort AND
           flag_gems.sort's returned indices (flag_gems has no argsort entry).
  topk     values exact + indices reproduce, vs torch.topk AND flag_gems.topk.

Both a power-of-two N (1024) and a NON-pow2 N (1000, padded to 1024) are run to
exercise the sentinel padding.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_sort_pk.py -q -s
"""

from __future__ import annotations

import os
import pathlib

import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

torch = pytest.importorskip("torch")
flag_gems = pytest.importorskip("flag_gems")

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


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _threads(N):
    """A power-of-two block <= 256 dividing P = next_pow2(N)."""
    P = _next_pow2(N)
    t = 256
    while t > P or P % t != 0:
        t >>= 1
    return t


def _sort_mlir(name, M, N, W, desc, ret):
    """`ret` in {"values","indices","both"} selects which results the func
    returns (drives the number of output memrefs)."""
    d = "true" if desc else "false"
    ty = (f"(tensor<{M}x{N}xf32>) -> "
          f"(tensor<{M}x{W}xf32>, tensor<{M}x{W}xi32>)")
    sig = {
        "values": (f"-> tensor<{M}x{W}xf32>", "return %v : tensor<{}x{}xf32>".format(M, W)),
        "indices": (f"-> tensor<{M}x{W}xi32>", "return %i : tensor<{}x{}xi32>".format(M, W)),
        "both": (f"-> (tensor<{M}x{W}xf32>, tensor<{M}x{W}xi32>)",
                 "return %v, %i : tensor<{}x{}xf32>, tensor<{}x{}xi32>".format(M, W, M, W)),
    }[ret]
    return f"""
func.func @{name}(%x: tensor<{M}x{N}xf32>) {sig[0]} {{
  %v, %i = intent.sort %x {{descending = {d}, axis = 1 : i64}} : {ty}
  {sig[1]}
}}
"""


@pytest.mark.parametrize("desc", [False, True])
@pytest.mark.parametrize("N", [1024, 1000])
def test_sort_values_pk(desc, N):
    from intentir.benchmarks.row_reduction import RowReductionKernel

    M = 2048
    T = _threads(N)
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda").contiguous()

    ref = torch.sort(x, dim=1, descending=desc).values
    gv, _gi = flag_gems.sort(x, dim=1, descending=desc)

    k = RowReductionKernel(f"sort_{desc}_{N}", {"M": M, "N": N}, T,
                           pass_name="convert-intent-sort-to-gpu",
                           mlir_text=_sort_mlir("sort_row", M, N, N, desc, "values"))
    out = torch.empty(M, N, device="cuda")
    k.thunk([x, out])(); torch.cuda.synchronize()

    rel_ref = float((out - ref).norm() / ref.norm())
    rel_gems = float((out - gv).norm() / gv.norm())
    exact = bool(torch.equal(out, ref))
    print(f"\nsort desc={desc} N={N} T={T}: rel_torch {rel_ref:.2e} "
          f"rel_gems {rel_gems:.2e} exact_torch={exact}")
    assert rel_ref == 0.0        # values exact vs torch.sort
    assert rel_gems == 0.0       # values exact vs the real flag_gems kernel


@pytest.mark.parametrize("desc", [False, True])
@pytest.mark.parametrize("N", [1024, 1000])
def test_argsort_pk(desc, N):
    from intentir.benchmarks.row_reduction import RowReductionKernel

    M = 2048
    T = _threads(N)
    torch.manual_seed(1)
    x = torch.randn(M, N, device="cuda").contiguous()

    k = RowReductionKernel(f"argsort_{desc}_{N}", {"M": M, "N": N}, T,
                           pass_name="convert-intent-sort-to-gpu",
                           mlir_text=_sort_mlir("argsort_row", M, N, N, desc, "indices"))
    out = torch.empty(M, N, dtype=torch.int32, device="cuda")
    k.thunk([x, out])(); torch.cuda.synchronize()

    idx = out.to(torch.int64)
    # 1) valid permutation: each row's indices are a permutation of [0,N).
    perm_ok = bool(torch.equal(torch.sort(idx, dim=1).values,
                               torch.arange(N, device="cuda").expand(M, N)))
    # 2) the permutation REPRODUCES the sorted values (the real correctness gate;
    #    ties are unstable so exact-index match to torch isn't required).
    gathered = torch.gather(x, 1, idx)
    ref_vals = torch.sort(x, dim=1, descending=desc).values
    gems_vals, _ = flag_gems.sort(x, dim=1, descending=desc)  # no flag_gems.argsort
    rel_ref = float((gathered - ref_vals).norm() / ref_vals.norm())
    rel_gems = float((gathered - gems_vals).norm() / gems_vals.norm())
    print(f"\nargsort desc={desc} N={N} T={T}: perm_ok={perm_ok} "
          f"reproduce_torch {rel_ref:.2e} reproduce_gems {rel_gems:.2e}")
    assert perm_ok
    assert rel_ref == 0.0        # index permutation reproduces torch's sorted row
    assert rel_gems == 0.0       # ... and flag_gems.sort's sorted row


@pytest.mark.parametrize("N", [1024, 1000])
@pytest.mark.parametrize("kk", [1, 50])
def test_topk_pk(N, kk):
    from intentir.benchmarks.row_reduction import RowReductionKernel

    M = 2048
    T = _threads(N)
    torch.manual_seed(2)
    x = torch.randn(M, N, device="cuda").contiguous()

    # topk = sort(descending) then keep the first k (largest-first).
    ref = torch.topk(x, kk, dim=1, largest=True, sorted=True)
    gv, gi = flag_gems.topk(x, kk, dim=1, largest=True, sorted=True)

    k = RowReductionKernel(f"topk_{N}_{kk}", {"M": M, "N": N}, T,
                           pass_name="convert-intent-sort-to-gpu",
                           mlir_text=_sort_mlir("topk_row", M, N, kk, True, "both"))
    vout = torch.empty(M, kk, device="cuda")
    iout = torch.empty(M, kk, dtype=torch.int32, device="cuda")
    k.thunk([x, vout, iout])(); torch.cuda.synchronize()

    rel_v_ref = float((vout - ref.values).norm() / ref.values.norm())
    rel_v_gems = float((vout - gv).norm() / gv.norm())
    # indices reproduce the top-k values (ties unstable).
    gathered = torch.gather(x, 1, iout.to(torch.int64))
    rel_i = float((gathered - ref.values).norm() / ref.values.norm())
    print(f"\ntopk N={N} k={kk} T={T}: val_torch {rel_v_ref:.2e} "
          f"val_gems {rel_v_gems:.2e} idx_reproduce {rel_i:.2e}")
    assert rel_v_ref == 0.0      # top-k values exact vs torch.topk
    assert rel_v_gems == 0.0     # ... and vs the real flag_gems.topk
    assert rel_i == 0.0          # indices reproduce the top-k values
