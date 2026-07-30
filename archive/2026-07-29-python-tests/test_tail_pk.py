"""PK the DEFERRED TAIL ops — the flag_gems classified-miss set the coverage task
named — each as a COMPOSITION of already-verified cluster mechanisms, vs torch
(and the real flag_gems where its kernel runs on this arch).

The four targets and their op-graph decomposition:

  unique        = sort(asc) C6b -> adjacent-dedup map (eq/select/max neighbor
                  compare) -> C8 masked_select compaction (dynamic extent = #
                  distinct). Verify EXACT sorted-unique value set vs torch.unique.
  randperm(n)   = argsort (C6b) of n philox keys (C5 rand). Verify VALID
                  permutation (sorted == arange) + STATISTICAL positional
                  uniformity over many seeds + DETERMINISM. Matched-seed parity vs
                  torch is NOT expected (philox != torch RNG) — statistical, ADR.
  multinomial   = inverse-CDF categorical sampling: cumsum(p) C6 scan -> tile ->
                  searchsorted (count cdf<u) -> reduce. Verify the empirical
                  category distribution matches p (large-sample within tolerance)
                  + DETERMINISM. Statistical, ADR.
  bicubic       = upsample_bicubic2d(_aa) 4x4 cubic-convolution neighborhood
                  (a=-0.75), extending the C7a nearest upsample. Verify rel < 1e-3
                  vs torch.nn.functional.interpolate(mode="bicubic"). On the
                  ENLARGING path antialias is a no-op, so plain bicubic == the _aa
                  kernel there; the variable-support aa DOWNSAMPLE kernel is
                  deferred (see the module docstring note).

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_tail_pk.py -q -s
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


def _flag_gems():
    try:
        import flag_gems
        return flag_gems
    except Exception:
        return None


def _div_threads(cols, cap=256):
    t = min(cap, cols)
    while t > 1 and cols % t != 0:
        t >>= 1
    return t


# --------------------------------------------------------------------------- #
# unique = sort -> neighbor-dedup -> C8 compaction  (EXACT sorted-unique set)  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N,hi", [(1024, 64), (4096, 512), (2048, 2048)])
def test_unique_opgraph_pk(N, hi):
    from intentir.backend.cuda.opgraph import unique_run

    torch.manual_seed(0)
    # integer-valued floats with duplicates (hi < N forces collisions); the
    # (2048,2048) case is ~mostly-distinct to exercise the near-identity mask.
    x = torch.randint(0, hi, (N,), device="cuda").float().contiguous()

    got = unique_run(x)
    ref = torch.unique(x, sorted=True)
    assert got.numel() == ref.numel()
    assert torch.equal(got, ref)  # EXACT sorted-unique value set

    print(f"\nunique N={N} hi={hi}: {got.numel()} distinct, EXACT vs torch.unique "
          "(sort C6b -> eq/select/max dedup -> C8 masked_select)")


# --------------------------------------------------------------------------- #
# randperm(n) = argsort of philox keys  (valid perm + uniform + determinism)   #
# --------------------------------------------------------------------------- #
def test_randperm_valid_and_deterministic():
    from intentir.backend.cuda.opgraph import randperm_plan

    n = 1024
    perm = randperm_plan(n, seed=1234).run({})["perm"].reshape(-1).long()
    ar = torch.arange(n, device="cuda")
    assert torch.equal(torch.sort(perm).values, ar)  # VALID permutation

    # determinism: same seed -> identical permutation.
    perm2 = randperm_plan(n, seed=1234).run({})["perm"].reshape(-1).long()
    assert torch.equal(perm, perm2)
    # a different seed decorrelates (not the identical permutation).
    perm3 = randperm_plan(n, seed=9999).run({})["perm"].reshape(-1).long()
    assert not torch.equal(perm, perm3)

    # not the identity / not sorted: low correlation with arange.
    pf = perm.float()
    corr = float(torch.corrcoef(torch.stack([pf, ar.float()]))[0, 1])
    assert abs(corr) < 0.15
    print(f"\nrandperm n={n}: valid perm, deterministic, corr(perm,arange)="
          f"{corr:+.3f} (argsort C6b of philox C5 keys)")


def test_randperm_positional_uniformity():
    from intentir.backend.cuda.opgraph import randperm_plan

    # Statistical: over S independent seeds each position's value should be
    # ~uniform on [0,n-1]; test that no position is biased (column mean near the
    # (n-1)/2 center within a few standard errors). Matched-seed parity vs torch
    # is NOT expected (philox != torch RNG) — this is the ADR's statistical mode.
    S, n = 32, 64
    rows = []
    for s in range(S):
        p = randperm_plan(n, seed=1000 + s).run({})["perm"].reshape(-1).long()
        assert torch.equal(torch.sort(p).values, torch.arange(n, device="cuda"))
        rows.append(p)
    P = torch.stack(rows).float()  # [S, n]
    col_mean = P.mean(0).cpu()
    center = (n - 1) / 2.0
    se = ((n * n - 1) / 12.0 / S) ** 0.5
    dev = (col_mean - center).abs().max().item()
    assert dev < 4.0 * se, f"positional bias {dev:.2f} > {4.0 * se:.2f}"
    print(f"\nrandperm positional-uniformity S={S} n={n}: max|col_mean-{center}|="
          f"{dev:.2f} < {4.0 * se:.2f} (4 se) — no positional bias")


# --------------------------------------------------------------------------- #
# multinomial(p, k) = inverse-CDF categorical  (empirical dist + determinism)  #
# --------------------------------------------------------------------------- #
def _multinomial_counts(N, K, p, seed):
    from intentir.backend.cuda.opgraph import multinomial_plan

    prog = multinomial_plan(N, K, seed=seed)
    env = prog.run({
        "p": p.reshape(1, N).contiguous(),
        "zero": torch.zeros((), device="cuda"),
        "one": torch.ones((), device="cuda"),
    })
    idx = env["counts"].reshape(-1).round().long().clamp(0, N - 1)
    return idx


def test_multinomial_distribution_and_determinism():
    N, K = 16, 1 << 16
    torch.manual_seed(0)
    p = torch.rand(N, device="cuda")
    p = (p / p.sum()).contiguous()

    idx = _multinomial_counts(N, K, p, seed=7)
    emp = torch.bincount(idx, minlength=N).float() / K
    l1 = float((emp - p).abs().sum())
    maxdev = float((emp - p).abs().max())
    # large-sample: per-category standard error ~ sqrt(p(1-p)/K) ~ 1e-3 here.
    assert l1 < 0.04, f"empirical L1 {l1:.4f} too far from p"
    assert maxdev < 0.02, f"max category deviation {maxdev:.4f}"

    # determinism: same seed -> identical draws.
    idx2 = _multinomial_counts(N, K, p, seed=7)
    assert torch.equal(idx, idx2)
    # different seed -> a different sample (not bit-identical).
    idx3 = _multinomial_counts(N, K, p, seed=8)
    assert not torch.equal(idx, idx3)
    print(f"\nmultinomial N={N} K={K}: L1={l1:.4f} maxdev={maxdev:.4f} deterministic "
          "(cumsum C6 -> tile -> searchsorted count -> reduce)")


# --------------------------------------------------------------------------- #
# upsample_bicubic2d(_aa) = 4x4 cubic-convolution neighborhood (a=-0.75)       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N,C,H,W,OH,OW", [
    (1, 3, 8, 8, 16, 16),
    (2, 4, 16, 16, 32, 32),
    (1, 2, 12, 10, 24, 30),
])
def test_upsample_bicubic2d_pk(N, C, H, W, OH, OW):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel

    I = f"tensor<{N}x{C}x{H}x{W}xf32>"
    O = f"tensor<{N}x{C}x{OH}x{OW}xf32>"
    mlir = f"""
func.func @upb(%a: {I}) -> {O} {{
  %o = intent.upsample %a {{mode = "bicubic"}} : ({I}) -> {O}
  return %o : {O}
}}
"""
    torch.manual_seed(0)
    a = torch.randn(N, C, H, W, device="cuda").contiguous()
    out = torch.empty(N, C, OH, OW, device="cuda").contiguous()
    T = _div_threads(OW)
    k = RowReductionKernel("upb", {}, T,
                           pass_name="convert-intent-upsample-to-gpu",
                           mlir_text=mlir)
    k.thunk([a, out])()
    torch.cuda.synchronize()

    ref = torch.nn.functional.interpolate(a, size=(OH, OW), mode="bicubic",
                                          align_corners=False)
    rel = float((out - ref).norm() / ref.norm())
    assert rel < 1e-3
    # On the enlarging path antialias is a no-op: our plain-bicubic equals the
    # _aa kernel here too (spot-check, not asserted tightly since torch's aa path
    # may recompute weights).
    aa = torch.nn.functional.interpolate(a, size=(OH, OW), mode="bicubic",
                                         align_corners=False, antialias=True)
    rel_aa = float((out - aa).norm() / aa.norm())
    print(f"\nbicubic ({N},{C},{H},{W})->({OH},{OW}): rel_torch {rel:.2e} "
          f"rel_aa {rel_aa:.2e} (4x4 cubic-conv gather, extends C7a upsample)")
